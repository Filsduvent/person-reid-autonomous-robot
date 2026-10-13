"""Diagnostic-only durable GPU admission evidence; never initializes CUDA.

The unchanged production guard makes the admission decision. Its exact query
responses are recorded in addition to before/after read-only inventories.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch

from reid.utils import lodo_device as dev

TITAN_UUID = 'GPU-bb0efb3f-f2ea-d40d-8769-e4e989bab10c'
GTX_UUID = 'GPU-3a7eaab1-70ed-9cd0-391a-67a0d8f4e6f5'
MIN_FREE_MIB = 7500  # Same fresh-launch threshold as the preceding diagnostic driver.
FIELDS = 'index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,compute_mode,mig.mode.current'
APPS = 'gpu_uuid,pid,process_name,used_gpu_memory'


def write_once_json(path, value):
    """Atomic publication without replacing even an existing symlink."""
    path = Path(path)
    if path.parent.resolve() != path.parent or path.is_symlink():
        raise ValueError('Aliased diagnostic destination')
    fd, temporary = tempfile.mkstemp(prefix='.admission-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic exclusive publication; EEXIST fails closed.
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        os.unlink(temporary)


def owner(pid):
    result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid=,user=,stat=,args='],
                            capture_output=True, text=True, timeout=10)
    if result.returncode or not result.stdout.strip():
        raise ValueError(f'Cannot establish compute process owner: {pid}')
    return result.stdout.strip()


class AdmissionJournal:
    def __init__(self, directory, *, target_uuid=TITAN_UUID, min_free_mib=MIN_FREE_MIB,
                 query=None, guard=None, owner_lookup=None):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.target_uuid = target_uuid
        if type(min_free_mib) is not int or min_free_mib < MIN_FREE_MIB:
            raise ValueError('Diagnostic headroom threshold cannot be weakened')
        self.min_free_mib = min_free_mib
        self.query = query or dev._query
        self.guard = guard or dev.audit_gpu
        self.owner_lookup = owner_lookup or owner
        self.sequence = 0

    def snapshot(self, record=None):
        record = {} if record is None else record
        record['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
        rows = self.query(FIELDS); record['gpu_rows'] = rows
        apps = self.query(APPS, 'compute-apps'); record['processes'] = apps
        if any(len(r) != 10 for r in rows) or any(len(r) != 4 for r in apps):
            raise ValueError('Incomplete diagnostic inventory')
        selected = [r for r in rows if r[1] == self.target_uuid]
        if len(selected) != 1: raise ValueError('Target UUID missing or ambiguous')
        row = selected[0]
        try: total, used, free, utilization = map(int, row[4:8])
        except ValueError as error: raise ValueError('Missing/non-numeric diagnostic metrics') from error
        if total <= 0 or not 0 <= used <= total or not 0 <= free <= total or not 0 <= utilization <= 100:
            raise ValueError('Invalid diagnostic resource metrics')
        owners = {}
        for app in apps:
            if app[0] == self.target_uuid:
                if not app[1].isdigit(): raise ValueError('Unverifiable compute PID')
                owners[app[1]] = self.owner_lookup(int(app[1]))
                if not owners[app[1]]: raise ValueError('Missing compute ownership')
        record.update(ownership=owners, target=dict(uuid=row[1], total_mib=total,
                      used_mib=used, free_mib=free, utilization_percent=utilization))
        return record

    def check(self, uuid, *, allow_pid=None):
        self.sequence += 1
        record = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), uuid=uuid,
                      visibility=os.environ.get('CUDA_VISIBLE_DEVICES'),
                      authorized_uuid=os.environ.get('LODO_AUTHORIZED_GPU_UUID'),
                      allow_pid=allow_pid, fresh_min_free_mib=self.min_free_mib,
                      required_utilization_percent=0 if allow_pid is None else None,
                      guard_queries=[], decision='REJECT', rejection_reason=None)
        error = None; evidence = None; current_check = 'uuid/visibility'
        original_query = self.query
        def captured(fields, kind='gpu'):
            entry = dict(fields=fields, kind=kind, timestamp_utc=datetime.now(timezone.utc).isoformat())
            record['guard_queries'].append(entry)
            try:
                entry['rows'] = original_query(fields, kind)
                return entry['rows']
            except BaseException as exception:
                entry['error'] = str(exception); raise
        try:
            if uuid != self.target_uuid or uuid != TITAN_UUID:
                raise ValueError('Wrong/excluded UUID for TITAN campaign')
            if record['visibility'] != uuid or record['authorized_uuid'] != uuid:
                raise ValueError('Diagnostic visibility/authorization declaration mismatch')
            current_check = 'before inventory'; record['before'] = {}
            self.snapshot(record['before'])
            current_check = 'unchanged production audit_gpu'
            with patch.object(dev, '_query', captured):
                evidence = self.guard(uuid, allow_pid=allow_pid)
            record['guard_evidence'] = evidence
            current_check = 'fresh diagnostic headroom'
            if allow_pid is None and evidence['free_mib'] < self.min_free_mib:
                raise ValueError(f"Fresh headroom {evidence['free_mib']} MiB below {self.min_free_mib} MiB")
            record['decision'] = 'ACCEPT'
        except BaseException as exception:
            error = exception; record['rejection_reason'] = str(exception)
            record['rejection_check'] = current_check
        finally:
            try:
                record['after'] = {}; self.snapshot(record['after'])
                selected = [p for p in record['after']['processes'] if p[0] == uuid]
                if any(int(p[1]) != allow_pid for p in selected):
                    raise ValueError('Competing process in after inventory')
                metrics = record['after']['target']
                if allow_pid is None and (metrics['free_mib'] < self.min_free_mib or metrics['utilization_percent'] != 0):
                    raise ValueError('After inventory activity/headroom unsafe')
            except BaseException as exception:
                record['after_error'] = str(exception)
                if error is None:
                    error = exception; record['rejection_reason'] = str(exception)
                    record['rejection_check'] = 'after inventory'
            if error is not None: record['decision'] = 'REJECT'
            write_once_json(self.directory / f'{self.sequence:04d}.json', record)
        if error is not None: raise error
        return evidence


@contextmanager
def logged_admission(directory):
    journal = AdmissionJournal(directory)
    with patch.object(dev, 'audit_gpu', journal.check): yield journal


def publish_final_status(path, summary, final_audit):
    """A rejected final audit cannot mask or prevent publication of STOPPED."""
    try: summary['final_audit'] = final_audit()
    except BaseException as error:
        summary['final_audit_error'] = dict(type=type(error).__name__, reason=str(error))
        if summary.get('status') in ('RUNNING', 'COMPLETED'):
            summary['status'] = 'STOPPED'
    finally: write_once_json(path, summary)
