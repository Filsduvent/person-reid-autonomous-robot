"""Reviewed GPU evaluator frontend; the original training checkout stays frozen."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def execute(job_path):
    from lodo_console import load_job, release, ENGINE, op
    job = load_job(Path(job_path))
    reviewed = release()
    if job['kind'] != 'evaluation' or reviewed != job['release']:
        raise ValueError('GPU worker requires the exact reviewed evaluation job')
    if op.sha(job['request']) != job['request_sha256']:
        raise ValueError('Pinned GPU request changed')
    from reid.utils.config import load_config
    from reid.utils.lodo_evaluation import evaluate_lodo
    from reid.utils.lodo_operational import evaluation_admission
    request = load_config(job['request'])
    if request.get('device') != 'cuda':
        raise ValueError('GPU worker cannot execute a CPU request')
    admission = evaluation_admission(job['request'], request, root=ENGINE)
    if admission != job['admission'] or admission['status'] != 'ADMITTED_CUDA_EVALUATION':
        raise ValueError('GPU evaluation admission changed')
    base = ENGINE / '.local/lodo_evaluation/v2'
    base.mkdir(parents=True, exist_ok=True)
    audit = Path(tempfile.mkdtemp(prefix='evaluation-', dir=base))
    op.write_private(audit/'execution-admission.json', admission)
    op.write_private(audit/'evaluator-release.json', reviewed)
    result, comparison = evaluate_lodo(request, project_root=ENGINE)
    op.write_private(audit/'completion.json', dict(status='COMPLETED', result=result['artifacts'],
                                                 evaluator_release_sha256=reviewed['sha256']))
    print('[Evaluation] result=' + result['artifacts']['result'], flush=True)
    return result, comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    execute(parser.parse_args().job)
