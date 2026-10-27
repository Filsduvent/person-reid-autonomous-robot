import time
import torch
from reid.engine.update_control import (UpdateEpochResult, UpdateFailure, epoch_batches,
                                        finite_gradients, finite_parameters, validate_update_mode)
from reid.models.outputs import ensure_output_dict, validate_logits


@torch.no_grad()
def classification_accuracy(logits, labels):
    """Return detached classification diagnostics; missing logits emit no metric.

    LossBundle owns the error when ID loss requires absent logits. A sequence
    reports the mean of independent head accuracies, never an ensemble prediction.
    """
    if logits is None:
        return {}
    if labels.ndim != 1:
        raise ValueError(f"Expected labels with shape [B], got {tuple(labels.shape)}")
    validate_logits(logits, batch_size=labels.shape[0])
    if torch.is_tensor(logits):
        pred = logits.argmax(dim=1)
        return {"acc/id": float((pred == labels).float().mean().detach().cpu())}
    accuracies = [(head.argmax(dim=1) == labels).float().mean() for head in logits]
    return {"acc/id_mean_heads": float(torch.stack(accuracies).mean().cpu())}


def learning_rate_metrics(optimizer):
    """Read current rates using rule metadata, or preserve historical LR tags."""
    groups = optimizer.param_groups
    if any("group_name" in group for group in groups):
        metrics = {}
        for group in groups:
            if "group_name" not in group:
                raise ValueError("Prefix optimizer groups must all carry group_name metadata.")
            tag = f"lr/{group['group_name']}"
            if tag in metrics and metrics[tag] != group["lr"]:
                raise ValueError(f"Inconsistent learning rates for optimizer group {tag!r}.")
            metrics[tag] = group["lr"]
        return metrics
    current_lr = groups[0]["lr"]
    metrics = {"lr": current_lr, "lr/base": current_lr}
    bias_lr = next((group["lr"] for group in groups[1:]
                    if group["lr"] != current_lr), None)
    if bias_lr is not None:
        metrics["lr/bias"] = bias_lr
    return metrics


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    aux_optimizer,
    device,
    amp: bool,
    log_interval: int,
    scheduler=None,
    tb_writer=None,
    epoch: int = 1,
    logger=None,
    *,
    update_progress=None,
    before_update=None,
):
    # Default Stage A still returns float and uses its original epoch scheduler.
    progress = update_progress
    if before_update is not None and progress is None:
        raise ValueError('Device admission guard is LODO update-mode only')
    if progress is not None:
        validate_update_mode(progress, loader, model, criterion, optimizer, aux_optimizer, scheduler, amp)
        if log_interval <= 0:
            raise ValueError('Update-mode log_interval must be positive')
        before = (progress.attempted_batches, progress.forward_batches, progress.backward_batches,
                  progress.completed_primary_updates, progress.auxiliary_steps,
                  sum(progress.exposure.presentations['forward'].values()),
                  sum(progress.exposure.presentations['successful'].values()))
    model.train()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp)
    num_steps = len(loader)

    t0 = time.time()
    last_log_time = t0
    running_total = 0.0
    running_logs = {}
    running_accuracy = {}
    accuracy_steps = {}
    weighted_logs = {}
    loss_batches = 0
    lr_used = lr_next = None
    batches = epoch_batches(loader, progress)

    try:
        for step, (imgs, labels) in enumerate(batches, start=1):
            if before_update is not None:
                before_update()  # stop on contention before image transfer/forward
            if progress is not None:
                progress.attempted_batches += 1
                progress.attempted_images += int(labels.shape[0])
                progress.index_history.append(loader.latest_indices)
                lookup_start = time.perf_counter()
                try:
                    records = progress.exposure.resolve(loader.latest_indices, labels)
                except Exception:
                    progress.unattributed_attempted_batches += 1
                    progress.unattributed_attempted_images += int(labels.shape[0])
                    raise
                finally:
                    progress.exposure.lookup_seconds += time.perf_counter() - lookup_start
                progress.exposure.record('attempted', records)
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if aux_optimizer is not None:
                aux_optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=amp_device, enabled=amp):
                if progress is not None:
                    progress.forward_batches += 1
                    progress.exposure.record('forward', records)
                outputs = ensure_output_dict(model(imgs))
                loss, logs = criterion(outputs, labels)

            if progress is not None:
                if not torch.isfinite(loss).all().item():
                    raise ValueError('Nonfinite loss before optimizer steps')
                running_total += float(loss.detach().cpu())
                loss_batches += 1
                for key, value in logs.items():
                    if key != 'loss/total':
                        running_logs[key] = running_logs.get(key, 0.0) + float(value)
                        weight = getattr(criterion, 'w_' + key.split('/')[-1], None)
                        if weight is not None:
                            weighted_logs[key] = weighted_logs.get(key, 0.0) + float(value) * weight
            scaler.scale(loss).backward()
            if progress is not None:
                progress.backward_batches += 1
                progress.exposure.record('backward', records)
                finite_gradients(optimizer)
                if aux_optimizer is not None:
                    finite_gradients(aux_optimizer)

            step_aux_optimizer = aux_optimizer is not None
            prepare_aux_step = getattr(criterion, "prepare_auxiliary_optimizer_step", None)
            if step_aux_optimizer and prepare_aux_step is not None:
                step_aux_optimizer = bool(prepare_aux_step())

            if progress is not None:
                if progress.auxiliary and not step_aux_optimizer:
                    raise ValueError('Expected Center step was not prepared')
                finite_gradients(optimizer)
                if aux_optimizer is not None:
                    finite_gradients(aux_optimizer)  # include inverse-weight rescaling
                lr_used = tuple(group['lr'] for group in optimizer.param_groups)
                if lr_used != scheduler.rates_after(progress.completed_primary_updates):
                    raise ValueError('Optimizer LR disagrees with successful-update schedule')
                progress.consistent = False  # any exception from this point may be partial
                optimizer.step()
                progress.main_steps_returned += 1
                finite_parameters(optimizer)
                if step_aux_optimizer:
                    aux_optimizer.step()
                    progress.auxiliary_steps += 1
                    finite_parameters(aux_optimizer)
                progress.completed_primary_updates += 1
                progress.exposure.record('successful', records)
                scheduler.step()  # only advance after the complete main+aux boundary
                lr_next = tuple(group['lr'] for group in optimizer.param_groups)
                progress.consistent = True
                progress.lr_trace.append(dict(update=progress.completed_primary_updates,
                                              lr_used=lr_used, lr_next=lr_next))
            else:
                scaler.step(optimizer)
                if step_aux_optimizer:
                    scaler.step(aux_optimizer)
                scaler.update()
                if scheduler is not None and getattr(scheduler, 'step_interval', 'batch') == 'batch':
                    scheduler.step()

                running_total += float(loss.detach().cpu())
                for key, value in logs.items():
                    if key == "loss/total":
                        continue
                    running_logs[key] = running_logs.get(key, 0.0) + float(value)
            for key, value in classification_accuracy(outputs.get("logits"), labels).items():
                running_accuracy[key] = running_accuracy.get(key, 0.0) + value
                accuracy_steps[key] = accuracy_steps.get(key, 0) + 1

            if (step % log_interval) == 0:
                dt = time.time() - t0
                interval_elapsed = time.time() - last_log_time
                batch_size = int(imgs.shape[0])
                interval_steps = min(log_interval, step)
                time_per_batch = interval_elapsed / max(1, interval_steps)
                speed = (batch_size * interval_steps) / max(interval_elapsed, 1e-12)
                last_log_time = time.time()
                avg = running_total / step
                lr_metrics = learning_rate_metrics(optimizer)
                avg_accuracy = {key: value / accuracy_steps[key]
                                for key, value in running_accuracy.items()}

                msg = f"Epoch [{epoch}] Iter [{step}/{num_steps}] loss_total={avg:.4f}"
                for key, value in lr_metrics.items():
                    if key != "lr/base":  # Historical console omits this duplicate tag.
                        msg += f" {key}={value:.6g}"
                for key, value in avg_accuracy.items():
                    msg += f" {key.replace('/', '_')}={value:.4f}"
                for key in sorted(running_logs):
                    avg_value = running_logs[key] / step
                    if avg_value > 0.0:
                        msg += f" {key.split('/')[-1]}={avg_value:.4f}"
                msg += f" time/batch={time_per_batch:.4f}s speed={speed:.2f} imgs/s elapsed={dt:.1f}s"
                if logger is not None:
                    logger.info(msg)
                else:
                    print(msg)

                if tb_writer is not None:
                    global_step = epoch * 100000 + step
                    if progress is not None:
                        global_step = progress.completed_primary_updates
                        for i, (used, following) in enumerate(zip(lr_used, lr_next)):
                            tb_writer.add_scalar(f'lr/used/group_{i}', used, global_step=global_step)
                            tb_writer.add_scalar(f'lr/next/group_{i}', following, global_step=global_step)
                    tb_writer.add_scalar("loss/total", avg, global_step=global_step)
                    for key, value in logs.items():
                        if key != "loss/total":
                            tb_writer.add_scalar(key, value, global_step=global_step)
                    for key, value in lr_metrics.items():
                        tb_writer.add_scalar(key, value, global_step=global_step)
                    tb_writer.add_scalar("time/batch", time_per_batch, global_step=global_step)
                    tb_writer.add_scalar("speed/img_per_sec", speed, global_step=global_step)
                    for key, value in avg_accuracy.items():
                        tb_writer.add_scalar(key, value, global_step=global_step)

        if progress is not None and not progress.budget_completed and progress.attempted_batches == before[0]:
            raise ValueError('Observed epoch yielded no batches')
    except BaseException as error:
        if progress is None:
            raise
        progress.fail(error)
        if isinstance(error, Exception):
            raise UpdateFailure(str(error), progress) from error
        raise
    finally:
        batches.close()
        if progress is not None:
            after = (progress.attempted_batches, progress.forward_batches, progress.backward_batches,
                     progress.completed_primary_updates, progress.auxiliary_steps,
                     sum(progress.exposure.presentations['forward'].values()),
                     sum(progress.exposure.presentations['successful'].values()))
            delta = tuple(a - b for a, b in zip(after, before))
            reason = 'failed' if progress.failed else ('budget_completed' if progress.budget_completed else 'epoch_exhausted')
            progress.epoch_results.append(UpdateEpochResult(
                epoch=epoch, stop_reason=reason, partial_epoch=delta[0] < num_steps,
                attempted_batches=delta[0], forward_batches=delta[1], backward_batches=delta[2],
                successful_updates=delta[3], auxiliary_steps=delta[4], forward_images=delta[5],
                successful_images=delta[6], loss_batches=loss_batches,
                mean_loss=running_total / loss_batches if loss_batches else None,
                mean_loss_terms={k: v / loss_batches for k, v in running_logs.items()} if loss_batches else {},
                mean_weighted_loss_terms={k: v / loss_batches for k, v in weighted_logs.items()} if loss_batches else {},
                lr_used=lr_used, lr_next=lr_next,
                observation=loader.observation() if delta[0] else {}))
    if progress is not None:
        progress.check_totals()
        return progress.epoch_results[-1]
    return running_total / max(1, num_steps)
