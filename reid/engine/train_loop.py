import time
import torch
from reid.models.outputs import ensure_output_dict


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
):
    model.train()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp)
    num_steps = len(loader)

    t0 = time.time()
    last_log_time = t0
    running_total = 0.0
    running_logs = {}
    running_acc_id = 0.0
    acc_steps = 0

    for step, (imgs, labels) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if aux_optimizer is not None:
            aux_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=amp):
            outputs = ensure_output_dict(model(imgs))
            loss, logs = criterion(outputs, labels)

        scaler.scale(loss).backward()

        step_aux_optimizer = aux_optimizer is not None
        prepare_aux_step = getattr(criterion, "prepare_auxiliary_optimizer_step", None)
        if step_aux_optimizer and prepare_aux_step is not None:
            step_aux_optimizer = bool(prepare_aux_step())

        scaler.step(optimizer)
        if step_aux_optimizer:
            scaler.step(aux_optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_total += float(loss.detach().cpu())
        for key, value in logs.items():
            if key == "loss/total":
                continue
            running_logs[key] = running_logs.get(key, 0.0) + float(value)
        logits = outputs.get("logits")
        batch_acc = None
        if logits is not None:
            pred = logits.argmax(dim=1)
            batch_acc = float((pred == labels).float().mean().detach().cpu())
            running_acc_id += batch_acc
            acc_steps += 1

        if (step % log_interval) == 0:
            dt = time.time() - t0
            interval_elapsed = time.time() - last_log_time
            batch_size = int(imgs.shape[0])
            interval_steps = min(log_interval, step)
            time_per_batch = interval_elapsed / max(1, interval_steps)
            speed = (batch_size * interval_steps) / max(interval_elapsed, 1e-12)
            last_log_time = time.time()
            avg = running_total / step
            current_lr = optimizer.param_groups[0]["lr"]
            bias_lr = next(
                (
                    group["lr"]
                    for group in optimizer.param_groups[1:]
                    if group["lr"] != current_lr
                ),
                None,
            )
            avg_acc = (running_acc_id / acc_steps) if acc_steps > 0 else None

            msg = (
                f"Epoch [{epoch}] Iter [{step}/{num_steps}] "
                f"loss_total={avg:.4f} lr={current_lr:.6g}"
            )
            if bias_lr is not None:
                msg += f" lr/bias={bias_lr:.6g}"
            if avg_acc is not None:
                msg += f" acc_id={avg_acc:.4f}"
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
                tb_writer.add_scalar("loss/total", avg, global_step=global_step)
                for key, value in logs.items():
                    if key != "loss/total":
                        tb_writer.add_scalar(key, value, global_step=global_step)
                tb_writer.add_scalar("lr", current_lr, global_step=global_step)
                tb_writer.add_scalar("lr/base", current_lr, global_step=global_step)
                tb_writer.add_scalar("time/batch", time_per_batch, global_step=global_step)
                tb_writer.add_scalar("speed/img_per_sec", speed, global_step=global_step)
                if avg_acc is not None:
                    tb_writer.add_scalar("acc/id", avg_acc, global_step=global_step)
                if bias_lr is not None:
                    tb_writer.add_scalar("lr/bias", bias_lr, global_step=global_step)

    return running_total / max(1, num_steps)
