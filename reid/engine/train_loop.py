import time
import torch
from reid.models.outputs import ensure_output_dict


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    center_optimizer,
    device,
    amp: bool,
    log_interval: int,
    scheduler=None,
    tb_writer=None,
    epoch: int = 1,
):
    model.train()
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(amp_device, enabled=amp)
    num_steps = len(loader)

    t0 = time.time()
    running_total = 0.0
    running_triplet = 0.0
    running_id = 0.0
    running_center = 0.0

    for step, (imgs, labels) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if center_optimizer is not None:
            center_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=amp_device, enabled=amp):
            outputs = ensure_output_dict(model(imgs))
            loss, logs = criterion(outputs, labels)

        scaler.scale(loss).backward()

        center_weight = float(getattr(criterion, "w_center", 0.0))
        center_loss = getattr(criterion, "center_loss", None)
        if center_loss is None:
            center_loss = getattr(criterion, "center", None)
        if center_optimizer is not None and center_loss is not None and center_weight > 0.0:
            for param in center_loss.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(1.0 / center_weight)

        scaler.step(optimizer)
        if center_optimizer is not None and center_loss is not None and center_weight > 0.0:
            scaler.step(center_optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        running_total += float(loss.detach().cpu())
        running_triplet += float(logs.get("loss/triplet", 0.0))
        running_id += float(logs.get("loss/id", 0.0))
        running_center += float(logs.get("loss/center", 0.0))

        if (step % log_interval) == 0:
            dt = time.time() - t0
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

            msg = f"[Epoch {epoch}] step {step:04d}/{num_steps:04d} loss={avg:.4f} lr={current_lr:.6g}"
            if bias_lr is not None:
                msg += f" lr/bias={bias_lr:.6g}"
            if running_triplet > 0.0:
                msg += f" triplet={running_triplet / step:.4f}"
            if running_id > 0.0:
                msg += f" id={running_id / step:.4f}"
            if running_center > 0.0:
                msg += f" center={running_center / step:.4f}"
            msg += f" ({dt:.1f}s)"
            print(msg)

            if tb_writer is not None:
                global_step = epoch * 100000 + step
                tb_writer.add_scalar("loss/total", avg, global_step=global_step)
                tb_writer.add_scalar("loss/triplet", logs.get("loss/triplet", 0.0), global_step=global_step)
                tb_writer.add_scalar("loss/id", logs.get("loss/id", 0.0), global_step=global_step)
                tb_writer.add_scalar("loss/center", logs.get("loss/center", 0.0), global_step=global_step)
                tb_writer.add_scalar("lr", current_lr, global_step=global_step)
                if bias_lr is not None:
                    tb_writer.add_scalar("lr/bias", bias_lr, global_step=global_step)

    return running_total / max(1, num_steps)
