import time
import torch


def train_one_epoch(model, loader, criterion, optimizer, device, amp: bool, log_interval: int, tb_writer=None, epoch: int = 1, steps_per_epoch: int = 200):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    t0 = time.time()
    running_total = 0.0
    running_triplet = 0.0
    running_id = 0.0
    running_center = 0.0

    for step, (imgs, labels) in enumerate(loader, start=1):
        if steps_per_epoch is not None and step > steps_per_epoch:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(imgs)
            loss, logs = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_total += float(loss.detach().cpu())
        running_triplet += float(logs.get("loss/triplet", 0.0))
        running_id += float(logs.get("loss/id", 0.0))
        running_center += float(logs.get("loss/center", 0.0))

        if (step % log_interval) == 0:
            dt = time.time() - t0
            avg = running_total / step
            msg = f"[Epoch {epoch}] step {step:04d} loss={avg:.4f}"
            if running_triplet > 0.0:
                msg += f" triplet={running_triplet / step:.4f}"
            if running_id > 0.0:
                msg += f" id={running_id / step:.4f}"
            if running_center > 0.0:
                msg += f" center={running_center / step:.4f}"
            msg += f" ({dt:.1f}s)"
            print(msg)

            if tb_writer is not None:
                tb_writer.add_scalar("loss/total", avg, global_step=(epoch * 100000 + step))
                tb_writer.add_scalar("loss/triplet", logs.get("loss/triplet", 0.0), global_step=(epoch * 100000 + step))
                tb_writer.add_scalar("loss/id", logs.get("loss/id", 0.0), global_step=(epoch * 100000 + step))
                tb_writer.add_scalar("loss/center", logs.get("loss/center", 0.0), global_step=(epoch * 100000 + step))

    return running_total / max(1, min(step, steps_per_epoch or step))
