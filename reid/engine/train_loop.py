import time
import torch

def train_one_epoch(model, loader, criterion, optimizer, device, amp: bool, log_interval: int, tb_writer=None, epoch: int = 1, steps_per_epoch: int = 200):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    t0 = time.time()
    running = 0.0

    for step, (imgs, labels) in enumerate(loader, start=1):
        if steps_per_epoch is not None and step > steps_per_epoch:
            break

        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            emb = model(imgs)
            loss, logs = criterion(emb, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.detach().cpu())

        if (step % log_interval) == 0:
            dt = time.time() - t0
            avg = running / step
            msg = f"[Epoch {epoch}] step {step:04d} loss={avg:.4f} ({dt:.1f}s)"
            print(msg)

            if tb_writer is not None:
                tb_writer.add_scalar("loss/total", avg, global_step=(epoch * 100000 + step))
                if "loss/triplet" in logs:
                    tb_writer.add_scalar("loss/triplet", logs["loss/triplet"], global_step=(epoch * 100000 + step))

    return running / max(1, min(step, steps_per_epoch or step))
