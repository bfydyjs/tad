import copy

import matplotlib.pyplot as plt
import torch


class LRFinder:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.history = {"lr": [], "loss": []}
        self.best_loss = None
        self.model_state = None
        self.optimizer_state = None

    def range_test(
        self,
        train_loader,
        start_lr=1e-5,
        end_lr=0.01,
        num_iter=100,
        smooth_f=0.05,
        diverge_th=5,
        step_mode="exp",
        amp=True,
    ):
        """Runs the learning rate range test in a simplified, readable flow.

        The implementation delegates complex steps to small helper methods to keep
        cyclomatic complexity low while preserving original behavior.
        """
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        self._save_states()

        if step_mode == "exp":
            gamma = (end_lr / start_lr) ** (1 / num_iter)
            step_size = None
        else:
            gamma = None
            step_size = (end_lr - start_lr) / num_iter

        for group in self.optimizer.param_groups:
            group["lr"] = start_lr

        self.model.train()
        scaler = self._init_scaler(amp)
        iter_wrapper = iter(train_loader)

        print(f"Starting LR Range Test from {start_lr} to {end_lr} with {num_iter} iterations...")

        for i in range(num_iter):
            data_dict, iter_wrapper = self._get_next_batch(iter_wrapper, train_loader)
            self._move_to_device(data_dict)

            self.optimizer.zero_grad()
            loss = self._run_step(data_dict, scaler, amp)

            current_lr = self._record_lr()

            avg_loss = self._update_loss_history(i, loss.item(), smooth_f)

            if i > num_iter // 10 and avg_loss > diverge_th * self.best_loss:
                print(f"Stopping early, the loss has diverged at iter {i}, loss {avg_loss:.4f}")
                break

            new_lr = self._compute_next_lr(current_lr, gamma, step_size, step_mode)
            self._set_lr(new_lr)

        print(f"LR Range Test finished. Best smoothed loss: {self.best_loss:.4f}")
        self.reset()

    # --- Helper methods to reduce complexity in range_test ---
    def _save_states(self):
        if hasattr(self.model, 'module'):
            self.model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())
        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

    def _init_scaler(self, amp):
        if not amp:
            return None
        # Try the most common APIs for GradScaler depending on PyTorch version
        try:
            return torch.cuda.amp.GradScaler()
        except Exception:
            try:
                return torch.amp.GradScaler()
            except Exception:
                return None

    def _get_next_batch(self, iter_wrapper, train_loader):
        try:
            data_dict = next(iter_wrapper)
        except StopIteration:
            iter_wrapper = iter(train_loader)
            data_dict = next(iter_wrapper)
        return data_dict, iter_wrapper

    def _move_to_device(self, data_dict):
        for k, v in list(data_dict.items()):
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.to(self.device)
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                data_dict[k] = [x.to(self.device) for x in v]

    def _run_step(self, data_dict, scaler, amp):
        if amp and scaler is not None:
            # Prefer the cuda autocast if available
            autocast = getattr(torch.cuda.amp, 'autocast', None) or getattr(torch, 'amp', None)
            with autocast(enabled=True):
                losses = self.model(**data_dict, return_loss=True)
                loss = losses['cost']
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
        else:
            losses = self.model(**data_dict, return_loss=True)
            loss = losses['cost']
            loss.backward()
            self.optimizer.step()
        return loss

    def _record_lr(self):
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["lr"].append(current_lr)
        return current_lr

    def _update_loss_history(self, i, loss_val, smooth_f):
        if i == 0:
            avg_loss = loss_val
        else:
            avg_loss = smooth_f * loss_val + (1 - smooth_f) * self.history["loss"][-1]

        if i == 0 or avg_loss < self.best_loss:
            self.best_loss = avg_loss

        self.history["loss"].append(avg_loss)
        return avg_loss

    def _compute_next_lr(self, current_lr, gamma, step_size, step_mode):
        if step_mode == "exp":
            return current_lr * gamma
        else:
            return current_lr + step_size

    def _set_lr(self, new_lr):
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def plot(self, skip_start=10, skip_end=5, log_lr=True, save_path=None):
        if not self.history["lr"]:
            print("No history to plot.")
            return

        # Handle indices to avoid out of bounds
        n = len(self.history["lr"])

        # Adjust skip logic if n is small
        real_skip_start = min(skip_start, n // 3)
        real_skip_end = min(skip_end, n // 3)

        start = max(0, real_skip_start)
        end = n - max(0, real_skip_end)

        if start >= end:
             print("Not enough data points to plot. Plotting all data.")
             start = 0
             end = n

        lrs = self.history["lr"][start:end]
        losses = self.history["loss"][start:end]

        plt.figure(figsize=(10, 6))
        plt.plot(lrs, losses)

        # Calculate gradients to find steepest slope
        if len(losses) > 1:
            grads = [losses[i] - losses[i-1] for i in range(1, len(losses))]
            min_grad_idx = grads.index(min(grads))
            suggested_lr = lrs[min_grad_idx]
            suggested_loss = losses[min_grad_idx]

            print(
                f"Suggested LR (steepest gradient): {suggested_lr:.2e}"
            )
            plt.scatter(
                suggested_lr,
                suggested_loss,
                s=75,
                marker='o',
                color='red',
                zorder=10,
                label=f"Suggested LR: {suggested_lr:.2e}",
            )

            # Find minimum loss point
            min_loss_idx = losses.index(min(losses))
            min_loss_lr = lrs[min_loss_idx]
            min_loss_val = losses[min_loss_idx]
            print(f"Min Loss LR: {min_loss_lr:.2e}")
            plt.scatter(
                min_loss_lr,
                min_loss_val,
                s=75,
                marker='o',
                color='red',
                zorder=10,
                label=f"Min Loss LR: {min_loss_lr:.2e}",
            )

            # Mark 0.5x and 0.1x Min Loss LR
            for factor in [0.5, 0.1]:
                target_lr = min_loss_lr * factor
                # Find closest available data point
                closest_idx = min(
                    range(len(lrs)), key=lambda i: abs(lrs[i] - target_lr)
                )
                closest_lr = lrs[closest_idx]
                closest_loss = losses[closest_idx]

                print(f"{factor}x Min Loss LR: {closest_lr:.2e}")
                plt.scatter(
                    closest_lr,
                    closest_loss,
                    s=75,
                    marker='o',
                    color='red',
                    zorder=10,
                    label=f"{factor}x Min Loss LR: {closest_lr:.2e}",
                )

            plt.legend()

        if log_lr:
            plt.xscale("log")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("LR Range Test")
        plt.grid(True, which="both", ls="-", alpha=0.5)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def reset(self):
        print("Restoring model and optimizer states...")
        if self.model_state:
            if hasattr(self.model, 'module'):
                self.model.module.load_state_dict(self.model_state)
            else:
                self.model.load_state_dict(self.model_state)
        if self.optimizer_state:
            self.optimizer.load_state_dict(self.optimizer_state)
