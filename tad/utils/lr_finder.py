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
        """
        Runs the learning rate range test.
        
        Args:
            train_loader (DataLoader): The training data loader.
            start_lr (float): The starting learning rate.
            end_lr (float): The ending learning rate.
            num_iter (int): Number of iterations to run the test.
            smooth_f (float): Smoothing factor for loss.
            diverge_th (float): Threshold for loss divergence to stop the test.
            step_mode (str): 'exp' for exponential increase, 'linear' for linear increase.
            amp (bool): Whether to use mixed precision training.
        """
        self.history = {"lr": [], "loss": []}
        self.best_loss = None

        # Save states to restore later
        if hasattr(self.model, 'module'):
            self.model_state = copy.deepcopy(self.model.module.state_dict())
        else:
            self.model_state = copy.deepcopy(self.model.state_dict())

        self.optimizer_state = copy.deepcopy(self.optimizer.state_dict())

        # Calculate step factor
        if step_mode == "exp":
            gamma = (end_lr / start_lr) ** (1 / num_iter)
        else:
            step_size = (end_lr - start_lr) / num_iter

        # Set initial LR
        for group in self.optimizer.param_groups:
            group["lr"] = start_lr

        self.model.train()

        # AMP scaler initialization
        scaler = None
        if amp:
            # Compatible with PyTorch versions
            if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
                 scaler = torch.amp.GradScaler('cuda')
            elif hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
                scaler = torch.cuda.amp.GradScaler()

        iter_wrapper = iter(train_loader)

        print(f"Starting LR Range Test from {start_lr} to {end_lr} with {num_iter} iterations...")

        for i in range(num_iter):
            try:
                data_dict = next(iter_wrapper)
            except StopIteration:
                iter_wrapper = iter(train_loader)
                data_dict = next(iter_wrapper)

            # Move data to device
            # Note: Users should ensure data_dict structure matches model input
            for k, v in data_dict.items():
                if isinstance(v, torch.Tensor):
                    data_dict[k] = v.to(self.device)
                elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                     data_dict[k] = [x.to(self.device) for x in v]

            self.optimizer.zero_grad()

            # Forward pass
            if amp and scaler is not None:
                with torch.amp.autocast('cuda', enabled=True):
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

            # Update LR
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)

            # Record loss
            loss_val = loss.item()
            if i == 0:
                avg_loss = loss_val
            else:
                avg_loss = smooth_f * loss_val + (1 - smooth_f) * self.history["loss"][-1]

            if i == 0 or avg_loss < self.best_loss:
                self.best_loss = avg_loss

            self.history["loss"].append(avg_loss)

            # Check for divergence
            if i > num_iter // 10 and avg_loss > diverge_th * self.best_loss:
                print(f"Stopping early, the loss has diverged at iter {i}, loss {avg_loss:.4f}")
                break

            # Update next LR
            if step_mode == "exp":
                new_lr = current_lr * gamma
            else:
                new_lr = current_lr + step_size

            for group in self.optimizer.param_groups:
                group["lr"] = new_lr

        print(f"LR Range Test finished. Best smoothed loss: {self.best_loss:.4f}")
        self.reset()

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

            print(f"Suggested LR (steepest gradient): {suggested_lr:.2e}")
            plt.scatter(suggested_lr, suggested_loss, s=75, marker='o', color='red', zorder=10, label=f'Suggested LR: {suggested_lr:.2e}')

            # Find minimum loss point
            min_loss_idx = losses.index(min(losses))
            min_loss_lr = lrs[min_loss_idx]
            min_loss_val = losses[min_loss_idx]
            print(f"Min Loss LR: {min_loss_lr:.2e}")
            plt.scatter(min_loss_lr, min_loss_val, s=75, marker='o', color='red', zorder=10, label=f'Min Loss LR: {min_loss_lr:.2e}')

            # Mark 0.5x and 0.1x Min Loss LR
            for factor in [0.5, 0.1]:
                target_lr = min_loss_lr * factor
                # Find closest available data point
                closest_idx = min(range(len(lrs)), key=lambda i: abs(lrs[i] - target_lr))
                closest_lr = lrs[closest_idx]
                closest_loss = losses[closest_idx]

                print(f"{factor}x Min Loss LR: {closest_lr:.2e}")
                plt.scatter(closest_lr, closest_loss, s=75, marker='o', color='red', zorder=10, label=f'{factor}x Min Loss LR: {closest_lr:.2e}')

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
