from copy import deepcopy

import torch


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        if hasattr(model, "module"):
            model = model.module

        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

    def _update(self, model, update_fn):
        if hasattr(model, "module"):
            model = model.module

        # Ensure EMA model is on the same device as the model
        try:
            device = next(model.parameters()).device
            if next(self.module.parameters()).device != device:
                self.module.to(device)
        except StopIteration:
            pass

        with torch.no_grad():
            # Create a dictionary of model parameters and buffers for name-based lookup
            model_params = dict(model.named_parameters())
            model_buffers = dict(model.named_buffers())

            for name, ema_v in self.module.named_parameters():
                if name in model_params:
                    ema_v.copy_(update_fn(ema_v, model_params[name]))

            for name, ema_v in self.module.named_buffers():
                if name in model_buffers:
                    ema_v.copy_(model_buffers[name])

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
