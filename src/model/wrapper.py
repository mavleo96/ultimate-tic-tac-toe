import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NetworkWrapper:
    def __init__(self, model: nn.Module, device: str | torch.device = "cpu") -> None:
        self.model = model.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def to_tensors(self, states) -> tuple[torch.Tensor, ...]:
        device = self.device
        return tuple(
            torch.tensor(np.stack([s[i] for s in states]), dtype=torch.long).to(device)
            for i in range(len(states[0]))
        )

    def predict(self, states) -> tuple[np.ndarray, np.ndarray]:
        tensors = self.to_tensors(states)

        self.model.eval()
        with torch.no_grad():
            policy_logits, value = self.model(*tensors)

        priors = F.softmax(policy_logits, dim=1).cpu().numpy().astype(np.float32)
        values = value.reshape(-1).cpu().numpy().astype(np.float32)
        return priors, values
