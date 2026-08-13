"""Deployment export helpers independent of the simulator."""

import copy
from pathlib import Path

import torch


class PolicyExporter(torch.nn.Module):
    """Single inference graph containing adaptation and actor modules."""

    def __init__(self, actor_critic):
        super().__init__()
        self.adaptation_module = copy.deepcopy(actor_critic.adaptation_module)
        self.actor_body = copy.deepcopy(actor_critic.actor_body)

    def forward(self, obs_history):
        latent = self.adaptation_module(obs_history)
        return self.actor_body(torch.cat((obs_history, latent), dim=-1))


def export_policy_as_jit(actor_critic, path, filename="policy.pt"):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    output_path = path / filename
    exporter = PolicyExporter(actor_critic).cpu().eval()
    torch.jit.script(exporter).save(str(output_path))
    return output_path
