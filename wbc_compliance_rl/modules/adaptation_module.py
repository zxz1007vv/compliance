"""Construction of the concurrent state-estimation network.

The builder intentionally returns ``nn.Sequential`` so the historical
``adaptation_module.<index>.*`` state-dict keys remain unchanged.
"""

import torch.nn as nn


def build_adaptation_module(input_dim, output_dim, hidden_dims, activation):
    layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
    for index, hidden_dim in enumerate(hidden_dims):
        if index == len(hidden_dims) - 1:
            layers.append(nn.Linear(hidden_dim, output_dim))
        else:
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dims[index + 1]), activation]
            )
    return nn.Sequential(*layers)
