# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# This file overrides the step method of PyTorch Optimizer
# Adopts fix clipping from https://arxiv.org/abs/1710.06963

import torch
from torch.optim import Optimizer

torch.set_printoptions(precision=10)


class ClipOptimizer(object):
    def __init__(
        self, base_optimizer: Optimizer, device, clip_norm: float, clip_freq: int = 1
    ):
        super().__init__()
        self.base_optimizer = base_optimizer
        self.device = device
        self.clip_norm = clip_norm
        self.clip_freq = clip_freq
        self.counter = 0

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def param_groups(self):
        return self.base_optimizer.param_groups

    def step(self, global_model_state, last_iter) -> None:
        self.base_optimizer.step()
        self.counter += 1
        ordered_state_keys = list(global_model_state.keys())

        # perform clipping at the specified frequency and at final iteration
        if self.counter % self.clip_freq == 0 or last_iter:
            # compute delta
            local_params = self.param_groups()[0]["params"]
            delta_params = [
                torch.sub(
                    local_params[idx], global_model_state[ordered_state_keys[idx]]
                )
                for idx in range(len(local_params))
            ]

            # calculate clip factor
            per_param_delta_norm = torch.stack(
                [torch.norm(param, dim=()) for param in delta_params]
            )
            delta_norm = torch.norm(per_param_delta_norm, dim=())
            # 1e-4 is needed below since torch model dicts round to this precision
            clip_factor = (self.clip_norm / (delta_norm + 1e-4)).clamp(max=1.0)

            # clip delta
            updated_params = [
                torch.add(
                    global_model_state[ordered_state_keys[idx]],
                    torch.mul(delta_params[idx], clip_factor),
                )
                for idx in range(len(delta_params))
            ]

            # clipped model
            for idx, param in enumerate(local_params):
                if param.data.shape != updated_params[idx].shape:
                    raise ValueError(
                        f"Trying to update params of shape: {param.data.shape}"
                        + f"with update of shape: {updated_params[idx].shape}"
                    )
                param.data = updated_params[idx]

    def load_state_dict(self, state_dict, **kwargs):
        return self.base_optimizer.load_state_dict(state_dict, **kwargs)

    def zero_grad(self, **kwargs):
        self.base_optimizer.zero_grad(**kwargs)
