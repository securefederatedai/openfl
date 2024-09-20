# Copyright (c) 2012-2022 Clement Vignac, Igor Krawczuk, Antoine Siraudin
# source: https://github.com/cvignac/DiGress/

import numpy as np
import torch
from digress import utils
from digress.diffusion import diffusion_utils


class PredefinedNoiseSchedule(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseSchedule, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            alphas2 = diffusion_utils.cosine_beta_schedule(timesteps)
        elif noise_schedule == 'custom':
            raise NotImplementedError()
        else:
            raise ValueError(noise_schedule)

        # print('alphas2', alphas2)

        sigmas2 = 1 - alphas2

        log_alphas2 = np.log(alphas2)
        log_sigmas2 = np.log(sigmas2)

        log_alphas2_to_sigmas2 = log_alphas2 - log_sigmas2     # (timesteps + 1, )

        # print('gamma', -log_alphas2_to_sigmas2)

        self.gamma = torch.nn.Parameter(
            torch.from_numpy(-log_alphas2_to_sigmas2).float(),
            requires_grad=False)

    def forward(self, t):
        t_int = torch.round(t * self.timesteps).long()
        return self.gamma[t_int]



class PredefinedNoiseScheduleDiscrete(torch.nn.Module):
    """
    Predefined noise schedule. Essentially creates a lookup array for predefined (non-learned) noise schedules.
    """

    def __init__(self, noise_schedule, timesteps):
        super(PredefinedNoiseScheduleDiscrete, self).__init__()
        self.timesteps = timesteps

        if noise_schedule == 'cosine':
            betas = diffusion_utils.cosine_beta_schedule_discrete(timesteps)
        elif noise_schedule == 'custom':
            betas = diffusion_utils.custom_beta_schedule_discrete(timesteps)
        else:
            raise NotImplementedError(noise_schedule)

        self.register_buffer('betas', torch.from_numpy(betas).float())

        self.alphas = 1 - torch.clamp(self.betas, min=0, max=0.9999)

        log_alpha = torch.log(self.alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.alphas_bar = torch.exp(log_alpha_bar)
        # print(f"[Noise schedule: {noise_schedule}] alpha_bar:", self.alphas_bar)

    def forward(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.betas[t_int.long()]

    def get_alpha_bar(self, t_normalized=None, t_int=None):
        assert int(t_normalized is None) + int(t_int is None) == 1
        if t_int is None:
            t_int = torch.round(t_normalized * self.timesteps)
        return self.alphas_bar.to(t_int.device)[t_int.long()]


class DiscreteUniformTransition:
    def __init__(self, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes
        self.u_x = torch.ones(1, self.X_classes, self.X_classes)
        if self.X_classes > 0:
            self.u_x = self.u_x / self.X_classes

        self.u_e = torch.ones(1, self.E_classes, self.E_classes)
        if self.E_classes > 0:
            self.u_e = self.u_e / self.E_classes

        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) / K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class MarginalUniformTransition:
    def __init__(self, x_marginals, e_marginals, y_classes):
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals

        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes

    def get_Qt(self, beta_t, device):
        """ Returns one-step transition matrices for X and E, from step t - 1 to step t.
        Qt = (1 - beta_t) * I + beta_t / K

        beta_t: (bs)                         noise level between 0 and 1
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy). """
        beta_t = beta_t.unsqueeze(1)
        beta_t = beta_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes, device=device).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes, device=device).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes, device=device).unsqueeze(0)

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)

    def get_Qt_bar(self, alpha_bar_t, device):
        """ Returns t-step transition matrices for X and E, from step 0 to step t.
        Qt = prod(1 - beta_t) * I + (1 - prod(1 - beta_t)) * K

        alpha_bar_t: (bs)         Product of the (1 - beta_t) for each time step from 0 to t.
        returns: qx (bs, dx, dx), qe (bs, de, de), qy (bs, dy, dy).
        """
        alpha_bar_t = alpha_bar_t.unsqueeze(1)
        alpha_bar_t = alpha_bar_t.to(device)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        q_x = alpha_bar_t * torch.eye(self.X_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes, device=device).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y)


class AbsorbingStateTransition:
    def __init__(self, abs_state: int, x_classes: int, e_classes: int, y_classes: int):
        self.X_classes = x_classes
        self.E_classes = e_classes
        self.y_classes = y_classes

        self.u_x = torch.zeros(1, self.X_classes, self.X_classes)
        self.u_x[:, :, abs_state] = 1

        self.u_e = torch.zeros(1, self.E_classes, self.E_classes)
        self.u_e[:, :, abs_state] = 1

        self.u_y = torch.zeros(1, self.y_classes, self.y_classes)
        self.u_e[:, :, abs_state] = 1

    def get_Qt(self, beta_t):
        """ Returns two transition matrix for X and E"""
        beta_t = beta_t.unsqueeze(1)
        q_x = beta_t * self.u_x + (1 - beta_t) * torch.eye(self.X_classes).unsqueeze(0)
        q_e = beta_t * self.u_e + (1 - beta_t) * torch.eye(self.E_classes).unsqueeze(0)
        q_y = beta_t * self.u_y + (1 - beta_t) * torch.eye(self.y_classes).unsqueeze(0)
        return q_x, q_e, q_y

    def get_Qt_bar(self, alpha_bar_t):
        """ beta_t: (bs)
        Returns transition matrices for X and E"""

        alpha_bar_t = alpha_bar_t.unsqueeze(1)

        q_x = alpha_bar_t * torch.eye(self.X_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_x
        q_e = alpha_bar_t * torch.eye(self.E_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_e
        q_y = alpha_bar_t * torch.eye(self.y_classes).unsqueeze(0) + (1 - alpha_bar_t) * self.u_y

        return q_x, q_e, q_y
