# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 13:30:58 2025

@author: thoma
"""

import math
import torch
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':
        betas = 1. / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s)
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if isfunction(d) else d)


# -------------------------------------------------------------------
#  用于采样 Rayleigh 分布并返回其 log(M) 值，以模拟乘性噪声。
#  做了一些 clamp 防止 log(0) 或 exp(∞) 导致 NaN/Inf。
# -------------------------------------------------------------------
def sample_rayleigh_log(shape, scale=1.0, device='cuda'):
    """
    M ~ Rayleigh(scale):
       M = scale * sqrt(-2.0 * log(U)), U ~ Uniform(0,1)

    这里返回 log(M) 供网络直接学习。
    """
    U = torch.rand(shape, device=device)
    # 避免 log(0) / log(1)
    U = torch.clamp(U, min=1e-6, max=1.0 - 1e-6)

    M = scale * torch.sqrt(-2.0 * torch.log(U) + 1e-8)  # Rayleigh
    M = torch.clamp(M, min=1e-8, max=1e6)  # 再次防止出现过大/过小
    logM = torch.log(M + 1e-8)
    return logM


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end']
        )
        if isinstance(betas, torch.Tensor):
            betas = betas.detach().cpu().numpy()

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )

    # ----------------------------------------------------------------
    #  前向扩散改为 乘性噪声:
    #    x_noisy = sqrt_alpha * x_start * exp(log(M)) = sqrt_alpha * x_start * exp(noise).
    #  如果外部不提供 noise(= log(M)), 则用 Rayleigh 分布采样。
    # ----------------------------------------------------------------
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        原SR3加性:
          x_noisy = c * x_start + sqrt(1 - c^2)*noise
        现改为乘性: 
          x_noisy = c * x_start * exp(noise)
        其中 noise = log(M), M~Rayleigh(...).
        """
        if noise is None:
            noise = sample_rayleigh_log(
                shape=x_start.shape,
                scale=1.0,
                device=x_start.device
            )
        # 为避免 exp(∞)，这里做一次 clamp
        noise = torch.clamp(noise, min=-20.0, max=20.0)

        x_noisy = continuous_sqrt_alpha_cumprod * x_start * torch.exp(noise)
        return x_noisy

    # ----------------------------------------------------------------
    #  根据网络预测到的 log(M_t) 还原 x_0:
    #    x_0 = x_t / [ sqrt_recip_alphas_cumprod[t] * exp(log(M_t)) ]
    # ----------------------------------------------------------------
    def predict_start_from_noise(self, x_t, t, noise):
        """
        原DDPM: x_0 = (x_t - sqrt(1-alpha_t)*eps) / sqrt(alpha_t)
        现:     x_0 = x_t / [ (1/ sqrt(alpha_t)) * exp(log(M_t)) ]
               = x_t / [ sqrt_recip_alphas_cumprod[t] * exp(noise) ]
        """
        # clamp 避免极端情况
        noise = torch.clamp(noise, min=-20.0, max=20.0)

        denom = self.sqrt_recip_alphas_cumprod[t] * torch.exp(noise) + 1e-8
        x_0 = x_t / denom
        return x_0

    def q_posterior(self, x_start, x_t, t):
        """
        依然沿用原DDPM的后验近似:
          q(x_{t-1}| x_t, x_0) ~ N(
              posterior_mean_coef1[t]*x_0 + posterior_mean_coef2[t]*x_t,
              posterior_variance[t]
          )
        """
        posterior_mean = self.posterior_mean_coef1[t] * x_start + \
                         self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        """
        后向推断的关键:
         1) 先由网络预测 log(M_t)
         2) 再还原 x_0 = predict_start_from_noise()
         3) 最后套用原DDPM的 q_posterior() 近似
        """
        batch_size = x.shape[0]
        # SR3中，将 t+1 时刻 sqrt_alphas_cumprod_prev 作为 "noise_level" 输入
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]
        ).repeat(batch_size, 1).to(x.device)

        if condition_x is not None:
            # 送入网络: [ SR, x_noisy ], noise_level
            log_m_pred = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            log_m_pred = self.denoise_fn(x, noise_level)

        x_recon = self.predict_start_from_noise(x_t=x, t=t, noise=log_m_pred)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        """
        从 p(x_{t-1}| x_t) 中采样:
           x_{t-1} = model_mean + sigma * noise
        """
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        """
        SR3 的采样流程: 从 t = T 到 t = 0 逐步采样
        """
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps // 10))

        if not self.conditional:
            # 无条件生成
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            # 超分: x_in 是 condition_x (SR)
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        return ret_img if continous else ret_img[-1]

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), continous)

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    # ----------------------------------------------------------------
    #  训练时的核心: p_losses()
    #   - 先采样一个时刻 t, 并在 alpha[t-1], alpha[t] 区间随机取 continuous alpha
    #   - 再采样 log(M) (若没指定 noise)
    #   - 根据乘性方式合成 x_noisy
    #   - 网络预测 log(M)_pred, 与真值做 L1/L2 loss
    # ----------------------------------------------------------------
    def p_losses(self, x_in, noise=None):
        x_start = x_in['HR']  # 高分辨率真值
        b, c, h, w = x_start.shape

        # 随机从 1~num_timesteps 取一个时刻 t
        t = np.random.randint(1, self.num_timesteps + 1)

        # SR3 在 [sqrt_alphas_cumprod_prev[t-1], sqrt_alphas_cumprod_prev[t]] 区间采样
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        # 若外部没给 noise, 默认用 Rayleigh分布的 log(M)
        noise = default(
            noise,
            lambda: sample_rayleigh_log(x_start.shape, scale=1.0, device=x_start.device)
        )

        # 合成带乘性噪声的 x_noisy
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise  # log(M)
        )

        # 送入网络：若是 conditional=True, 则拼 SR
        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1),
                continuous_sqrt_alpha_cumprod
            )

        # 与真值 log(M) 做对比
        loss = self.loss_func(x_recon, noise)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
