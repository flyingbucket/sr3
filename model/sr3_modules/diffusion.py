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
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
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

# ---------------------------------------------------------
#  用于采样 Rayleigh 分布并返回其对数值: log(M)。
#  这样就能让网络学习 log(M)，更适合乘性噪声。
# ---------------------------------------------------------
def sample_rayleigh_log(shape, scale=1.0, device='cuda'):
    """
    先采样 Rayleigh 分布的 M，再对其取对数:
        M = scale * sqrt(-2.0 * log(U)),  U ~ Uniform(0,1)
    返回值: log(M)
    """
    U = torch.rand(shape, device=device)
    # 避免 log(0) 或 sqrt(负数)
    M = scale * torch.sqrt(-2.0 * torch.log(U) + 1e-8)
    # 返回 log(M)
    return torch.log(M + 1e-8)

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

        # posterior 分布系数 (原 DDPM 公式)
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

    # -----------------------------------------------------------------------------------
    #  将前向扩散改为乘性噪声:
    #    x_noisy = c * x_start * exp(noise),
    #  其中 noise = log(M_t)。如果外部不提供，就用 Rayleigh 分布采样得到 log(M)。
    # -----------------------------------------------------------------------------------
    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """
        原版加性写法:
            x_noisy = continuous_sqrt_alpha_cumprod * x_start
                      + sqrt(1 - continuous_sqrt_alpha_cumprod^2) * noise
        现改为乘性形式(示例):
            x_noisy = continuous_sqrt_alpha_cumprod * x_start * exp(noise)
        """
        if noise is None:
            # 默认用 Rayleigh 分布，并取 log
            noise = sample_rayleigh_log(
                x_start.shape,
                scale=1.0,                 # 若需要可自行修改
                device=x_start.device
            )
        return continuous_sqrt_alpha_cumprod * x_start * torch.exp(noise)

    # -----------------------------------------------------------------------------------
    #  反向过程关键函数: 将网络预测到的 log(M_t) 还原出 x_0。
    #    x_0 = x_t / ( alpha_t * exp(log(M_t)) )
    # -----------------------------------------------------------------------------------
    def predict_start_from_noise(self, x_t, t, noise):
        """
        原加性DDPM中: x_0 = (x_t - sqrt(1 - alpha_t)*noise) / sqrt(alpha_t)
        这里改成乘性形式: x_0 = x_t / [ sqrt(alphas_cumprod[t]) * exp(noise) ]
        """
        # sqrt_recip_alphas_cumprod[t] = 1 / sqrt_alphas_cumprod[t]
        # => alpha_t = sqrt_alphas_cumprod[t]
        alpha_t_inv = self.sqrt_recip_alphas_cumprod[t]  # = 1 / alpha_t
        return x_t / (alpha_t_inv * torch.exp(noise) + 1e-8)  # 避免除0

    # -----------------------------------------------------------------------------------
    #  posterior 近似依旧用原公式(DDPM 加性形式), 做一个启发式近似
    # -----------------------------------------------------------------------------------
    def q_posterior(self, x_start, x_t, t):
        """
        原DDPM后验: q(x_{t-1}|x_t,x_0) ~ N(
            posterior_mean_coef1[t]*x_0 + posterior_mean_coef2[t]*x_t,
            posterior_variance[t]
        )
        这里为了兼容SR3原本流程，仍使用该加性形式近似。
        """
        posterior_mean = self.posterior_mean_coef1[t] * x_start + \
                         self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    # -----------------------------------------------------------------------------------
    #  p_mean_variance: 先预测 x_0，再用 q_posterior 得到 x_{t-1} 的均值、log方差
    # -----------------------------------------------------------------------------------
    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        # network noise_level: SR3中将 t+1 时刻 sqrt_alphas_cumprod_prev[t+1] 喂给网络
        noise_level = torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)

        # 网络输出的是 log(M_t)
        if condition_x is not None:
            # 条件输入形如 (SR, x_noisy)
            log_m_pred = self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level)
        else:
            log_m_pred = self.denoise_fn(x, noise_level)

        # 根据 log(M_t) 反推 x_0
        x_recon = self.predict_start_from_noise(x, t=t, noise=log_m_pred)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        # 用原DDPM的公式得出后验分布近似
        model_mean, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        """
        从 p(x_{t-1}|x_t) 采样
        """
        model_mean, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            clip_denoised=clip_denoised,
            condition_x=condition_x
        )
        # 加性采样过程保持不变(启发式)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        """
        与原SR3一致，循环从 t=T 到 t=0
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
            # 超分场景: x_in 是 condition_x (SR)
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

    # -----------------------------------------------------------------------------------
    #  训练时的核心: p_losses()。这里改为采样 log(M_t) 并用乘性噪声生成 x_noisy。
    # -----------------------------------------------------------------------------------
    def p_losses(self, x_in, noise=None):
        """
        x_in['HR']: 高分辨率真实图
        x_in['SR']: 低分辨率输入图 (若 conditional=True)
        """

        x_start = x_in['HR']
        b, c, h, w = x_start.shape
        # 随机取一个时间 t
        t = np.random.randint(1, self.num_timesteps + 1)

        # SR3 中要在 [ sqrt_alphas_cumprod_prev[t-1], sqrt_alphas_cumprod_prev[t] ] 内采样
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        # 如果外部没给定 noise，则默认用 Rayleigh 分布的 log(M)
        noise = default(
            noise,
            lambda: sample_rayleigh_log(x_start.shape, scale=1.0, device=x_start.device)
        )

        # 生成 x_noisy
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
            noise=noise  # 此处即 log(M_t)
        )

        # 送入网络
        if not self.conditional:
            # 无条件
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            # 条件输入 (SR, x_noisy)
            x_recon = self.denoise_fn(
                torch.cat([x_in['SR'], x_noisy], dim=1),
                continuous_sqrt_alpha_cumprod
            )

        # 网络输出的 x_recon 同样视为对 log(M_t) 的预测
        # 训练目标: 回归 ground truth 的 log(M_t)
        loss = self.loss_func(x_recon, noise)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
