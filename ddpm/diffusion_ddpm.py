from .ddpm_schedulers import (
    linear_beta_schedule,
    cosine_beta_schedule,
    sigmoid_beta_schedule,
)
from .ddpm_utils import (
    identity,
    normalize_to_neg_one_to_one,
    unnormalize_to_zero_to_one,
    extract,
)
import torch
import torch.nn as nn
from torch.functional import F
from typing import Optional


class DiffusionDDPM(nn.Module):
    SCHEDULER_MAPPING = {
        "linear": linear_beta_schedule,
        "cosine": cosine_beta_schedule,
        "sigmoid": sigmoid_beta_schedule,
    }

    def __init__(
        self,
        beta_scheduler: str = "linear",
        timesteps: int = 1000,
        schedule_fn_kwargs: Optional[dict] = None,
        auto_normalize: bool = True,
    ) -> None:
        super().__init__()
        # self.model = model

        # self.channels = channels
        # self.image_size = image_size

        self.beta_scheduler_fn = self.SCHEDULER_MAPPING.get(beta_scheduler)
        if self.beta_scheduler_fn is None:
            raise ValueError(f"unknown beta schedule {beta_scheduler}")

        if schedule_fn_kwargs is None:
            schedule_fn_kwargs = {}

        betas = self.beta_scheduler_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32)
        )

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        register_buffer("posterior_variance", posterior_variance)

        timesteps, *_ = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = timesteps

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    def q_sample(
        self, x_start: torch.Tensor, t: int, noise: torch.Tensor = None
    ) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


    def ddpm_noising(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, device = *x.shape, x.device
        # assert h == w == img_size, f"image size must be {img_size}"

        timestamp = torch.randint(0, self.num_timesteps, (1,)).long().to(device)
        # x = self.normalize(x)
        return self.q_sample(x, timestamp), timestamp