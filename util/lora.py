import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class QkvWithLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, alpha):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = LoRALayer(self.dim, self.dim, rank, alpha)
        self.lora_v = LoRALayer(self.dim, self.dim, rank, alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        return qkv
    

class AdaptiveLoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, initial_alpha=1.0):
        super(LoRALayer, self).__init__()
        std = torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank) / std)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        # set alpha as a learning parameter
        self.raw_alpha = nn.Parameter(torch.tensor([math.log((initial_alpha - 0.8) / (1.2 - initial_alpha))]))

    def forward(self, x):
        alpha = 0.4 * torch.sigmoid(self.raw_alpha) + 0.8
        return alpha * (x @ self.A @ self.B)


class QkvWithAdaptiveLoRA(torch.nn.Module):
    def __init__(self, qkv, rank, initial_alpha=1.0):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.lora_q = AdaptiveLoRALayer(self.dim, self.dim, rank, initial_alpha)
        self.lora_v = AdaptiveLoRALayer(self.dim, self.dim, rank, initial_alpha)

    def forward(self, x):
        qkv = self.qkv(x)
        qkv[:, :, :self.dim] += self.lora_q(x)
        qkv[:, :, -self.dim:] += self.lora_v(x)
        return qkv
    



class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        return self.linear(x) + self.lora(x)

