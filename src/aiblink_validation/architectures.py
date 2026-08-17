"""MIT-native detector architectures that do not depend on participant weights."""

from __future__ import annotations

from collections.abc import Sequence


def _torch_modules():
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    return torch, nn, functional


def _stochastic_depth(x, probability: float, training: bool):
    if probability <= 0.0 or not training:
        return x
    torch, _, _ = _torch_modules()
    keep = 1.0 - probability
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep)
    return x * mask / keep


def _high_pass_bank():
    torch, _, _ = _torch_modules()
    kernels = torch.tensor(
        [
            [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, 4, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            [[0, 0, 0, 0, 0], [0, -1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, -1, 0], [0, 0, 0, 0, 0]],
        ],
        dtype=torch.float32,
    )
    kernels = kernels / kernels.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1.0)
    return kernels[:, None].repeat(3, 1, 1, 1)


def build_convnext_binary(
    *,
    depths: Sequence[int] = (3, 3, 27, 3),
    dims: Sequence[int] = (96, 192, 384, 768),
    drop_path_rate: float = 0.1,
):
    """Return the standard ConvNeXt-Small topology with a binary head."""
    torch, nn, _ = _torch_modules()

    class LayerNorm2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm = nn.LayerNorm(channels, eps=1e-6)

        def forward(self, x):
            return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    class Block(nn.Module):
        def __init__(self, channels: int, drop_probability: float):
            super().__init__()
            self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
            self.norm = nn.LayerNorm(channels, eps=1e-6)
            self.expand = nn.Linear(channels, 4 * channels)
            self.activation = nn.GELU()
            self.project = nn.Linear(4 * channels, channels)
            self.gamma = nn.Parameter(torch.full((channels,), 1e-6))
            self.drop_probability = drop_probability

        def forward(self, x):
            residual = x
            x = self.depthwise(x).permute(0, 2, 3, 1)
            x = self.project(self.activation(self.expand(self.norm(x))))
            x = (self.gamma * x).permute(0, 3, 1, 2)
            return residual + _stochastic_depth(x, self.drop_probability, self.training)

    class ConvNeXtBinary(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, dims[0], 4, stride=4), LayerNorm2d(dims[0]))
            self.downsample = nn.ModuleList(
                nn.Sequential(LayerNorm2d(dims[index]), nn.Conv2d(dims[index], dims[index + 1], 2, stride=2))
                for index in range(3)
            )
            total_blocks = sum(depths)
            probabilities = [drop_path_rate * index / max(1, total_blocks - 1) for index in range(total_blocks)]
            cursor = 0
            stages = []
            for depth, channels in zip(depths, dims, strict=True):
                stages.append(nn.Sequential(*(Block(channels, probabilities[cursor + offset]) for offset in range(depth))))
                cursor += depth
            self.stages = nn.ModuleList(stages)
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
            self.head = nn.Linear(dims[-1], 1)
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward_features(self, x):
            x = self.stem(x)
            for index, stage in enumerate(self.stages):
                x = stage(x)
                if index < len(self.downsample):
                    x = self.downsample[index](x)
            return self.norm(x.mean(dim=(-2, -1)))

        def forward(self, x):
            return self.head(self.forward_features(x))

        def get_classifier(self):
            return self.head

        def reset_classifier(self, num_classes: int = 1):
            self.head = nn.Linear(dims[-1], num_classes)

    return ConvNeXtBinary()


def build_convnext_large_binary():
    """Return the standard ConvNeXt-Large topology with a binary head."""
    return build_convnext_binary(
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        drop_path_rate=0.2,
    )


def build_forensic_convnext_binary(
    *,
    depths: Sequence[int] = (3, 3, 12, 3),
    dims: Sequence[int] = (96, 192, 384, 640),
    drop_path_rate: float = 0.1,
):
    """Return a scratch ConvNeXt with fixed residual filters fused at the stem."""
    torch, nn, functional = _torch_modules()

    class LayerNorm2d(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.norm = nn.LayerNorm(channels, eps=1e-6)

        def forward(self, x):
            return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    class Block(nn.Module):
        def __init__(self, channels: int, drop_probability: float):
            super().__init__()
            self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
            self.norm = nn.LayerNorm(channels, eps=1e-6)
            self.expand = nn.Linear(channels, 4 * channels)
            self.activation = nn.GELU()
            self.project = nn.Linear(4 * channels, channels)
            self.gamma = nn.Parameter(torch.full((channels,), 1e-6))
            self.drop_probability = drop_probability

        def forward(self, x):
            residual = x
            x = self.depthwise(x).permute(0, 2, 3, 1)
            x = self.project(self.activation(self.expand(self.norm(x))))
            x = (self.gamma * x).permute(0, 3, 1, 2)
            return residual + _stochastic_depth(x, self.drop_probability, self.training)

    class ForensicConvNeXtBinary(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("high_pass_weight", _high_pass_bank(), persistent=True)
            self.stem = nn.Sequential(nn.Conv2d(15, dims[0], 4, stride=4), LayerNorm2d(dims[0]))
            self.downsample = nn.ModuleList(
                nn.Sequential(LayerNorm2d(dims[index]), nn.Conv2d(dims[index], dims[index + 1], 2, stride=2))
                for index in range(3)
            )
            total_blocks = sum(depths)
            probabilities = [drop_path_rate * index / max(1, total_blocks - 1) for index in range(total_blocks)]
            cursor = 0
            stages = []
            for depth, channels in zip(depths, dims, strict=True):
                stages.append(nn.Sequential(*(Block(channels, probabilities[cursor + offset]) for offset in range(depth))))
                cursor += depth
            self.stages = nn.ModuleList(stages)
            self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
            self.head = nn.Linear(dims[-1], 1)
            self.apply(self._initialize)

        @staticmethod
        def _initialize(module):
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        def forward_features(self, x):
            residual = functional.conv2d(x, self.high_pass_weight.to(dtype=x.dtype), padding=2, groups=3)
            x = self.stem(torch.cat((x, torch.tanh(0.5 * residual)), dim=1))
            for index, stage in enumerate(self.stages):
                x = stage(x)
                if index < len(self.downsample):
                    x = self.downsample[index](x)
            return self.norm(x.mean(dim=(-2, -1)))

        def forward(self, x):
            return self.head(self.forward_features(x))

        def get_classifier(self):
            return self.head

        def reset_classifier(self, num_classes: int = 1):
            self.head = nn.Linear(dims[-1], num_classes)

    return ForensicConvNeXtBinary()
