"""Model and objective reference for the three right-arm IMU routes.

Tensor conventions
------------------
right-arm student input: ``[batch, time, 7]``
four-location teacher input: ``[batch, 4, time, 7]``
location order: right arm, right leg, left leg, left arm

The seven channels are acceleration (x/y/z), first differences (x/y/z), and
the acceleration-vector magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


LOCATION_ORDER = ("right_arm", "right_leg", "left_leg", "left_arm")


@dataclass(frozen=True)
class ModelSpec:
    """Architecture specification shared by the student and teacher."""

    input_channels: int = 7
    window_steps: int = 50
    temporal_channels: tuple[int, ...] = (64, 96)
    embedding_dim: int = 192
    num_classes: int = 19
    kernel_size: int = 5
    dropout: float = 0.15


class ModelOutput(NamedTuple):
    logits: Tensor
    feature: Tensor


class RouteLoss(NamedTuple):
    """Scalar objective plus unweighted component means for reporting."""

    total: Tensor
    hard_label: Tensor
    logit_distillation: Tensor
    feature_distillation: Tensor


class DynamicWeights(NamedTuple):
    """Independent per-sample weights for the two knowledge sources."""

    logit: Tensor
    feature: Tensor


class MetaParameterProbes(NamedTuple):
    """Symmetric parameter probes derived from the right-arm meta objective."""

    plus: dict[str, Tensor]
    minus: dict[str, Tensor]
    meta_gradient_norm: Tensor
    active: bool


def build_seven_channel_imu(
    raw_acceleration: Tensor,
    *,
    window_steps: int = 50,
) -> Tensor:
    """Convert ``[..., 50, 3]`` acceleration into ``[..., 50, 7]``.

    The first-difference value at the first time step is zero.
    """

    if raw_acceleration.ndim < 2 or tuple(raw_acceleration.shape[-2:]) != (
        window_steps,
        3,
    ):
        raise ValueError(
            f"raw_acceleration must end in [{window_steps}, 3]"
        )
    jerk = torch.diff(
        raw_acceleration,
        dim=-2,
        prepend=raw_acceleration[..., :1, :],
    )
    magnitude = torch.linalg.vector_norm(raw_acceleration, dim=-1, keepdim=True)
    return torch.cat((raw_acceleration, jerk, magnitude), dim=-1)


class TemporalConvBlock(nn.Module):
    """Dilated residual temporal convolution used by all encoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, values: Tensor) -> Tensor:
        residual = self.residual(values)
        values = self.conv(values)
        values = self.norm(values)
        values = F.gelu(values)
        values = self.dropout(values)
        return F.gelu(values + residual)


class TCNEncoder(nn.Module):
    """TCN followed by concatenated temporal mean and max pooling."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        if not spec.temporal_channels:
            raise ValueError("temporal_channels cannot be empty")
        blocks: list[nn.Module] = []
        in_channels = spec.input_channels
        for index, out_channels in enumerate(spec.temporal_channels):
            blocks.append(
                TemporalConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size=spec.kernel_size,
                    dilation=2**index,
                    dropout=spec.dropout,
                )
            )
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.output_dim = 2 * in_channels

    def forward(self, windows: Tensor) -> Tensor:
        if windows.ndim != 3:
            raise ValueError("TCNEncoder expects [batch, time, channels]")
        encoded = self.blocks(windows.transpose(1, 2))
        return torch.cat((encoded.mean(dim=-1), encoded.amax(dim=-1)), dim=-1)


class RightArmStudent(nn.Module):
    """The common deployment model used by Route 01, Route 02, and Route 03."""

    def __init__(self, spec: ModelSpec = ModelSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = TCNEncoder(spec)
        self.preclassifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, spec.embedding_dim),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.classifier = nn.Linear(spec.embedding_dim, spec.num_classes)

    def forward(self, right_arm_imu: Tensor) -> ModelOutput:
        expected = (self.spec.window_steps, self.spec.input_channels)
        if right_arm_imu.ndim != 3 or tuple(right_arm_imu.shape[1:]) != expected:
            raise ValueError(
                "student input must be "
                f"[batch, {self.spec.window_steps}, {self.spec.input_channels}]"
            )
        feature = self.preclassifier(self.encoder(right_arm_imu))
        return ModelOutput(self.classifier(feature), feature)


class FourLocationTeacher(nn.Module):
    """Training-only teacher with shared encoding and learned late fusion."""

    def __init__(self, spec: ModelSpec = ModelSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = TCNEncoder(spec)
        if self.encoder.output_dim != spec.embedding_dim:
            raise ValueError(
                "teacher per-location feature dimension must equal embedding_dim"
            )
        self.fusion = nn.Sequential(
            nn.Linear(
                self.encoder.output_dim * len(LOCATION_ORDER),
                spec.embedding_dim,
            ),
            nn.GELU(),
            nn.Dropout(spec.dropout),
        )
        self.classifier = nn.Linear(spec.embedding_dim, spec.num_classes)

    def encode_locations(self, four_location_imu: Tensor) -> Tensor:
        """Return one shared-encoder feature per location in ``LOCATION_ORDER``."""

        if (
            four_location_imu.ndim != 4
            or four_location_imu.shape[1] != len(LOCATION_ORDER)
            or four_location_imu.shape[2] != self.spec.window_steps
            or four_location_imu.shape[-1] != self.spec.input_channels
        ):
            raise ValueError(
                "teacher input must be "
                f"[batch, 4, {self.spec.window_steps}, {self.spec.input_channels}] "
                "in LOCATION_ORDER"
            )
        batch, locations, steps, channels = four_location_imu.shape
        return self.encoder(
            four_location_imu.reshape(batch * locations, steps, channels)
        ).reshape(batch, locations, -1)

    def forward_each(self, four_location_imu: Tensor) -> tuple[Tensor, Tensor]:
        """Return per-location logits and features from the shared branch."""

        per_location = self.encode_locations(four_location_imu)
        return self.classifier(per_location), per_location

    def fuse_features(self, per_location: Tensor) -> Tensor:
        """Fuse four fixed-order location features into the teacher feature."""

        expected = (len(LOCATION_ORDER), self.encoder.output_dim)
        if per_location.ndim != 3 or tuple(per_location.shape[1:]) != expected:
            raise ValueError(
                "per_location must be "
                f"[batch, {len(LOCATION_ORDER)}, {self.encoder.output_dim}]"
            )
        return self.fusion(per_location.flatten(start_dim=1))

    def forward(self, four_location_imu: Tensor) -> ModelOutput:
        per_location = self.encode_locations(four_location_imu)
        feature = self.fuse_features(per_location)
        return ModelOutput(self.classifier(feature), feature)


def hard_label_loss_per_sample(
    logits: Tensor,
    labels: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Per-sample supervised cross-entropy shared by all three routes."""

    return F.cross_entropy(logits, labels, weight=class_weights, reduction="none")


def teacher_supervised_objective(
    teacher: ModelOutput,
    labels: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> Tensor:
    """Supervised objective for the four-location teacher."""

    return hard_label_loss_per_sample(
        teacher.logits,
        labels,
        class_weights=class_weights,
    ).mean()


def distillation_components_per_sample(
    student: ModelOutput,
    teacher: ModelOutput,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Return separate logit-KL and feature-MSE losses for every sample."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher_probability = F.softmax(teacher.logits.detach() / temperature, dim=-1)
    student_log_probability = F.log_softmax(student.logits / temperature, dim=-1)
    logit_kd = (temperature**2) * F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="none",
    ).sum(dim=-1)
    feature_kd = (student.feature - teacher.feature.detach()).square().mean(dim=-1)
    return logit_kd, feature_kd


def route_01_supervised_objective(
    student: ModelOutput,
    labels: Tensor,
    *,
    class_weights: Tensor | None = None,
) -> RouteLoss:
    """Route 01: hard-label supervision only."""

    hard = hard_label_loss_per_sample(
        student.logits,
        labels,
        class_weights=class_weights,
    ).mean()
    zero = hard.new_zeros(())
    return RouteLoss(hard, hard, zero, zero)


def route_02_ordinary_distillation_objective(
    student: ModelOutput,
    teacher: ModelOutput,
    labels: Tensor,
    *,
    temperature: float,
    logit_coefficient: float,
    feature_coefficient: float,
    distillation_scale: float,
    class_weights: Tensor | None = None,
) -> RouteLoss:
    """Route 02: uniform logit and feature distillation.

    "Uniform" means that every sample receives the same weight within each
    knowledge source.  The scalar coefficients of the two sources may differ.
    """

    hard_per_sample = hard_label_loss_per_sample(
        student.logits,
        labels,
        class_weights=class_weights,
    )
    logit_kd, feature_kd = distillation_components_per_sample(
        student,
        teacher,
        temperature=temperature,
    )
    total = (
        hard_per_sample
        + distillation_scale
        * (logit_coefficient * logit_kd + feature_coefficient * feature_kd)
    ).mean()
    return RouteLoss(total, hard_per_sample.mean(), logit_kd.mean(), feature_kd.mean())


@torch.no_grad()
def positive_influence_weights(
    loss_at_plus: Tensor,
    loss_at_minus: Tensor,
    *,
    finite_difference_step: float,
    uncertainty_multiplier: float,
) -> Tensor:
    """Map directional influence evidence to weights in ``[0, 1]``.

    Positive evidence is normalized by the largest positive evidence in the
    batch.  If the probe contains no reliable positive evidence, all weights
    are zero, so that knowledge source is ignored for that batch.
    """

    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive")
    plus = loss_at_plus.detach().double()
    minus = loss_at_minus.detach().double()
    if (
        plus.shape != minus.shape
        or not torch.isfinite(plus).all()
        or not torch.isfinite(minus).all()
    ):
        raise ValueError("plus/minus probe losses must have the same finite shape")

    influence = (plus - minus) / (2.0 * finite_difference_step)
    float32_epsilon = 2.0**-23
    threshold = (
        uncertainty_multiplier
        * float32_epsilon
        * (plus.abs() + minus.abs())
        / (2.0 * finite_difference_step)
    )
    evidence = F.relu(influence - threshold)
    if not bool((evidence > 0).any()):
        return torch.zeros_like(loss_at_plus)
    weights = evidence / evidence.max()
    return weights.to(dtype=loss_at_plus.dtype, device=loss_at_plus.device)


def build_meta_parameter_probes(
    current_parameters: Mapping[str, Tensor],
    virtual_parameters: Mapping[str, Tensor],
    meta_loss: Tensor,
    *,
    finite_difference_step: float,
    minimum_meta_gradient_norm: float,
) -> MetaParameterProbes:
    """Build ``theta +/- xi * normalize(g_meta)`` for Route 03.

    ``virtual_parameters`` are produced by one differentiable update on the fit
    objective.  ``meta_loss`` is the right-arm hard-label objective evaluated at
    those virtual parameters.  The resulting meta-gradient direction is applied
    symmetrically around the current student parameters.
    """

    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive")
    if minimum_meta_gradient_norm < 0:
        raise ValueError("minimum_meta_gradient_norm cannot be negative")
    if not current_parameters or set(current_parameters) != set(virtual_parameters):
        raise ValueError("current and virtual parameters must share non-empty names")
    if meta_loss.ndim != 0 or not meta_loss.requires_grad:
        raise ValueError("meta_loss must be a differentiable scalar")

    names = tuple(current_parameters)
    meta_gradients = torch.autograd.grad(
        meta_loss,
        tuple(virtual_parameters[name] for name in names),
        allow_unused=True,
    )
    directions = tuple(
        torch.zeros_like(virtual_parameters[name]) if gradient is None else gradient
        for name, gradient in zip(names, meta_gradients, strict=True)
    )
    meta_gradient_norm = torch.sqrt(
        sum(direction.detach().double().square().sum() for direction in directions)
    )
    if not torch.isfinite(meta_gradient_norm):
        raise ValueError("meta gradient contains NaN or Inf")
    norm_value = float(meta_gradient_norm)
    active = norm_value > 0.0 and norm_value >= minimum_meta_gradient_norm

    if active:
        normalized = tuple(
            direction / meta_gradient_norm.to(direction)
            for direction in directions
        )
    else:
        normalized = tuple(torch.zeros_like(direction) for direction in directions)

    plus = {
        name: current_parameters[name].detach()
        + finite_difference_step * direction.detach()
        for name, direction in zip(names, normalized, strict=True)
    }
    minus = {
        name: current_parameters[name].detach()
        - finite_difference_step * direction.detach()
        for name, direction in zip(names, normalized, strict=True)
    }
    return MetaParameterProbes(
        plus=plus,
        minus=minus,
        meta_gradient_norm=meta_gradient_norm.detach(),
        active=active,
    )


def infer_dynamic_source_weights(
    student_at_plus: ModelOutput,
    student_at_minus: ModelOutput,
    teacher: ModelOutput,
    *,
    temperature: float,
    finite_difference_step: float,
    uncertainty_multiplier: float,
    meta_gradient_norm: Tensor | float,
    minimum_meta_gradient_norm: float,
) -> DynamicWeights:
    """Estimate Route 03 weights independently for logits and features.

    ``student_at_plus`` and ``student_at_minus`` are evaluations at parameters
    perturbed by ``+xi`` and ``-xi`` along the normalized meta-loss gradient.
    The virtual update and meta-gradient construction are summarized in the
    README.
    """

    if minimum_meta_gradient_norm < 0:
        raise ValueError("minimum_meta_gradient_norm cannot be negative")
    norm = torch.as_tensor(meta_gradient_norm).detach().double()
    if norm.numel() != 1 or not torch.isfinite(norm):
        raise ValueError("meta_gradient_norm must be one finite scalar")
    norm_value = float(norm)
    if norm_value <= 0.0 or norm_value < minimum_meta_gradient_norm:
        batch = student_at_plus.logits.shape[0]
        zero = student_at_plus.logits.new_zeros(batch)
        return DynamicWeights(logit=zero, feature=zero.clone())

    plus_logit, plus_feature = distillation_components_per_sample(
        student_at_plus,
        teacher,
        temperature=temperature,
    )
    minus_logit, minus_feature = distillation_components_per_sample(
        student_at_minus,
        teacher,
        temperature=temperature,
    )
    kwargs = {
        "finite_difference_step": finite_difference_step,
        "uncertainty_multiplier": uncertainty_multiplier,
    }
    return DynamicWeights(
        logit=positive_influence_weights(plus_logit, minus_logit, **kwargs),
        feature=positive_influence_weights(plus_feature, minus_feature, **kwargs),
    )


def route_03_dynamic_distillation_objective(
    student: ModelOutput,
    teacher: ModelOutput,
    labels: Tensor,
    dynamic_weights: DynamicWeights,
    *,
    temperature: float,
    logit_coefficient: float,
    feature_coefficient: float,
    distillation_scale: float,
    class_weights: Tensor | None = None,
) -> RouteLoss:
    """Route 03: per-sample, per-source dynamically weighted distillation."""

    hard_per_sample = hard_label_loss_per_sample(
        student.logits,
        labels,
        class_weights=class_weights,
    )
    logit_kd, feature_kd = distillation_components_per_sample(
        student,
        teacher,
        temperature=temperature,
    )
    if (
        dynamic_weights.logit.shape != logit_kd.shape
        or dynamic_weights.feature.shape != feature_kd.shape
    ):
        raise ValueError("dynamic weights must contain one logit and feature weight per sample")

    weighted_logit = dynamic_weights.logit.detach() * logit_kd
    weighted_feature = dynamic_weights.feature.detach() * feature_kd
    total = (
        hard_per_sample
        + distillation_scale
        * (
            logit_coefficient * weighted_logit
            + feature_coefficient * weighted_feature
        )
    ).mean()
    return RouteLoss(total, hard_per_sample.mean(), logit_kd.mean(), feature_kd.mean())


def trainable_parameter_count(module: nn.Module) -> int:
    """Return the number of trainable parameters in a module."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


__all__: Sequence[str] = (
    "LOCATION_ORDER",
    "ModelSpec",
    "ModelOutput",
    "RouteLoss",
    "DynamicWeights",
    "MetaParameterProbes",
    "build_seven_channel_imu",
    "TemporalConvBlock",
    "TCNEncoder",
    "RightArmStudent",
    "FourLocationTeacher",
    "hard_label_loss_per_sample",
    "teacher_supervised_objective",
    "distillation_components_per_sample",
    "route_01_supervised_objective",
    "route_02_ordinary_distillation_objective",
    "positive_influence_weights",
    "build_meta_parameter_probes",
    "infer_dynamic_source_weights",
    "route_03_dynamic_distillation_objective",
    "trainable_parameter_count",
)
