"""构造 SSTW 在 Flow latent 上使用的密钥条件 tubelet code。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import cos, isfinite, pi, sin
from typing import Any, Sequence


ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE = "zero_bit_key_presence"
INDEPENDENT_BINARY_PAYLOAD_MODE = "independent_binary_payload"
FLOW_TUBELET_PAYLOAD_MODES = (
    ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE,
    INDEPENDENT_BINARY_PAYLOAD_MODE,
)


@dataclass(frozen=True)
class FlowTubeletKeyCodeConfig:
    """定义 Flow latent 的时空 tubelet 划分和相位窗口。"""

    temporal_size: int = 2
    spatial_height: int = 8
    spatial_width: int = 8
    phase_window_start: float = 0.25
    phase_window_end: float = 0.75


@dataclass(frozen=True)
class FlowTubeletKeyContext:
    """绑定正式 tubelet code 使用的生成上下文与 payload 语义。

    ``zero_bit_key_presence`` 表示每个 tubelet 的符号由所有者密钥派生，只检测
    水印是否存在。``independent_binary_payload`` 表示调用方提供独立二进制消息，
    各 bit 在多个 tubelet 上重复扩频。prompt 摘要与 sampler 签名参与载波、
    payload 映射和 phase code 的域分离，防止不同生成上下文复用同一轨迹码。
    """

    prompt_digest: str
    sampler_signature: str
    payload_mode: str = ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE
    payload_bits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        prompt_digest = str(self.prompt_digest).strip()
        if len(prompt_digest) != 64 or any(
            character not in "0123456789abcdefABCDEF"
            for character in prompt_digest
        ):
            raise ValueError("正式 tubelet key context 要求64位十六进制 prompt digest")
        if not str(self.sampler_signature).strip():
            raise ValueError("正式 tubelet key context 缺少 sampler signature")
        if self.payload_mode not in FLOW_TUBELET_PAYLOAD_MODES:
            raise ValueError(f"不支持的 tubelet payload mode: {self.payload_mode}")
        invalid_bits = [value for value in self.payload_bits if value not in (0, 1)]
        if invalid_bits:
            raise ValueError("独立 tubelet payload 只能包含0或1")
        if self.payload_mode == ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE and self.payload_bits:
            raise ValueError("zero-bit 水印不得同时提供独立 payload bits")
        if self.payload_mode == INDEPENDENT_BINARY_PAYLOAD_MODE and not self.payload_bits:
            raise ValueError("independent payload mode 必须提供至少1个 payload bit")

    @classmethod
    def independent_payload(
        cls,
        *,
        prompt_digest: str,
        sampler_signature: str,
        payload_bits: Sequence[int],
    ) -> "FlowTubeletKeyContext":
        """构造可复用的独立二进制 payload 上下文。"""

        return cls(
            prompt_digest=str(prompt_digest),
            sampler_signature=str(sampler_signature),
            payload_mode=INDEPENDENT_BINARY_PAYLOAD_MODE,
            payload_bits=tuple(int(value) for value in payload_bits),
        )


def flow_tubelet_key_context_digest(context: FlowTubeletKeyContext) -> str:
    """计算不随 Flow phase 变化的 prompt、sampler 与 payload 上下文摘要。"""

    payload = {
        "prompt_digest": context.prompt_digest.lower(),
        "sampler_signature": context.sampler_signature,
        "payload_mode": context.payload_mode,
        "payload_bits": list(context.payload_bits),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _stable_seed(*parts: object) -> int:
    """把密钥和 tubelet 坐标转换成跨进程稳定的 PyTorch 种子。"""

    text = "::".join(str(part) for part in parts)
    digest = sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def flow_phase_weight(flow_phase: float, config: FlowTubeletKeyCodeConfig) -> float:
    """计算仅在中段 Flow phase 激活的平滑权重。"""

    phase = max(0.0, min(1.0, float(flow_phase)))
    start = float(config.phase_window_start)
    end = float(config.phase_window_end)
    if not 0.0 <= start < end <= 1.0:
        raise ValueError("Flow phase 窗口必须满足 0 <= start < end <= 1")
    if phase <= start or phase >= end:
        return 0.0
    normalized = (phase - start) / (end - start)
    return float(sin(pi * normalized) ** 2)


def _tubelet_phase_code(
    *,
    key_text: str,
    context: FlowTubeletKeyContext,
    flow_phase: float,
    batch_index: int,
    frame_start: int,
    top: int,
    left: int,
) -> float:
    """生成非负 tubelet-phase 调制码，使不同 phase 改变能量分配而不抵消 endpoint。"""

    phase = max(0.0, min(1.0, float(flow_phase)))
    phase_seed = _stable_seed(
        "sstw_tubelet_phase_offset",
        key_text,
        context.prompt_digest,
        context.sampler_signature,
        batch_index,
        frame_start,
        top,
        left,
    )
    phase_offset = phase_seed / float(2**63 - 1)
    # cos^2 的输出位于 [0, 1]，可让每个 tubelet 使用密钥条件相位，同时保持
    # 所有 phase 对 endpoint 载波贡献同号，避免正负相位相互抵消。
    return float(0.1 + 0.9 * cos(pi * (phase + phase_offset)) ** 2)


def _payload_bit(
    *,
    key_text: str,
    context: FlowTubeletKeyContext | None,
    tubelet_index: int,
    batch_index: int,
    frame_start: int,
    top: int,
    left: int,
) -> int:
    """返回 zero-bit 密钥符号或调用方独立 payload 的重复扩频 bit。"""

    if context is not None and context.payload_mode == INDEPENDENT_BINARY_PAYLOAD_MODE:
        return int(context.payload_bits[tubelet_index % len(context.payload_bits)])
    return int(
        _stable_seed(
            "sstw_zero_bit_payload",
            key_text,
            "" if context is None else context.prompt_digest,
            "" if context is None else context.sampler_signature,
            batch_index,
            frame_start,
            top,
            left,
        )
        & 1
    )


def build_flow_tubelet_key_direction_like(
    reference: Any,
    *,
    key_text: str,
    config: FlowTubeletKeyCodeConfig | None = None,
    flow_phase: float | None = None,
    key_context: FlowTubeletKeyContext | None = None,
    phase_code_override: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """生成与五维 Flow latent 同形状的密钥条件 tubelet 方向。

    方向在每个 tubelet 内独立归一化, 再执行全局归一化。payload bit 已经作为
    正负号写入方向。显式提供 ``key_context`` 和 ``flow_phase`` 时，prompt、
    sampler、payload 与 phase 会共同决定每个 tubelet 的载波和相位能量。

    未提供完整上下文的旧调用仍可得到确定性 zero-bit 方向，但返回的
    ``flow_tubelet_formal_context_complete`` 为 ``False``，不得支持正式 claim。
    """

    import torch

    config = config or FlowTubeletKeyCodeConfig()
    if not isinstance(key_text, str) or not key_text:
        raise ValueError("key_text 不能为空")
    if getattr(reference, "ndim", None) != 5:
        raise ValueError("Flow latent 必须使用 [B, C, T, H, W] 五维张量")
    if min(config.temporal_size, config.spatial_height, config.spatial_width) <= 0:
        raise ValueError("tubelet 尺寸必须为正整数")

    if flow_phase is not None and not 0.0 <= float(flow_phase) <= 1.0:
        raise ValueError("flow_phase 必须位于 [0, 1]")
    if phase_code_override is not None and (
        not isfinite(float(phase_code_override))
        or abs(float(phase_code_override)) <= 1e-12
    ):
        raise ValueError("显式 phase code 必须是有限非零数")

    batch, channels, frames, height, width = (int(value) for value in reference.shape)
    direction = torch.zeros(reference.shape, device=reference.device, dtype=torch.float32)
    generator = torch.Generator(device=reference.device)

    tubelet_count = 0
    positive_payload_count = 0
    phase_codes: list[float] = []
    for batch_index in range(batch):
        for frame_start in range(0, frames, config.temporal_size):
            frame_end = min(frames, frame_start + config.temporal_size)
            for top in range(0, height, config.spatial_height):
                bottom = min(height, top + config.spatial_height)
                for left in range(0, width, config.spatial_width):
                    right = min(width, left + config.spatial_width)
                    block_shape = (
                        1,
                        channels,
                        frame_end - frame_start,
                        bottom - top,
                        right - left,
                    )
                    generator.manual_seed(_stable_seed(
                        "sstw_flow_tubelet_carrier",
                        key_text,
                        "" if key_context is None else key_context.prompt_digest,
                        "" if key_context is None else key_context.sampler_signature,
                        batch_index,
                        frame_start,
                        top,
                        left,
                        block_shape,
                    ))
                    block = torch.randn(
                        block_shape,
                        device=reference.device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                    payload_positive = bool(_payload_bit(
                        key_text=key_text,
                        context=key_context,
                        tubelet_index=tubelet_count,
                        batch_index=batch_index,
                        frame_start=frame_start,
                        top=top,
                        left=left,
                    ))
                    payload_sign = 1.0 if payload_positive else -1.0
                    block = block / block.norm().clamp_min(1e-8)
                    phase_code = (
                        float(phase_code_override)
                        if phase_code_override is not None
                        else _tubelet_phase_code(
                            key_text=key_text,
                            context=key_context,
                            flow_phase=float(flow_phase),
                            batch_index=batch_index,
                            frame_start=frame_start,
                            top=top,
                            left=left,
                        )
                        if key_context is not None and flow_phase is not None
                        else 1.0
                    )
                    direction[
                        batch_index : batch_index + 1,
                        :channels,
                        frame_start:frame_end,
                        top:bottom,
                        left:right,
                    ] = block * payload_sign * phase_code
                    tubelet_count += 1
                    positive_payload_count += int(payload_positive)
                    phase_codes.append(phase_code)

    if (
        key_context is not None
        and key_context.payload_mode == INDEPENDENT_BINARY_PAYLOAD_MODE
        and len(key_context.payload_bits) > tubelet_count
    ):
        raise ValueError(
            "独立 payload bit 数量不能超过可用 tubelet 数量: "
            f"bits={len(key_context.payload_bits)}, tubelets={tubelet_count}"
        )

    direction = direction / direction.norm().clamp_min(1e-8)
    formal_context_complete = key_context is not None and flow_phase is not None
    metadata = {
        "flow_tubelet_key_code_status": "ready",
        "flow_tubelet_code_semantics": "prompt_sampler_payload_phase_joint_tubelet_code",
        "flow_tubelet_formal_context_complete": formal_context_complete,
        "flow_tubelet_count": tubelet_count,
        "flow_tubelet_temporal_size": config.temporal_size,
        "flow_tubelet_spatial_height": config.spatial_height,
        "flow_tubelet_spatial_width": config.spatial_width,
        "flow_payload_positive_count": positive_payload_count,
        "flow_payload_negative_count": tubelet_count - positive_payload_count,
        "flow_payload_mode": (
            key_context.payload_mode
            if key_context is not None
            else ZERO_BIT_KEY_PRESENCE_PAYLOAD_MODE
        ),
        "flow_payload_bit_count": (
            len(key_context.payload_bits)
            if key_context is not None
            and key_context.payload_mode == INDEPENDENT_BINARY_PAYLOAD_MODE
            else 0
        ),
        "flow_prompt_digest_binding": (
            key_context.prompt_digest if key_context is not None else None
        ),
        "flow_tubelet_key_context_digest": (
            flow_tubelet_key_context_digest(key_context)
            if key_context is not None
            else None
        ),
        "flow_sampler_signature_digest": (
            sha256(key_context.sampler_signature.encode("utf-8")).hexdigest()
            if key_context is not None
            else None
        ),
        "flow_tubelet_phase": None if flow_phase is None else round(float(flow_phase), 8),
        "flow_tubelet_phase_code_semantics": (
            "explicit_schedule_carrier_code"
            if phase_code_override is not None
            else "nonnegative_key_conditioned_cosine_squared"
            if key_context is not None and flow_phase is not None
            else "compatibility_static"
        ),
        "flow_tubelet_phase_code_minimum": round(min(phase_codes), 8),
        "flow_tubelet_phase_code_maximum": round(max(phase_codes), 8),
        "flow_tubelet_phase_code_mean": round(sum(phase_codes) / len(phase_codes), 8),
        "flow_key_direction_norm": round(float(direction.norm().item()), 6),
        "flow_key_direction_digest": sha256(
            (
                f"{key_text}::{tuple(reference.shape)}::{tubelet_count}::"
                f"{'' if key_context is None else key_context.prompt_digest}::"
                f"{'' if key_context is None else key_context.sampler_signature}::"
                f"{flow_phase}::"
                f"{phase_code_override}::"
                f"{() if key_context is None else key_context.payload_bits}"
            ).encode("utf-8")
        ).hexdigest(),
    }
    return direction.to(dtype=reference.dtype), metadata


def build_integrated_flow_tubelet_key_direction_like(
    reference: Any,
    *,
    key_text: str,
    key_context: FlowTubeletKeyContext,
    flow_phases: Sequence[float],
    integration_weights: Sequence[float],
    config: FlowTubeletKeyCodeConfig | None = None,
    phase_code_override: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """按预注册 Flow 网格积分 joint code，构造 endpoint/replay 共用参考方向。

    该函数用于把逐 phase 的 tubelet 方向还原为同一 schedule 下的累计载波。
    ``integration_weights`` 应由生成 scheduler 的 ``|delta_sigma|`` 与水印强度调度
    共同确定，不读取测试标签或最终检测分数，因此可在其他 Flow 模型中复用。
    """

    import torch

    phases = [float(value) for value in flow_phases]
    weights = [float(value) for value in integration_weights]
    if not phases or len(phases) != len(weights):
        raise ValueError("integrated tubelet code 需要等长非空 phase 与 weight")
    if any(not 0.0 <= phase <= 1.0 for phase in phases):
        raise ValueError("integrated tubelet code 的 phase 必须位于 [0, 1]")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("integrated tubelet code 的权重必须非负且总和为正")
    accumulated = torch.zeros_like(reference, dtype=torch.float32)
    phase_schedule_bindings: list[str] = []
    for phase, weight in zip(phases, weights):
        direction, metadata = build_flow_tubelet_key_direction_like(
            reference,
            key_text=key_text,
            config=config,
            flow_phase=phase,
            key_context=key_context,
            phase_code_override=phase_code_override,
        )
        accumulated = accumulated + direction.detach().float() * weight
        phase_schedule_bindings.append(
            f"{metadata['flow_key_direction_digest']}::{phase:.17g}::{weight:.17g}"
        )
    norm = accumulated.norm()
    if float(norm.item()) <= 1e-8:
        raise RuntimeError("integrated tubelet code 在预注册网格上发生完全抵消")
    integrated = accumulated / norm
    return integrated.to(dtype=reference.dtype), {
        "flow_tubelet_key_code_status": "ready",
        "flow_tubelet_code_semantics": "integrated_prompt_sampler_payload_phase_joint_tubelet_code",
        "flow_tubelet_formal_context_complete": True,
        "flow_payload_mode": key_context.payload_mode,
        "flow_payload_bit_count": len(key_context.payload_bits),
        "flow_tubelet_key_context_digest": flow_tubelet_key_context_digest(
            key_context
        ),
        "flow_integrated_phase_count": len(phases),
        "flow_integrated_weight_sum": round(sum(weights), 10),
        "flow_key_direction_norm": round(float(integrated.norm().item()), 6),
        "flow_key_direction_digest": sha256(
            "||".join(phase_schedule_bindings).encode("utf-8")
        ).hexdigest(),
    }


def iter_tubelet_slices(shape: tuple[int, ...], config: FlowTubeletKeyCodeConfig | None = None):
    """按与 key direction 相同的规则枚举 tubelet 切片。"""

    config = config or FlowTubeletKeyCodeConfig()
    if len(shape) != 5:
        raise ValueError("Flow latent shape 必须包含5个维度")
    batch, _channels, frames, height, width = (int(value) for value in shape)
    for batch_index in range(batch):
        for frame_start in range(0, frames, config.temporal_size):
            frame_end = min(frames, frame_start + config.temporal_size)
            for top in range(0, height, config.spatial_height):
                bottom = min(height, top + config.spatial_height)
                for left in range(0, width, config.spatial_width):
                    right = min(width, left + config.spatial_width)
                    yield (
                        slice(batch_index, batch_index + 1),
                        slice(None),
                        slice(frame_start, frame_end),
                        slice(top, bottom),
                        slice(left, right),
                    )
