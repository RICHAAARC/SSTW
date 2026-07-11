"""对单个真实视频执行可复现的黑盒 adaptive attack 优化。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.attacks.video_runtime_attack_protocol import apply_runtime_attack_to_frames


@dataclass(frozen=True)
class AdaptiveVideoCandidate:
    """保存一次真实候选视频查询及其质量、endpoint 与检测结果。"""

    candidate_index: int
    attack_name: str
    video_path: str
    video_sha256: str
    decoded_frame_count: int
    quality_psnr: float
    detector_score: float
    detector_score_source: str
    frozen_final_score_threshold: float | None
    threshold_source_split: str | None
    test_time_threshold_update_blocked: bool
    endpoint_score: float
    path_score: float
    decision: bool
    admissible: bool
    attack_parameters: Mapping[str, float | int | str]

    def as_dict(self) -> dict[str, Any]:
        """转换为可写入 governed query log 的字段。"""

        return dict(self.__dict__)


@dataclass(frozen=True)
class AdaptiveVideoOptimizationResult:
    """保存单视频优化的全部查询和预注册目标下的最优候选。"""

    objective: str
    selected: AdaptiveVideoCandidate
    candidates: tuple[AdaptiveVideoCandidate, ...]
    endpoint_reference: float
    endpoint_tolerance: float
    minimum_quality_psnr: float
    query_budget: int

    def as_dict(self) -> dict[str, Any]:
        """转换为正式 adaptive execution record 可复用的摘要。"""

        return {
            "adaptive_attack_objective": self.objective,
            "adaptive_attack_query_count": len(self.candidates),
            "adaptive_attack_query_budget": self.query_budget,
            "adaptive_attack_selected_candidate_index": self.selected.candidate_index,
            "adaptive_attack_selected_transform": self.selected.attack_name,
            "adaptive_attack_selected_parameters": dict(
                self.selected.attack_parameters
            ),
            "adaptive_attack_output_video_path": self.selected.video_path,
            "adaptive_attack_output_video_sha256": self.selected.video_sha256,
            "adaptive_attack_output_quality_psnr": self.selected.quality_psnr,
            "adaptive_attack_endpoint_reference": self.endpoint_reference,
            "adaptive_attack_endpoint_tolerance": self.endpoint_tolerance,
            "adaptive_attack_minimum_quality_psnr": self.minimum_quality_psnr,
            "adaptive_attack_candidate_records": [row.as_dict() for row in self.candidates],
        }


def _read_video(path: Path) -> list[Any]:
    """解码完整视频帧, 保证候选确实由视频文件生成。"""

    import imageio.v3 as iio

    frames = [frame for frame in iio.imiter(path)]
    if not frames:
        raise ValueError(f"adaptive source video 无可解码帧: {path}")
    return frames


def _write_video(path: Path, frames: Sequence[Any], metadata: Mapping[str, Any]) -> None:
    """使用攻击协议给出的 codec 参数写出候选视频。"""

    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {"fps": 8}
    codec = metadata.get("video_writer_codec")
    output_params = metadata.get("video_writer_output_params")
    if codec:
        kwargs["codec"] = codec
    if output_params:
        kwargs["output_params"] = list(output_params)
    iio.imwrite(path, list(frames), **kwargs)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned_psnr(reference: Sequence[Any], candidate: Sequence[Any]) -> float:
    """按公共帧数计算 PSNR, 用作攻击可感知质量约束。"""

    import math
    import numpy as np

    count = min(len(reference), len(candidate))
    if count <= 0:
        return 0.0
    errors: list[float] = []
    for index in range(count):
        left = np.asarray(reference[index], dtype=np.float32)
        right = np.asarray(candidate[index], dtype=np.float32)
        if left.shape != right.shape:
            from PIL import Image

            right = np.asarray(
                Image.fromarray(right.astype(np.uint8)).resize(
                    (left.shape[1], left.shape[0])
                ),
                dtype=np.float32,
            )
        errors.append(float(np.mean((left - right) ** 2)))
    mse = sum(errors) / len(errors)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10((255.0**2) / mse))


def _candidate_from_score(
    *,
    candidate_index: int,
    attack_name: str,
    candidate_path: Path,
    decoded_candidate_frames: Sequence[Any],
    source_frames: Sequence[Any],
    score: Mapping[str, Any],
    objective: str,
    endpoint_reference: float,
    endpoint_tolerance: float,
    minimum_quality_psnr: float,
    attack_parameters: Mapping[str, float | int | str],
) -> AdaptiveVideoCandidate:
    """把一次真实文件查询转换为统一候选记录。"""

    endpoint_score = float(score["endpoint_score"])
    quality_psnr = _aligned_psnr(source_frames, decoded_candidate_frames)
    endpoint_ok = abs(endpoint_score - float(endpoint_reference)) <= float(
        endpoint_tolerance
    )
    quality_ok = quality_psnr >= float(minimum_quality_psnr)
    return AdaptiveVideoCandidate(
        candidate_index=int(candidate_index),
        attack_name=str(attack_name),
        video_path=str(candidate_path),
        video_sha256=_file_sha256(candidate_path),
        decoded_frame_count=len(decoded_candidate_frames),
        quality_psnr=round(quality_psnr, 8),
        detector_score=float(score["S_final_conservative"]),
        detector_score_source=str(
            score.get("flow_detector_score_source") or "unspecified_test_scorer"
        ),
        frozen_final_score_threshold=(
            float(score["frozen_final_score_threshold"])
            if score.get("frozen_final_score_threshold") is not None
            else None
        ),
        threshold_source_split=(
            str(score["threshold_source_split"])
            if score.get("threshold_source_split") is not None
            else None
        ),
        test_time_threshold_update_blocked=(
            score.get("test_time_threshold_update_blocked") is True
        ),
        endpoint_score=endpoint_score,
        path_score=float(score["S_path_inv"]),
        decision=bool(score["decision"]),
        admissible=bool(
            quality_ok
            and (
                endpoint_ok
                if objective == "minimize_path_with_fixed_endpoint"
                else True
            )
        ),
        attack_parameters=dict(attack_parameters),
    )


def _candidate_order_key(
    candidate: AdaptiveVideoCandidate,
    objective: str,
) -> tuple[float, float, int]:
    """返回预注册目标对应的候选排序键。"""

    if objective == "minimize_path_with_fixed_endpoint":
        return (
            candidate.path_score,
            candidate.detector_score,
            candidate.candidate_index,
        )
    return (
        candidate.detector_score,
        candidate.path_score,
        candidate.candidate_index,
    )


def _next_bounded_parameter(
    candidates: Sequence[AdaptiveVideoCandidate],
    queried_parameters: Sequence[float],
    *,
    objective: str,
    lower_bound: float,
    upper_bound: float,
) -> float:
    """依据已有冻结检测器查询选择下一次连续参数。

    前两次查询边界, 后续围绕当前最优可接受候选二分尚未探索的最大邻域。
    因此后续候选确实依赖此前 detector 输出, 而不是预先固定的攻击列表。
    """

    if not queried_parameters:
        return float(lower_bound)
    if len(queried_parameters) == 1:
        return float(upper_bound)
    feasible = [candidate for candidate in candidates if candidate.admissible]
    ranked = feasible or list(candidates)
    best = min(ranked, key=lambda row: _candidate_order_key(row, objective))
    best_parameter = float(best.attack_parameters["attack_strength"])
    points = sorted(
        set([float(lower_bound), float(upper_bound), *map(float, queried_parameters)])
    )
    best_index = min(
        range(len(points)),
        key=lambda index: abs(points[index] - best_parameter),
    )
    intervals: list[tuple[float, float]] = []
    if best_index > 0:
        intervals.append((points[best_index - 1], points[best_index]))
    if best_index + 1 < len(points):
        intervals.append((points[best_index], points[best_index + 1]))
    intervals.extend(zip(points[:-1], points[1:]))
    queried = {round(float(value), 12) for value in queried_parameters}
    for left, right in sorted(
        intervals,
        key=lambda item: (item[1] - item[0], -item[0]),
        reverse=True,
    ):
        midpoint = (left + right) / 2.0
        if round(midpoint, 12) not in queried and right - left > 1e-8:
            return midpoint
    raise RuntimeError("adaptive bounded search 已耗尽可区分参数")


def _parameterized_attack_frames(
    source_frames: Sequence[Any],
    *,
    attack_family: str,
    strength: float,
) -> tuple[list[Any], dict[str, Any]]:
    """按连续强度生成 detector-aware 视频攻击候选。"""

    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    value = max(0.0, min(1.0, float(strength)))
    frames = [np.asarray(frame)[..., :3].astype(np.uint8) for frame in source_frames]
    if attack_family == "endpoint_path_perturbation":
        alpha = 0.05 + 0.45 * value
        attacked = []
        for index, frame in enumerate(frames):
            neighbour = frames[min(index + 1, len(frames) - 1)]
            attacked.append(
                np.clip((1.0 - alpha) * frame + alpha * neighbour, 0, 255)
                .astype(np.uint8)
            )
        return attacked, {"temporal_blend_alpha": alpha}
    if attack_family == "public_detector_probe":
        brightness = 1.0 + 0.16 * value
        contrast = 1.0 + 0.20 * value
        attacked = [
            np.asarray(
                ImageEnhance.Contrast(
                    ImageEnhance.Brightness(Image.fromarray(frame)).enhance(brightness)
                ).enhance(contrast)
            )
            for frame in frames
        ]
        return attacked, {
            "brightness_factor": brightness,
            "contrast_factor": contrast,
        }
    if attack_family == "watermark_removal":
        radius = 0.25 + 2.75 * value
        quantization_step = max(1, int(round(1 + 15 * value)))
        attacked = []
        for frame in frames:
            filtered = np.asarray(
                Image.fromarray(frame).filter(ImageFilter.GaussianBlur(radius=radius))
            )
            attacked.append(
                (filtered // quantization_step * quantization_step).astype(np.uint8)
            )
        return attacked, {
            "gaussian_blur_radius": radius,
            "quantization_step": quantization_step,
        }
    if attack_family == "detector_evasion":
        crop_ratio = 1.0 - 0.24 * value
        noise_sigma = 1.0 + 7.0 * value
        generator = np.random.default_rng(17_071 + int(round(value * 1_000_000)))
        attacked = []
        for frame in frames:
            height, width = frame.shape[:2]
            crop_h = max(1, int(round(height * crop_ratio)))
            crop_w = max(1, int(round(width * crop_ratio)))
            top = (height - crop_h) // 2
            left = (width - crop_w) // 2
            cropped = frame[top : top + crop_h, left : left + crop_w]
            restored = np.asarray(
                Image.fromarray(cropped).resize((width, height), Image.Resampling.BICUBIC)
            ).astype(np.float32)
            noise = generator.normal(0.0, noise_sigma, size=restored.shape)
            attacked.append(np.clip(restored + noise, 0, 255).astype(np.uint8))
        return attacked, {
            "crop_ratio": crop_ratio,
            "noise_sigma": noise_sigma,
        }
    raise ValueError(f"未注册的 parameterized adaptive attack family: {attack_family}")


def optimize_bounded_parameter_attack_for_video(
    source_video_path: str | Path,
    output_dir: str | Path,
    *,
    attack_family: str,
    scorer: Callable[[Path], Mapping[str, Any]],
    objective: str,
    endpoint_reference: float,
    endpoint_tolerance: float = 0.08,
    minimum_quality_psnr: float = 24.0,
    query_budget: int = 5,
    initial_strength: float | None = None,
) -> AdaptiveVideoOptimizationResult:
    """用冻结检测器反馈逐次优化一个连续攻击强度参数。"""

    if objective not in {"minimize_detector_score", "minimize_path_with_fixed_endpoint"}:
        raise ValueError(f"未注册的 adaptive objective: {objective}")
    budget = int(query_budget)
    if budget < 3:
        raise ValueError("连续 adaptive optimization 至少需要3次查询")
    source_path = Path(source_video_path)
    source_frames = _read_video(source_path)
    destination = Path(output_dir)
    candidates: list[AdaptiveVideoCandidate] = []
    parameters: list[float] = []
    for index in range(budget):
        seeded_parameters = (
            [max(0.0, min(1.0, float(initial_strength))), 0.0, 1.0]
            if initial_strength is not None
            else []
        )
        unqueried_seed = next(
            (
                value
                for value in seeded_parameters
                if round(value, 12)
                not in {round(parameter, 12) for parameter in parameters}
            ),
            None,
        )
        if unqueried_seed is not None:
            strength = unqueried_seed
        else:
            strength = _next_bounded_parameter(
                candidates,
                parameters,
                objective=objective,
                lower_bound=0.0,
                upper_bound=1.0,
            )
        parameters.append(strength)
        attacked_frames, family_parameters = _parameterized_attack_frames(
            source_frames,
            attack_family=attack_family,
            strength=strength,
        )
        candidate_path = destination / (
            f"candidate_{index:03d}_{attack_family}_{strength:.6f}.mp4"
        )
        _write_video(candidate_path, attacked_frames, {})
        decoded = _read_video(candidate_path)
        score = dict(scorer(candidate_path))
        candidates.append(_candidate_from_score(
            candidate_index=index,
            attack_name=f"{attack_family}_bounded_parameter_search",
            candidate_path=candidate_path,
            decoded_candidate_frames=decoded,
            source_frames=source_frames,
            score=score,
            objective=objective,
            endpoint_reference=endpoint_reference,
            endpoint_tolerance=endpoint_tolerance,
            minimum_quality_psnr=minimum_quality_psnr,
            attack_parameters={
                "attack_strength": strength,
                **family_parameters,
            },
        ))
    feasible = [row for row in candidates if row.admissible]
    if not feasible:
        raise RuntimeError("连续 adaptive optimization 没有满足质量与 endpoint 约束的候选")
    selected = min(feasible, key=lambda row: _candidate_order_key(row, objective))
    return AdaptiveVideoOptimizationResult(
        objective=objective,
        selected=selected,
        candidates=tuple(candidates),
        endpoint_reference=float(endpoint_reference),
        endpoint_tolerance=float(endpoint_tolerance),
        minimum_quality_psnr=float(minimum_quality_psnr),
        query_budget=budget,
    )


def write_model_vae_regeneration_candidate(
    pipeline: Any,
    source_video_path: str | Path,
    output_path: str | Path,
    *,
    latent_noise_ratio: float,
    random_seed: int,
) -> dict[str, Any]:
    """执行真实模型 VAE encode-perturb-decode 生成式重压缩攻击。"""

    import torch

    from main.methods.state_space_watermark.endpoint_latent_detector import (
        _retrieve_vae_latent,
        load_video_tensor_for_wan_vae,
    )

    vae = pipeline.vae
    if vae is None:
        raise RuntimeError("生成式重压缩攻击要求 pipeline.vae 可用")
    device = pipeline._execution_device
    dtype = vae.dtype
    video, source_frame_count = load_video_tensor_for_wan_vae(
        source_video_path,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        latent = _retrieve_vae_latent(vae.encode(video))
    latent = latent.to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(int(random_seed))
    noise = torch.randn(
        latent.shape,
        generator=generator,
        device=device,
        dtype=latent.dtype,
    )
    scale = latent.std(unbiased=False).clamp_min(1e-8) * float(latent_noise_ratio)
    perturbed = latent + noise * scale
    with torch.inference_mode():
        decoded_output = vae.decode(perturbed.to(dtype=dtype), return_dict=False)
    decoded = decoded_output[0] if isinstance(decoded_output, tuple) else decoded_output
    if hasattr(decoded, "sample"):
        decoded = decoded.sample
    if decoded.ndim != 5:
        raise RuntimeError("VAE regeneration 输出必须为 [B, C, T, H, W]")
    array = (
        ((decoded[0].detach().float().clamp(-1.0, 1.0) + 1.0) * 127.5)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
        .round()
        .astype("uint8")
    )
    path = Path(output_path)
    _write_video(path, list(array), {})
    return {
        "model_vae_regeneration_status": "measured_model_vae_encode_perturb_decode",
        "model_vae_class": type(vae).__name__,
        "model_vae_latent_noise_ratio": float(latent_noise_ratio),
        "model_vae_random_seed_random": int(random_seed),
        "model_vae_noise_direction_policy": (
            "fixed_per_source_video_across_strength_queries"
        ),
        "model_vae_source_frame_count": int(source_frame_count),
        "model_vae_output_frame_count": int(len(array)),
    }


def optimize_model_vae_regeneration_attack_for_video(
    pipeline: Any,
    source_video_path: str | Path,
    output_dir: str | Path,
    *,
    scorer: Callable[[Path], Mapping[str, Any]],
    endpoint_reference: float,
    endpoint_tolerance: float = 0.08,
    minimum_quality_psnr: float = 24.0,
    query_budget: int = 5,
) -> AdaptiveVideoOptimizationResult:
    """用冻结检测器反馈优化真实模型 VAE regeneration 的 latent 噪声强度。"""

    budget = int(query_budget)
    if budget < 3:
        raise ValueError("VAE regeneration adaptive optimization 至少需要3次查询")
    source_path = Path(source_video_path)
    source_frames = _read_video(source_path)
    destination = Path(output_dir)
    candidates: list[AdaptiveVideoCandidate] = []
    strengths: list[float] = []
    # 同一视频的所有强度查询复用一个确定性噪声方向, 只改变 latent 噪声幅度。
    # 这样 detector feedback 优化的是连续强度, 不会把随机方向变化混入目标。
    regeneration_seed = int(_file_sha256(source_path)[:8], 16)
    for index in range(budget):
        normalized_strength = _next_bounded_parameter(
            candidates,
            strengths,
            objective="minimize_detector_score",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        strengths.append(normalized_strength)
        noise_ratio = 0.08 * normalized_strength
        candidate_path = destination / (
            f"candidate_{index:03d}_model_vae_regeneration_{noise_ratio:.6f}.mp4"
        )
        generation = write_model_vae_regeneration_candidate(
            pipeline,
            source_path,
            candidate_path,
            latent_noise_ratio=noise_ratio,
            random_seed=regeneration_seed,
        )
        decoded = _read_video(candidate_path)
        score = dict(scorer(candidate_path))
        candidates.append(_candidate_from_score(
            candidate_index=index,
            attack_name="model_vae_regeneration_bounded_parameter_search",
            candidate_path=candidate_path,
            decoded_candidate_frames=decoded,
            source_frames=source_frames,
            score=score,
            objective="minimize_detector_score",
            endpoint_reference=endpoint_reference,
            endpoint_tolerance=endpoint_tolerance,
            minimum_quality_psnr=minimum_quality_psnr,
            attack_parameters={
                "attack_strength": normalized_strength,
                "latent_noise_ratio": noise_ratio,
                **generation,
            },
        ))
    feasible = [row for row in candidates if row.admissible]
    if not feasible:
        raise RuntimeError("VAE regeneration 没有满足质量约束的候选")
    selected = min(
        feasible,
        key=lambda row: _candidate_order_key(row, "minimize_detector_score"),
    )
    return AdaptiveVideoOptimizationResult(
        objective="minimize_detector_score",
        selected=selected,
        candidates=tuple(candidates),
        endpoint_reference=float(endpoint_reference),
        endpoint_tolerance=float(endpoint_tolerance),
        minimum_quality_psnr=float(minimum_quality_psnr),
        query_budget=budget,
    )


def optimize_adaptive_attack_for_video(
    source_video_path: str | Path,
    output_dir: str | Path,
    *,
    candidate_attack_names: Sequence[str],
    scorer: Callable[[Path], Mapping[str, Any]],
    objective: str,
    endpoint_reference: float,
    endpoint_tolerance: float = 0.08,
    minimum_quality_psnr: float = 24.0,
    query_budget: int | None = None,
) -> AdaptiveVideoOptimizationResult:
    """逐候选生成、落盘并查询同一视频的冻结检测器。

    该函数属于通用黑盒优化结构。项目特定部分是目标函数: removal/evasion
    最小化完整概率后验, endpoint-preserving path attack 在 endpoint 容差内最小化
    path evidence。候选筛选只使用冻结检测输出与无标签质量约束, 不读取 test label。
    """

    if objective not in {"minimize_detector_score", "minimize_path_with_fixed_endpoint"}:
        raise ValueError(f"未注册的 adaptive objective: {objective}")
    names = [str(name) for name in candidate_attack_names]
    if query_budget is not None:
        names = names[: max(0, int(query_budget))]
    if not names:
        raise ValueError("adaptive optimization 至少需要一个候选变换")
    source_path = Path(source_video_path)
    source_frames = _read_video(source_path)
    destination = Path(output_dir)
    candidates: list[AdaptiveVideoCandidate] = []
    for index, attack_name in enumerate(names):
        attacked_frames, metadata = apply_runtime_attack_to_frames(source_frames, attack_name)
        candidate_path = destination / f"candidate_{index:03d}_{attack_name}.mp4"
        _write_video(candidate_path, attacked_frames, metadata)
        decoded_candidate_frames = _read_video(candidate_path)
        score = dict(scorer(candidate_path))
        endpoint_score = float(score["endpoint_score"])
        quality_psnr = _aligned_psnr(source_frames, decoded_candidate_frames)
        endpoint_ok = abs(endpoint_score - float(endpoint_reference)) <= float(endpoint_tolerance)
        quality_ok = quality_psnr >= float(minimum_quality_psnr)
        candidates.append(AdaptiveVideoCandidate(
            candidate_index=index,
            attack_name=attack_name,
            video_path=str(candidate_path),
            video_sha256=_file_sha256(candidate_path),
            decoded_frame_count=len(decoded_candidate_frames),
            quality_psnr=round(quality_psnr, 8),
            detector_score=float(score["S_final_conservative"]),
            detector_score_source=str(score.get("flow_detector_score_source") or "unspecified_test_scorer"),
            frozen_final_score_threshold=(
                float(score["frozen_final_score_threshold"])
                if score.get("frozen_final_score_threshold") is not None
                else None
            ),
            threshold_source_split=(
                str(score["threshold_source_split"])
                if score.get("threshold_source_split") is not None
                else None
            ),
            test_time_threshold_update_blocked=(
                score.get("test_time_threshold_update_blocked") is True
            ),
            endpoint_score=endpoint_score,
            path_score=float(score["S_path_inv"]),
            decision=bool(score["decision"]),
            admissible=bool(
                quality_ok
                and (
                    endpoint_ok
                    if objective == "minimize_path_with_fixed_endpoint"
                    else True
                )
            ),
            attack_parameters={"registered_attack_name": attack_name},
        ))
    feasible = [row for row in candidates if row.admissible]
    if not feasible:
        raise RuntimeError("adaptive optimization 没有满足预注册质量与 endpoint 约束的候选")
    if objective == "minimize_path_with_fixed_endpoint":
        selected = min(feasible, key=lambda row: (row.path_score, row.detector_score, row.candidate_index))
    else:
        selected = min(feasible, key=lambda row: (row.detector_score, row.candidate_index))
    return AdaptiveVideoOptimizationResult(
        objective=objective,
        selected=selected,
        candidates=tuple(candidates),
        endpoint_reference=float(endpoint_reference),
        endpoint_tolerance=float(endpoint_tolerance),
        minimum_quality_psnr=float(minimum_quality_psnr),
        query_budget=len(names),
    )


def write_cross_video_blend(
    primary_video_path: str | Path,
    secondary_video_path: str | Path,
    output_path: str | Path,
    *,
    secondary_weight: float,
) -> dict[str, Any]:
    """把两个真实视频逐帧混合, 用于 copy/spoof 与不同 key collusion。

    `secondary_weight` 必须显式预注册。函数按较短视频长度对齐, 并把次视频帧
    调整到主视频空间尺寸。该实现产生新的可哈希视频文件, 不会用已有 record
    的分数冒充跨视频攻击结果。
    """

    import numpy as np
    from PIL import Image

    weight = float(secondary_weight)
    if not 0.0 < weight < 1.0:
        raise ValueError("cross-video blend 权重必须位于 (0, 1)")
    primary = _read_video(Path(primary_video_path))
    secondary = _read_video(Path(secondary_video_path))
    count = min(len(primary), len(secondary))
    blended: list[Any] = []
    for index in range(count):
        left = np.asarray(primary[index], dtype=np.float32)
        right = np.asarray(secondary[index], dtype=np.float32)
        if right.shape != left.shape:
            right = np.asarray(
                Image.fromarray(right.astype(np.uint8)).resize(
                    (left.shape[1], left.shape[0])
                ),
                dtype=np.float32,
            )
        blended.append(np.clip((1.0 - weight) * left + weight * right, 0.0, 255.0).astype(np.uint8))
    path = Path(output_path)
    _write_video(path, blended, {})
    decoded_blend = _read_video(path)
    return {
        "adaptive_attack_output_video_path": str(path),
        "adaptive_attack_output_video_sha256": _file_sha256(path),
        "adaptive_attack_output_quality_psnr": round(
            _aligned_psnr(primary, decoded_blend),
            8,
        ),
        "adaptive_attack_output_decoded_frame_count": len(decoded_blend),
        "adaptive_attack_cross_video_weight": weight,
        "adaptive_attack_aligned_frame_count": count,
    }
