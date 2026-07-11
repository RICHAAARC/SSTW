"""对单个真实视频执行可复现的黑盒 adaptive attack 优化。"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from evaluation.attacks.video_runtime_attack_protocol import apply_runtime_attack_to_frames


ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL = (
    "two_coordinate_detector_feedback_pattern_search_v1"
)


_ATTACK_FAMILY_COORDINATE_NAMES: dict[str, tuple[str, str]] = {
    "endpoint_path_perturbation": (
        "normalized_temporal_blend_strength",
        "normalized_temporal_offset_strength",
    ),
    "public_detector_probe": (
        "normalized_brightness_strength",
        "normalized_contrast_strength",
    ),
    "watermark_removal": (
        "normalized_gaussian_blur_strength",
        "normalized_quantization_strength",
    ),
    "detector_evasion": (
        "normalized_crop_strength",
        "normalized_noise_strength",
    ),
}


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
    replay_likelihood_model_id: str | None = None
    replay_likelihood_calibration_protocol: str | None = None
    replay_likelihood_calibration_cluster_count: int = 0
    replay_relative_observation_noise_standard_deviation: float | None = None
    adaptive_search_protocol: str | None = None
    adaptive_search_query_phase: str | None = None
    adaptive_search_coordinate_1_name: str | None = None
    adaptive_search_coordinate_1_value: float | None = None
    adaptive_search_coordinate_2_name: str | None = None
    adaptive_search_coordinate_2_value: float | None = None
    adaptive_search_feedback_parent_candidate_index: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """转换为可写入 governed query log 的字段。"""

        payload = dict(self.__dict__)
        for field_name in (
            "adaptive_search_protocol",
            "adaptive_search_query_phase",
            "adaptive_search_coordinate_1_name",
            "adaptive_search_coordinate_1_value",
            "adaptive_search_coordinate_2_name",
            "adaptive_search_coordinate_2_value",
            "adaptive_search_feedback_parent_candidate_index",
        ):
            if payload.get(field_name) is None:
                payload.pop(field_name, None)
        return payload


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
    adaptive_search_protocol: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """转换为正式 adaptive execution record 可复用的摘要。"""

        payload = {
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
            "adaptive_attack_replay_likelihood_model_id": (
                self.selected.replay_likelihood_model_id
            ),
            "adaptive_attack_replay_likelihood_calibration_protocol": (
                self.selected.replay_likelihood_calibration_protocol
            ),
            "adaptive_attack_replay_likelihood_calibration_cluster_count": (
                self.selected.replay_likelihood_calibration_cluster_count
            ),
            "adaptive_attack_replay_relative_observation_noise_standard_deviation": (
                self.selected.replay_relative_observation_noise_standard_deviation
            ),
        }
        if self.adaptive_search_protocol is not None:
            payload.update({
                "adaptive_search_protocol": self.adaptive_search_protocol,
                "adaptive_search_coordinate_names": [
                    self.selected.adaptive_search_coordinate_1_name,
                    self.selected.adaptive_search_coordinate_2_name,
                ],
            })
        return payload


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
    adaptive_search_protocol: str | None = None,
    adaptive_search_query_phase: str | None = None,
    adaptive_search_coordinate_1_name: str | None = None,
    adaptive_search_coordinate_1_value: float | None = None,
    adaptive_search_coordinate_2_name: str | None = None,
    adaptive_search_coordinate_2_value: float | None = None,
    adaptive_search_feedback_parent_candidate_index: int | None = None,
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
        replay_likelihood_model_id=(
            str(score["replay_likelihood_model_id"])
            if score.get("replay_likelihood_model_id") is not None
            else None
        ),
        replay_likelihood_calibration_protocol=(
            str(score["replay_likelihood_calibration_protocol"])
            if score.get("replay_likelihood_calibration_protocol") is not None
            else None
        ),
        replay_likelihood_calibration_cluster_count=int(
            score.get("replay_likelihood_calibration_cluster_count") or 0
        ),
        replay_relative_observation_noise_standard_deviation=(
            float(score["replay_relative_observation_noise_standard_deviation"])
            if score.get("replay_relative_observation_noise_standard_deviation")
            is not None
            else None
        ),
        adaptive_search_protocol=adaptive_search_protocol,
        adaptive_search_query_phase=adaptive_search_query_phase,
        adaptive_search_coordinate_1_name=adaptive_search_coordinate_1_name,
        adaptive_search_coordinate_1_value=(
            float(adaptive_search_coordinate_1_value)
            if adaptive_search_coordinate_1_value is not None
            else None
        ),
        adaptive_search_coordinate_2_name=adaptive_search_coordinate_2_name,
        adaptive_search_coordinate_2_value=(
            float(adaptive_search_coordinate_2_value)
            if adaptive_search_coordinate_2_value is not None
            else None
        ),
        adaptive_search_feedback_parent_candidate_index=(
            int(adaptive_search_feedback_parent_candidate_index)
            if adaptive_search_feedback_parent_candidate_index is not None
            else None
        ),
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


def _bounded_coordinate(value: float) -> float:
    """把搜索坐标限制到预注册的单位正方形。"""

    return max(0.0, min(1.0, float(value)))


def _legacy_attack_strength(coordinates: tuple[float, float]) -> float:
    """把二维坐标压缩为兼容旧记录的单个强度字段。

    `attack_strength` 不再驱动两个原生参数, 仅作为旧统计代码需要的标量摘要。
    这里使用无理数权重, 可避免常用二进制有理网格上的两个不同坐标被压成
    同一个值。正式复现实验应以两个显式坐标字段为准。
    """

    square_root_two = 2.0**0.5
    first, second = coordinates
    return float((square_root_two * first + second) / (square_root_two + 1.0))


def _initial_two_coordinate_seeds(
    initial_strength: float | None,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """返回基点和两个独立坐标探针, 共消耗3次冻结检测器查询。"""

    base = _bounded_coordinate(initial_strength if initial_strength is not None else 0.0)
    coordinate_target = 1.0 if base < 0.5 else 0.0
    return (
        (base, base),
        (coordinate_target, base),
        (base, coordinate_target),
    )


def _candidate_search_coordinates(
    candidate: AdaptiveVideoCandidate,
) -> tuple[float, float]:
    """从候选记录读取已真实执行的两个归一化搜索坐标。"""

    if (
        candidate.adaptive_search_coordinate_1_value is not None
        and candidate.adaptive_search_coordinate_2_value is not None
    ):
        return (
            _bounded_coordinate(candidate.adaptive_search_coordinate_1_value),
            _bounded_coordinate(candidate.adaptive_search_coordinate_2_value),
        )
    parameters = candidate.attack_parameters
    if (
        parameters.get("adaptive_search_coordinate_1_value") is not None
        and parameters.get("adaptive_search_coordinate_2_value") is not None
    ):
        return (
            _bounded_coordinate(float(parameters["adaptive_search_coordinate_1_value"])),
            _bounded_coordinate(float(parameters["adaptive_search_coordinate_2_value"])),
        )
    raise ValueError("二维 adaptive candidate 缺少可审计搜索坐标")


def _next_two_coordinate_parameters(
    candidates: Sequence[AdaptiveVideoCandidate],
    queried_coordinates: Sequence[tuple[float, float]],
    *,
    objective: str,
    initial_strength: float | None,
) -> tuple[tuple[float, float], str, int | None]:
    """选择下一组二维攻击参数及其反馈父候选。

    前3次查询固定为基点、坐标1探针和坐标2探针。之后每次先按
    admissibility 过滤, 再使用此前冻结检测器分数选出 incumbent, 最后只沿
    一个坐标执行有界 pattern refinement。由此, 后续查询同时依赖检测器输出
    与质量/endpoint 可接受性, 而不是由预制强度列表决定。
    """

    seeds = _initial_two_coordinate_seeds(initial_strength)
    query_index = len(queried_coordinates)
    if query_index < len(seeds):
        phases = ("base_point", "coordinate_1_probe", "coordinate_2_probe")
        return seeds[query_index], phases[query_index], None
    if len(candidates) != query_index:
        raise ValueError("二维 adaptive 搜索要求每个已查询坐标都有 detector 结果")

    admissible = [candidate for candidate in candidates if candidate.admissible]
    ranked = sorted(
        admissible or list(candidates),
        key=lambda row: _candidate_order_key(row, objective),
    )
    if not ranked:
        raise RuntimeError("二维 adaptive 搜索缺少可用于反馈更新的候选")

    queried = {
        (round(float(first), 12), round(float(second), 12))
        for first, second in queried_coordinates
    }
    # 每两次自适应查询缩小一次坐标步长。正式5次查询时先执行0.5尺度的
    # 两次反馈更新, 更高预算可继续在同一协议下局部细化。
    adaptive_query_index = query_index - len(seeds)
    primary_step = 0.5 / (2.0 ** (adaptive_query_index // 2))
    step_sizes = [primary_step / (2.0**level) for level in range(12)]

    for parent in ranked:
        anchor = _candidate_search_coordinates(parent)
        for step in step_sizes:
            # 先优化坐标2, 再优化坐标1。是否接受某个方向仍由 incumbent
            # 的 detector 分数决定; 已查询方向会被跳过, 从而形成序贯细化。
            for coordinate_index in (1, 0):
                current = anchor[coordinate_index]
                directions = (1.0, -1.0) if current <= 0.5 else (-1.0, 1.0)
                for direction in directions:
                    proposal = list(anchor)
                    proposal[coordinate_index] = _bounded_coordinate(
                        current + direction * step
                    )
                    coordinates = (float(proposal[0]), float(proposal[1]))
                    key = (round(coordinates[0], 12), round(coordinates[1], 12))
                    if key not in queried and coordinates != anchor:
                        return (
                            coordinates,
                            "detector_feedback_pattern_refinement",
                            int(parent.candidate_index),
                        )
    raise RuntimeError("二维 adaptive 搜索已耗尽可区分坐标")


def _parameterized_attack_frames(
    source_frames: Sequence[Any],
    *,
    attack_family: str,
    strength: float,
    parameter_coordinates: Sequence[float] | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """按两个独立连续坐标生成 detector-aware 视频攻击候选。

    `strength` 仅保留给旧调用者。正式二维优化会传入
    `parameter_coordinates`, 并分别控制攻击族的两个原生参数。
    """

    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    if parameter_coordinates is None:
        first_value = second_value = _bounded_coordinate(strength)
    else:
        if len(parameter_coordinates) != 2:
            raise ValueError("parameter_coordinates 必须恰好包含两个归一化坐标")
        first_value = _bounded_coordinate(float(parameter_coordinates[0]))
        second_value = _bounded_coordinate(float(parameter_coordinates[1]))
    frames = [np.asarray(frame)[..., :3].astype(np.uint8) for frame in source_frames]
    if attack_family == "endpoint_path_perturbation":
        alpha = 0.05 + 0.45 * first_value
        temporal_offset_frames = 1 + int(round(2.0 * second_value))
        attacked = []
        for index, frame in enumerate(frames):
            neighbour = frames[
                min(index + temporal_offset_frames, len(frames) - 1)
            ]
            attacked.append(
                np.clip((1.0 - alpha) * frame + alpha * neighbour, 0, 255)
                .astype(np.uint8)
            )
        return attacked, {
            "temporal_blend_alpha": alpha,
            "temporal_offset_frames": temporal_offset_frames,
        }
    if attack_family == "public_detector_probe":
        brightness = 1.0 + 0.16 * first_value
        contrast = 1.0 + 0.20 * second_value
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
        radius = 0.25 + 2.75 * first_value
        quantization_step = max(1, int(round(1 + 15 * second_value)))
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
        crop_ratio = 1.0 - 0.24 * first_value
        noise_sigma = 1.0 + 7.0 * second_value
        # 固定噪声方向只让第二坐标改变噪声幅度, 避免把随机方向混入反馈。
        generator = np.random.default_rng(17_071)
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
    """用冻结检测器反馈逐次优化两个独立原生攻击参数。

    该函数保持原调用 API。`initial_strength` 只用于设置二维基点; 两个坐标
    探针及其后的 detector-feedback pattern refinement 均在函数内部完成。
    """

    if objective not in {"minimize_detector_score", "minimize_path_with_fixed_endpoint"}:
        raise ValueError(f"未注册的 adaptive objective: {objective}")
    budget = int(query_budget)
    if budget < 3:
        raise ValueError("二维 adaptive optimization 至少需要3次查询")
    if attack_family not in _ATTACK_FAMILY_COORDINATE_NAMES:
        raise ValueError(f"未注册的 parameterized adaptive attack family: {attack_family}")
    source_path = Path(source_video_path)
    source_frames = _read_video(source_path)
    destination = Path(output_dir)
    candidates: list[AdaptiveVideoCandidate] = []
    coordinates_by_query: list[tuple[float, float]] = []
    coordinate_names = _ATTACK_FAMILY_COORDINATE_NAMES[attack_family]
    for index in range(budget):
        coordinates, query_phase, feedback_parent_index = (
            _next_two_coordinate_parameters(
                candidates,
                coordinates_by_query,
                objective=objective,
                initial_strength=initial_strength,
            )
        )
        coordinates_by_query.append(coordinates)
        strength = _legacy_attack_strength(coordinates)
        attacked_frames, family_parameters = _parameterized_attack_frames(
            source_frames,
            attack_family=attack_family,
            strength=strength,
            parameter_coordinates=coordinates,
        )
        candidate_path = destination / (
            f"candidate_{index:03d}_{attack_family}_"
            f"{coordinates[0]:.6f}_{coordinates[1]:.6f}.mp4"
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
                "attack_strength_semantics": (
                    "sqrt2_weighted_mean_legacy_summary_not_search_control"
                ),
                "adaptive_search_protocol": ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
                "adaptive_search_query_phase": query_phase,
                "adaptive_search_coordinate_1_name": coordinate_names[0],
                "adaptive_search_coordinate_1_value": coordinates[0],
                "adaptive_search_coordinate_2_name": coordinate_names[1],
                "adaptive_search_coordinate_2_value": coordinates[1],
                **(
                    {
                        "adaptive_search_feedback_parent_candidate_index": (
                            feedback_parent_index
                        )
                    }
                    if feedback_parent_index is not None
                    else {}
                ),
                **family_parameters,
            },
            adaptive_search_protocol=ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
            adaptive_search_query_phase=query_phase,
            adaptive_search_coordinate_1_name=coordinate_names[0],
            adaptive_search_coordinate_1_value=coordinates[0],
            adaptive_search_coordinate_2_name=coordinate_names[1],
            adaptive_search_coordinate_2_value=coordinates[1],
            adaptive_search_feedback_parent_candidate_index=feedback_parent_index,
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
        adaptive_search_protocol=ADAPTIVE_TWO_COORDINATE_SEARCH_PROTOCOL,
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
