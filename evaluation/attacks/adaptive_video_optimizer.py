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
