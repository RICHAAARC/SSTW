"""Read-only CPU analysis of the existing six saved Gate A diagnostic videos."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Any, Mapping
from zipfile import ZipFile

import imageio.v3 as iio
import numpy as np

from evaluation.protocol.existing_six_video_spatiotemporal_signed_response_contract import (
    CLAIM_SUPPORT_STATUS,
    DEFAULT_CONFIG_PATH,
    PLAN_IDS,
    PROFILE_ID,
    RECORD_VERSION,
    classify_spatiotemporal_signed_response,
    compute_spatiotemporal_signed_response,
    load_spatiotemporal_diagnostic_config,
    records_as_dicts,
    validate_existing_six_video_source,
)
from evaluation.protocol.impulse_observability_contract import (
    canonical_json_digest,
)
from evaluation.protocol.record_writer import write_json, write_jsonl


MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
MAX_ARCHIVE_TOTAL_BYTES = 256 * 1024 * 1024


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise ValueError("spatiotemporal source ZIP destination 必须为空")
    total_size = 0
    planned_members: list[tuple[Any, tuple[str, ...], bool]] = []
    normalized_targets: dict[tuple[str, ...], bool] = {}
    destination_root = destination.resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            normalized_parts = tuple(
                part for part in relative.parts if part not in ("", ".")
            )
            is_directory = member.is_dir()
            if (
                not member.filename
                or "\x00" in member.filename
                or "\\" in member.filename
                or relative.is_absolute()
                or ".." in relative.parts
                or not normalized_parts
                or stat.S_ISLNK(mode)
                or member.file_size > MAX_ARCHIVE_MEMBER_BYTES
            ):
                raise ValueError("spatiotemporal source ZIP member 非法")
            normalized_target = (
                destination / Path(*normalized_parts)
            ).resolve()
            if not normalized_target.is_relative_to(destination_root):
                raise ValueError("spatiotemporal source ZIP 路径越界")
            if normalized_parts in normalized_targets:
                raise ValueError(
                    "spatiotemporal source ZIP 规范化 target 重复"
                )
            normalized_targets[normalized_parts] = is_directory
            planned_members.append(
                (member, normalized_parts, is_directory)
            )
            total_size += member.file_size
            if total_size > MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError("spatiotemporal source ZIP 解压规模超限")

        for normalized_parts, is_directory in normalized_targets.items():
            for ancestor_stop in range(1, len(normalized_parts)):
                ancestor = normalized_parts[:ancestor_stop]
                if normalized_targets.get(ancestor) is False:
                    raise ValueError(
                        "spatiotemporal source ZIP file/directory 冲突"
                    )
            if not is_directory and any(
                len(other) > len(normalized_parts)
                and other[: len(normalized_parts)] == normalized_parts
                for other in normalized_targets
            ):
                raise ValueError(
                    "spatiotemporal source ZIP file/directory 冲突"
                )

        for member, normalized_parts, is_directory in planned_members:
            target = destination / Path(*normalized_parts)
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _discover_source_root(extracted_root: Path) -> Path:
    expected_manifest = (
        extracted_root
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_manifest.json"
    )
    manifests = list(
        extracted_root.rglob(
            "artifacts/gate_a_root_cause_amplitude_feedback_manifest.json"
        )
    )
    if len(manifests) != 1 or manifests[0] != expected_manifest:
        raise ValueError(
            "spatiotemporal source ZIP 必须在根目录唯一包含完整 "
            "root-cause manifest"
        )
    return extracted_root


def _read_rgb24_video(
    path: Path,
    *,
    expected_shape: tuple[int, int, int, int],
) -> np.ndarray:
    frames = tuple(np.asarray(frame) for frame in iio.imiter(path))
    if not frames:
        raise ValueError(f"saved video 没有可读帧: {path}")
    value = np.ascontiguousarray(np.stack(frames, axis=0))
    if value.shape != expected_shape or value.dtype != np.uint8:
        raise ValueError(
            f"saved video RGB24 shape/dtype 漂移: "
            f"{path.name} {value.shape}/{value.dtype}"
        )
    return value


def _authorization_false_fields() -> dict[str, bool]:
    return {
        "gate_a_retry": False,
        "gate_a_pass": False,
        "cross_identity_confirmation_allowed": False,
        "gate_b_execution_allowed": False,
        "gate_c_execution_allowed": False,
        "wrong_key_execution_allowed": False,
        "observer_execution_allowed": False,
        "state_dynamics_design_allowed": False,
        "frozen_feedback_diagnostic_execution_allowed": False,
        "gpu_execution_allowed": False,
        "colab_execution_allowed": False,
        "drive_update_allowed": False,
        "attack_execution_allowed": False,
        "fixed_fpr_execution_allowed": False,
        "external_baseline_execution_allowed": False,
        "paper_claim_allowed": False,
        "automatic_followup_execution_allowed": False,
    }


def _run_from_directory(
    source_root: Path,
    output_root: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source = validate_existing_six_video_source(source_root, config)
    plan = config["analysis_plan"]
    expected_shape = (
        int(plan["frame_count"]),
        int(plan["height"]),
        int(plan["width"]),
        int(plan["channel_count"]),
    )
    frames_by_probe = {
        probe_id: _read_rgb24_video(
            source.video_paths[probe_id],
            expected_shape=expected_shape,
        )
        for probe_id in PLAN_IDS
    }
    records = compute_spatiotemporal_signed_response(
        config,
        frames_by_probe,
    )
    classification = classify_spatiotemporal_signed_response(
        config,
        records,
    )
    config_digest = canonical_json_digest(config)
    governed_records = records_as_dicts(records)
    write_jsonl(
        output_root
        / "records"
        / "spatiotemporal_signed_response_statistics.jsonl",
        governed_records,
    )
    decision = {
        "record_version": RECORD_VERSION,
        "profile_id": PROFILE_ID,
        "spatiotemporal_signed_response_diagnostic_decision": (
            "existing_six_video_candidate_classification_recorded"
        ),
        "candidate_classification": classification,
        "source_repository_commit": source.decision["repository_commit"],
        "source_config_digest": source.decision["config_digest"],
        "source_snapshot_digest": source.source_snapshot_digest,
        "historical_gate_a_source_snapshot_digest": source.decision[
            "historical_source_snapshot_digest"
        ],
        "source_gate_a_pass_preserved": False,
        "source_clean_all_frozen_representations_equal": True,
        "spatiotemporal_record_count": len(governed_records),
        "config_digest": config_digest,
        **_authorization_false_fields(),
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    write_json(
        output_root
        / "artifacts"
        / "spatiotemporal_signed_response_decision.json",
        decision,
    )
    manifest = {
        "manifest_kind": (
            "existing_six_video_spatiotemporal_signed_response_manifest"
        ),
        "profile_id": PROFILE_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot_digest": source.source_snapshot_digest,
        "source_repository_commit": source.decision["repository_commit"],
        "config_path": DEFAULT_CONFIG_PATH,
        "config_digest": config_digest,
        "video_sha256_by_probe": config["source_binding"][
            "video_sha256_by_probe"
        ],
        "spatiotemporal_record_count": len(governed_records),
        "output_paths": [
            "records/spatiotemporal_signed_response_statistics.jsonl",
            "artifacts/spatiotemporal_signed_response_decision.json",
        ],
        "cpu_only": True,
        "source_read_only": True,
        "formal_result": False,
        "stage_progression_allowed": False,
        "claim_support_status": CLAIM_SUPPORT_STATUS,
    }
    write_json(
        output_root
        / "artifacts"
        / "spatiotemporal_signed_response_manifest.json",
        manifest,
    )
    return decision


def run_existing_six_video_spatiotemporal_signed_response_diagnostic(
    source_result: str | Path,
    output_root: str | Path,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """Analyze only the six immutable saved videos on CPU."""

    source_path = Path(source_result).resolve()
    output = Path(output_root).resolve()
    source_is_directory = source_path.is_dir()
    source_is_zip = (
        source_path.is_file() and source_path.suffix.lower() == ".zip"
    )
    if not source_is_directory and not source_is_zip:
        raise ValueError(
            "spatiotemporal source 必须是完整 result ZIP 或目录"
        )
    if source_is_directory and (
        output == source_path
        or output.is_relative_to(source_path)
        or source_path.is_relative_to(output)
    ):
        raise ValueError(
            "spatiotemporal directory source 与 output 必须完全隔离"
        )
    if output.exists():
        raise FileExistsError(
            "spatiotemporal diagnostic output root 已存在，拒绝覆盖"
        )
    config = load_spatiotemporal_diagnostic_config(config_path)
    output.mkdir(parents=True)
    try:
        if source_is_directory:
            return _run_from_directory(source_path, output, config)
        with tempfile.TemporaryDirectory(
            prefix="sstw_existing_six_video_"
        ) as temporary:
            extraction_root = Path(temporary)
            _safe_extract_zip(source_path, extraction_root)
            source_root = _discover_source_root(extraction_root)
            return _run_from_directory(source_root, output, config)
    except Exception as exc:
        failure = {
            "record_version": RECORD_VERSION,
            "profile_id": PROFILE_ID,
            "spatiotemporal_signed_response_diagnostic_decision": (
                "source_or_analysis_failure_stop"
            ),
            "failure_reason": str(exc),
            **_authorization_false_fields(),
            "formal_result": False,
            "stage_progression_allowed": False,
            "claim_support_status": (
                "failure_recovery_only_not_claim_evidence"
            ),
        }
        write_json(
            output
            / "artifacts"
            / "spatiotemporal_signed_response_decision.json",
            failure,
        )
        raise


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Analyze the immutable six-video Gate A diagnostic on CPU"
        )
    )
    parser.add_argument("--source-result", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args()
    decision = (
        run_existing_six_video_spatiotemporal_signed_response_diagnostic(
            args.source_result,
            args.output_root,
            config_path=args.config_path,
        )
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
