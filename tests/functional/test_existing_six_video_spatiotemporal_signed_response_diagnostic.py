from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZipInfo

import numpy as np
import pytest

from evaluation.protocol.existing_six_video_spatiotemporal_signed_response_contract import (
    CLAIM_SUPPORT_STATUS,
    DEFAULT_CONFIG_PATH,
    FROZEN_CONFIG_DIGEST,
    PLAN_IDS,
    SpatiotemporalSignedResponseRecord,
    canonical_json_digest,
    classify_frozen_feedback_signed_response_design,
    classify_spatiotemporal_signed_response,
    compute_spatiotemporal_signed_response,
    load_spatiotemporal_diagnostic_config,
    records_as_dicts,
    validate_existing_six_video_source,
)
from experiments.generative_video_model_probe.existing_six_video_spatiotemporal_signed_response_diagnostic import (
    _discover_source_root,
    _safe_extract_zip,
    run_existing_six_video_spatiotemporal_signed_response_diagnostic,
)


REAL_SOURCE_ROOT = Path(
    "/tmp/sstw_gate_a_root_cause_f06a0934.9OQY2p"
)
REAL_SOURCE_ZIP = Path("/tmp/sstw_gate_a_root_cause_f06a0934.zip")


def _config() -> dict:
    return load_spatiotemporal_diagnostic_config(DEFAULT_CONFIG_PATH)


def _small_config() -> dict:
    config = json.loads(json.dumps(_config()))
    config["analysis_plan"]["height"] = 2
    config["analysis_plan"]["width"] = 3
    return config


def _write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _symmetric_frames(config: dict) -> dict[str, np.ndarray]:
    plan = config["analysis_plan"]
    shape = (
        plan["frame_count"],
        plan["height"],
        plan["width"],
        plan["channel_count"],
    )
    clean = np.full(shape, 128, dtype=np.uint8)
    signal = np.empty(shape, dtype=np.int16)
    for frame_index in range(shape[0]):
        signal[frame_index].fill(5 + 2 * (frame_index % 3))
    positive = (clean.astype(np.int16) + signal).astype(np.uint8)
    negative = (clean.astype(np.int16) - signal).astype(np.uint8)
    return {
        "clean_start": clean.copy(),
        "positive_early_flow_channel_0": positive.copy(),
        "negative_early_flow_channel_0": negative.copy(),
        "positive_late_flow_channel_0": positive.copy(),
        "negative_late_flow_channel_0": negative.copy(),
        "clean_end": clean.copy(),
    }


@pytest.mark.quick
def test_config_digest_and_authorization_boundary_are_exact() -> None:
    config = _config()
    assert canonical_json_digest(config) == FROZEN_CONFIG_DIGEST
    assert config["cpu_only"] is True
    assert config["formal_result"] is False
    assert config["stage_progression_allowed"] is False
    assert all(
        value is False
        for value in config["authorization_boundary"].values()
    )
    assert config["signed_response_gate"] == {
        "minimum_antisymmetry_cosine": 0.9,
        "maximum_antisymmetry_residual": 0.25,
        "maximum_common_odd_ratio": 0.5,
        "minimum_odd_norm": 1e-12,
    }


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["signed_response_gate"].__setitem__(
            "minimum_antisymmetry_cosine", -999.0
        ),
        lambda value: value["analysis_plan"][
            "video_frame_intervals"
        ][0].__setitem__("frame_stop", 12),
        lambda value: value["analysis_plan"][
            "statistics_formulas"
        ].__setitem__("odd", "caller_chosen"),
        lambda value: value["source_binding"].__setitem__(
            "source_snapshot_digest", "0" * 64
        ),
        lambda value: value["classification_contract"].__setitem__(
            "single_frame_or_single_transition_pass_insufficient", False
        ),
        lambda value: value["authorization_boundary"].__setitem__(
            "gpu_execution_allowed", True
        ),
        lambda value: value.__setitem__(
            "claim_support_status", "method_evidence"
        ),
    ),
)
def test_every_semantic_config_mutation_fails_closed(
    tmp_path: Path,
    mutation,
) -> None:
    config = _config()
    mutation(config)
    path = tmp_path / "mutated.json"
    _write_config(path, config)
    with pytest.raises(ValueError, match="exact digest"):
        load_spatiotemporal_diagnostic_config(path)


@pytest.mark.quick
def test_exact_symmetric_rgb24_response_passes_all_interval_gates() -> None:
    config = _small_config()
    records = compute_spatiotemporal_signed_response(
        config,
        _symmetric_frames(config),
    )
    assert len(records) == 142
    interval_records = [
        record
        for record in records
        if record.representation_id
        in {
            "frame_interval_full_rgb24",
            "adjacent_difference_interval_full_rgb24",
        }
    ]
    assert len(interval_records) == 12
    assert all(record.signed_response_gate_passed for record in interval_records)
    assert all(
        record.antisymmetry_cosine == pytest.approx(1.0, abs=1e-12)
        for record in records
    )
    assert all(record.common_norm <= 1e-12 for record in records)
    classification = classify_spatiotemporal_signed_response(config, records)
    assert classification["temporal_feature_salvage_candidate"] is True
    assert classification["carrier_redesign_required_candidate"] is False
    assert classification["stable_interval_representation_families"] == [
        "frame_interval_full_rgb24",
        "adjacent_difference_interval_full_rgb24",
    ]
    assert classification["automatic_feature_selection_allowed"] is False


@pytest.mark.quick
def test_common_mode_and_single_frame_cannot_create_salvage_candidate() -> None:
    config = _small_config()
    frames = _symmetric_frames(config)
    for pair_id in ("early_flow_channel_0", "late_flow_channel_0"):
        frames[f"negative_{pair_id}"] = frames[f"positive_{pair_id}"].copy()
    records = list(compute_spatiotemporal_signed_response(config, frames))
    assert not any(
        record.signed_response_gate_passed
        for record in records
        if "interval" in record.representation_id
    )
    first_frame = next(
        index
        for index, record in enumerate(records)
        if record.representation_id == "per_frame_full_rgb24"
    )
    records[first_frame] = replace(
        records[first_frame],
        finite=True,
        signed_response_gate_passed=True,
    )
    classification = classify_spatiotemporal_signed_response(config, records)
    assert classification["temporal_feature_salvage_candidate"] is False
    assert classification["carrier_redesign_required_candidate"] is True
    assert (
        classification["single_frame_or_single_transition_pass_sufficient"]
        is False
    )


@pytest.mark.quick
def test_classifier_rejects_reordered_or_forged_record_coverage() -> None:
    config = _small_config()
    records = list(
        compute_spatiotemporal_signed_response(
            config,
            _symmetric_frames(config),
        )
    )
    records[0], records[1] = records[1], records[0]
    result = classify_spatiotemporal_signed_response(config, records)
    assert result["classification_status"] == "indeterminate"
    assert result["temporal_feature_salvage_candidate"] is False

    records = list(
        compute_spatiotemporal_signed_response(
            config,
            _symmetric_frames(config),
        )
    )
    records[-1] = replace(records[-1], segment_id="caller_chosen")
    result = classify_spatiotemporal_signed_response(config, records)
    assert result["classification_status"] == "indeterminate"
    assert result["carrier_redesign_required_candidate"] is False


@pytest.mark.quick
def test_clean_mismatch_and_frame_identity_fail_closed() -> None:
    config = _small_config()
    frames = _symmetric_frames(config)
    frames["clean_end"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="clean_start/end RGB24"):
        compute_spatiotemporal_signed_response(config, frames)

    frames = _symmetric_frames(config)
    reordered = {
        key: frames[key]
        for key in (
            PLAN_IDS[1],
            PLAN_IDS[0],
            *PLAN_IDS[2:],
        )
    }
    with pytest.raises(ValueError, match="identity/order"):
        compute_spatiotemporal_signed_response(config, reordered)


@pytest.mark.quick
def test_governed_records_are_nonformal_and_registry_bound() -> None:
    config = _small_config()
    records = compute_spatiotemporal_signed_response(
        config,
        _symmetric_frames(config),
    )
    row = records_as_dicts(records[:1])[0]
    assert row["formal_result"] is False
    assert row["stage_progression_allowed"] is False
    assert row["claim_support_status"] == CLAIM_SUPPORT_STATUS
    assert len(row["spatiotemporal_signed_response_record_id"]) == 64

    registry = (
        Path("docs/field_registry.md").read_text(encoding="utf-8")
    )
    registered = {
        line.split("|")[1].strip()
        for line in registry.splitlines()
        if line.startswith("| ") and len(line.split("|")) > 2
    }
    assert set(row).issubset(registered)


@pytest.mark.quick
def test_real_complete_six_video_source_validates_read_only() -> None:
    if not REAL_SOURCE_ROOT.is_dir():
        pytest.skip("real f06a0934 six-video result is not mounted")
    source = validate_existing_six_video_source(
        REAL_SOURCE_ROOT,
        _config(),
    )
    assert source.source_snapshot_digest == _config()["source_binding"][
        "source_snapshot_digest"
    ]
    assert tuple(source.video_paths) == PLAN_IDS
    assert source.decision["gate_a_pass"] is False


@pytest.mark.quick
@pytest.mark.parametrize(
    "mutation",
    ("video_bytes", "generation_order", "recovery_marker"),
)
def test_real_source_tamper_reorder_and_recovery_are_rejected(
    tmp_path: Path,
    mutation: str,
) -> None:
    if not REAL_SOURCE_ROOT.is_dir():
        pytest.skip("real f06a0934 six-video result is not mounted")
    source = tmp_path / "source"
    shutil.copytree(REAL_SOURCE_ROOT, source)
    if mutation == "video_bytes":
        path = source / "videos" / "00_clean_start.mp4"
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "generation_order":
        path = source / "records" / "impulse_generation_records.jsonl"
        rows = path.read_text(encoding="utf-8").splitlines()
        rows[0], rows[1] = rows[1], rows[0]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    elif mutation == "recovery_marker":
        (source / "recovery_marker.json").write_text(
            "{}\n", encoding="utf-8"
        )
    else:
        raise AssertionError(mutation)
    with pytest.raises(ValueError):
        validate_existing_six_video_source(source, _config())


@pytest.mark.quick
def test_safe_zip_extraction_rejects_traversal_and_symlink(
    tmp_path: Path,
) -> None:
    traversal = tmp_path / "traversal.zip"
    with ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.json", "{}")
    with pytest.raises(ValueError, match="member"):
        _safe_extract_zip(traversal, tmp_path / "traversal")

    symlink = tmp_path / "symlink.zip"
    info = ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with ZipFile(symlink, "w") as archive:
        archive.writestr(info, "target")
    with pytest.raises(ValueError, match="member"):
        _safe_extract_zip(symlink, tmp_path / "symlink")

    duplicate = tmp_path / "duplicate.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(duplicate, "w") as archive:
            archive.writestr("duplicate.json", "{}")
            archive.writestr("duplicate.json", '{"changed": true}')
    with pytest.raises(ValueError, match="规范化 target 重复"):
        _safe_extract_zip(duplicate, tmp_path / "duplicate")


@pytest.mark.quick
@pytest.mark.parametrize(
    ("first_name", "second_name"),
    (
        ("./x.json", "x.json"),
        ("a//b.json", "a/b.json"),
        ("node", "node/"),
        ("parent", "parent/child.json"),
    ),
)
def test_safe_zip_extraction_rejects_normalized_duplicates_before_write(
    tmp_path: Path,
    first_name: str,
    second_name: str,
) -> None:
    archive_path = tmp_path / "normalized_duplicate.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(first_name, "first")
        archive.writestr(second_name, "second")
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="重复|冲突"):
        _safe_extract_zip(archive_path, destination)
    assert not destination.exists()


@pytest.mark.quick
def test_safe_zip_extraction_rejects_dot_member_before_write(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "dot_member.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(".", "")
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="member"):
        _safe_extract_zip(archive_path, destination)
    assert not destination.exists()


@pytest.mark.quick
def test_real_package_normalized_manifest_duplicate_is_rejected_before_write(
    tmp_path: Path,
) -> None:
    if not REAL_SOURCE_ZIP.is_file():
        pytest.skip("real f06a0934 six-video result ZIP is not mounted")
    archive_path = tmp_path / "forged_duplicate.zip"
    manifest_path = (
        "artifacts/"
        "gate_a_root_cause_amplitude_feedback_manifest.json"
    )
    with ZipFile(REAL_SOURCE_ZIP) as source, ZipFile(
        archive_path, "w"
    ) as forged:
        forged.writestr(f"./{manifest_path}", '{"forged": true}')
        for member in source.infolist():
            forged.writestr(member, source.read(member))
    destination = tmp_path / "extracted"
    with pytest.raises(ValueError, match="规范化 target 重复"):
        _safe_extract_zip(archive_path, destination)
    assert not destination.exists()


@pytest.mark.quick
def test_source_root_discovery_rejects_wrapper_or_out_of_root_files(
    tmp_path: Path,
) -> None:
    wrapped = tmp_path / "wrapped"
    manifest = (
        wrapped
        / "result"
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="根目录"):
        _discover_source_root(wrapped)

    exact = tmp_path / "exact"
    manifest = (
        exact
        / "artifacts"
        / "gate_a_root_cause_amplitude_feedback_manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    assert _discover_source_root(exact) == exact


@pytest.mark.quick
def test_directory_source_cannot_write_output_inside_source(
    tmp_path: Path,
) -> None:
    if not REAL_SOURCE_ROOT.is_dir():
        pytest.skip("real f06a0934 six-video result is not mounted")
    source = tmp_path / "source"
    shutil.copytree(REAL_SOURCE_ROOT, source)
    before = tuple(
        path.relative_to(source).as_posix()
        for path in sorted(source.rglob("*"))
    )
    output = source / "analysis_output"
    with pytest.raises(ValueError, match="完全隔离"):
        run_existing_six_video_spatiotemporal_signed_response_diagnostic(
            source,
            output,
        )
    after = tuple(
        path.relative_to(source).as_posix()
        for path in sorted(source.rglob("*"))
    )
    assert before == after
    assert not output.exists()

    with pytest.raises(ValueError, match="完全隔离"):
        run_existing_six_video_spatiotemporal_signed_response_diagnostic(
            source,
            source,
        )
    with pytest.raises(ValueError, match="完全隔离"):
        run_existing_six_video_spatiotemporal_signed_response_diagnostic(
            source,
            source.parent,
        )

    alias = tmp_path / "source_alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="完全隔离"):
        run_existing_six_video_spatiotemporal_signed_response_diagnostic(
            source,
            alias / "symlink_output",
        )
    assert not (source / "symlink_output").exists()


@pytest.mark.quick
@pytest.mark.parametrize(
    (
        "clean_ready",
        "early_latent_signed",
        "late_latent_signed",
        "post_latent_signed",
        "expected_status",
        "expected_candidates",
    ),
    (
        (
            False,
            True,
            True,
            True,
            "indeterminate_stop",
            {"indeterminate_stop"},
        ),
        (
            True,
            True,
            True,
            True,
            "feedback_isolation_candidate",
            {"feedback_isolation_candidate"},
        ),
        (
            True,
            True,
            True,
            False,
            "multiple_candidates",
            {
                "feedback_isolation_candidate",
                "decoder_carrier_mismatch_candidate",
                "multiple_candidates",
            },
        ),
        (
            True,
            False,
            False,
            False,
            "stop_current_additive_random_carrier",
            {"stop_current_additive_random_carrier"},
        ),
        (
            True,
            True,
            False,
            False,
            "indeterminate_stop",
            {"indeterminate_stop"},
        ),
    ),
)
def test_frozen_feedback_design_table_is_mutually_exclusive(
    clean_ready: bool,
    early_latent_signed: bool,
    late_latent_signed: bool,
    post_latent_signed: bool,
    expected_status: str,
    expected_candidates: set[str],
) -> None:
    result = classify_frozen_feedback_signed_response_design(
        clean_coverage_and_guards_ready=clean_ready,
        early_full_final_latent_signed=early_latent_signed,
        late_full_final_latent_signed=late_latent_signed,
        all_post_latent_checkpoints_signed=post_latent_signed,
    )
    assert result["classification_status"] == expected_status
    assert set(result["candidate_classifications"]) == expected_candidates
    assert result["unique_root_cause_claim_allowed"] is False
    assert result["formal_result"] is False
    assert result["stage_progression_allowed"] is False


@pytest.mark.quick
def test_frozen_feedback_spec_is_design_only_and_has_exact_outputs() -> None:
    text = Path(
        "docs/builds/frozen_feedback_signed_response_diagnostic.md"
    ).read_text(encoding="utf-8")
    for probe_id in (
        "clean",
        "positive_early_flow_channel_0",
        "negative_early_flow_channel_0",
        "positive_late_flow_channel_0",
        "negative_late_flow_channel_0",
    ):
        assert f"`{probe_id}`" in text
    assert "共享 clean base-velocity trace" in text
    assert "禁止用 \\(z_t^p\\) 再调用模型" in text
    assert "construction-only" in text
    assert "既有 `colab_test` 白名单接入" in text
    assert "固定 Notebook" in text
    assert "formal_result=false" in text
    assert "stage_progression_allowed=false" in text
    assert "feedback_nonlinearity_primary_candidate" not in text
    assert "decoder_carrier_mismatch_primary_candidate" not in text
    assert "`feedback_isolation_candidate`" in text
    assert "`decoder_carrier_mismatch_candidate`" in text
    assert "`multiple_candidates`" in text
