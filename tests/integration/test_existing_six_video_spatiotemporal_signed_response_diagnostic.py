from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.generative_video_model_probe.existing_six_video_spatiotemporal_signed_response_diagnostic import (
    run_existing_six_video_spatiotemporal_signed_response_diagnostic,
)


REAL_SOURCE_ZIP = Path("/tmp/sstw_gate_a_root_cause_f06a0934.zip")


@pytest.mark.integration
def test_real_six_video_zip_extract_decode_and_classify(
    tmp_path: Path,
) -> None:
    if not REAL_SOURCE_ZIP.is_file():
        pytest.skip("real f06a0934 six-video result ZIP is not mounted")
    output = tmp_path / "result"
    decision = (
        run_existing_six_video_spatiotemporal_signed_response_diagnostic(
            REAL_SOURCE_ZIP,
            output,
        )
    )
    assert decision[
        "spatiotemporal_signed_response_diagnostic_decision"
    ] == "existing_six_video_candidate_classification_recorded"
    assert decision["spatiotemporal_record_count"] == 142
    assert decision["source_gate_a_pass_preserved"] is False
    assert decision["formal_result"] is False
    assert decision["stage_progression_allowed"] is False
    classification = decision["candidate_classification"]
    assert classification["temporal_feature_salvage_candidate"] is False
    assert classification["carrier_redesign_required_candidate"] is True
    assert all(
        summary["signed_gate_pass_count"] == 0
        for summary in classification["family_gate_summaries"].values()
    )
    manifest = json.loads(
        (
            output
            / "artifacts"
            / "spatiotemporal_signed_response_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["cpu_only"] is True
    assert manifest["source_read_only"] is True
