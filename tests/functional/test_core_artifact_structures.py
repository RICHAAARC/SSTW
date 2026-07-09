"""验证 records 和 manifests 的最小结构。"""

from __future__ import annotations

import csv

import pytest

from main.analysis.artifact_manifest import build_artifact_manifest
from main.core.records import ExperimentRecord, validate_record
from main.protocol.table_builder import write_csv


@pytest.mark.quick
def test_experiment_record_contains_required_fields() -> None:
    """实验 record 必须包含产物重建所需的最小字段。"""
    record = ExperimentRecord(
        record_id="record_example",
        run_id="run_example",
        split="validation",
        method_name="example_method",
        metric_name="accuracy",
        metric_value=0.9,
        metadata={},
    )
    assert validate_record(record.to_dict()) == []


@pytest.mark.quick
def test_artifact_manifest_records_rebuild_provenance() -> None:
    """产物 manifest 必须记录输入、输出、配置摘要和重建命令。"""
    manifest = build_artifact_manifest(
        artifact_id="table_example",
        artifact_type="table",
        input_paths=("outputs/records/example.jsonl",),
        output_paths=("outputs/tables/example.csv",),
        config={"metric_name": "accuracy"},
        code_version="uncommitted_template",
        rebuild_command="python scripts/rebuild_example_artifacts.py",
    )
    manifest_dict = manifest.to_dict()
    assert manifest_dict["artifact_id"] == "table_example"
    assert manifest_dict["config_digest"]
    assert manifest_dict["rebuild_command"]


@pytest.mark.quick
def test_write_csv_preserves_later_governed_record_fields(tmp_path) -> None:
    """CSV 写出器必须兼容同一 governed table 中后续行的补充字段。"""

    output_path = tmp_path / "tables" / "low_fpr_curve_table.csv"
    write_csv(
        output_path,
        [
            {
                "method_id": "sstw",
                "curve_point_fpr_level": 0.1,
                "tpr_at_target_fpr": 1.0,
            },
            {
                "method_id": "all_methods",
                "curve_point_fpr_level": 0.01,
                "curve_point_scope_note": "需要切换到对应 workflow profile 后生成。",
                "covered_method_count_at_current_target_fpr": 5,
            },
        ],
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == [
        "method_id",
        "curve_point_fpr_level",
        "tpr_at_target_fpr",
        "curve_point_scope_note",
        "covered_method_count_at_current_target_fpr",
    ]
    assert rows[0]["curve_point_scope_note"] == ""
    assert rows[1]["covered_method_count_at_current_target_fpr"] == "5"
