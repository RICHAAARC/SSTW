"""冻结 probe、pilot 与 full 共用的论文机制配置。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping


PAPER_PROFILE_NAMES = {"probe_paper", "pilot_paper", "full_paper"}
COMMON_CONTRACT_PATH_FIELD = "paper_profile_common_contract_path"
COMMON_CONTRACT_DIGEST_FIELD = "paper_profile_common_contract_sha256"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CANONICAL_COMMON_CONTRACT_RELATIVE_PATH = Path(
    "configs/protocol/paper_profile_common_contract.json"
)
CANONICAL_COMMON_CONTRACT_PATH = (
    REPOSITORY_ROOT / CANONICAL_COMMON_CONTRACT_RELATIVE_PATH
).resolve()
CANONICAL_COMMON_CONTRACT_ID = "sstw_paper_profile_common_contract_v1"

# 该摘要由仓库内 canonical common contract 的原始字节计算。修改公共机制契约时,
# 必须在代码审阅中同时更新本常量与三个 profile 的声明摘要。把期望摘要冻结在代码
# 而不是另一个可替换 JSON 中, 可以阻止自定义 contract 自行声明并信任自己的摘要。
CANONICAL_COMMON_CONTRACT_SHA256 = (
    "f688d35086c8fcbb160a1a05d846fe30278b6e8859626d4e00a8d921413cf88d"
)

PAPER_PROFILE_CANONICAL_RELATIVE_PATHS: dict[str, Path] = {
    "probe_paper": Path("configs/protocol/probe_paper_generative_probe.json"),
    "pilot_paper": Path("configs/protocol/pilot_paper_generative_probe.json"),
    "full_paper": Path("configs/protocol/full_paper_generative_probe.json"),
}
PAPER_PROFILE_CANONICAL_PATHS = {
    profile_name: (REPOSITORY_ROOT / relative_path).resolve()
    for profile_name, relative_path in PAPER_PROFILE_CANONICAL_RELATIVE_PATHS.items()
}

# 非 canonical profile 只允许在 pytest 的显式上下文中用于轻量单元测试。正式 CLI、
# workflow 和普通 Python 调用无法隐式进入该上下文, 因而默认始终 fail-closed。
_NONCANONICAL_TEST_PROFILE_ALLOWED: ContextVar[bool] = ContextVar(
    "sstw_noncanonical_test_profile_allowed",
    default=False,
)


@contextmanager
def allow_noncanonical_paper_profile_for_tests() -> Iterator[None]:
    """仅在 pytest 中临时允许构造局部门禁 fixture。

    该入口属于测试基础设施, 不能用于正式结果生产。测试 override 不会把自定义
    contract 标记为 canonical, 也不会生成可支持论文主张的可信摘要状态。
    """

    if not os.environ.get("PYTEST_CURRENT_TEST"):
        raise RuntimeError("非 canonical paper profile override 只允许在 pytest 中使用")
    token = _NONCANONICAL_TEST_PROFILE_ALLOWED.set(True)
    try:
        yield
    finally:
        _NONCANONICAL_TEST_PROFILE_ALLOWED.reset(token)

# 正式 profile 文件只允许在以下字段上表达层级差异。这里使用显式字段集合而不是
# `minimum_*` 一类宽泛前缀, 主要考虑在于避免未来把新的机制阈值或证据开关伪装成
# “统计规模”后绕过公共契约。新增差异字段时必须先在此处完成分类和代码审阅。
PAPER_PROFILE_ONLY_FIELD_CATEGORIES: dict[str, frozenset[str]] = {
    "profile_identity": frozenset({
        "paper_result_level",
        "paper_profile_names",
        "stage_id",
    }),
    "target_fpr_and_statistical_scale": frozenset({
        "target_fpr",
        "blocked_target_fpr",
        "minimum_prompt_count",
        "minimum_seed_per_prompt",
        "minimum_calibration_seed_per_prompt",
        "minimum_test_seed_per_prompt",
        "minimum_unique_video_count",
        "minimum_calibration_unique_video_count",
        "minimum_test_unique_video_count",
        "minimum_clean_negative_count",
        "minimum_calibration_negative_event_count",
        "minimum_heldout_test_negative_event_count",
        "minimum_heldout_attacked_positive_event_count",
        "minimum_calibration_negative_event_count_per_family",
        "minimum_heldout_negative_event_count_per_family",
        "minimum_negative_event_count_per_family",
        "minimum_attack_event_count_per_attack",
        "minimum_independent_negative_video_count_for_fpr_upper_bound",
        "minimum_external_baseline_trace_count",
        "minimum_internal_ablation_trace_count",
        "minimum_heldout_posterior_positive_cluster_count",
        "minimum_heldout_posterior_negative_cluster_count",
        "minimum_heldout_posterior_attack_cluster_count",
        "minimum_adaptive_attack_source_video_cluster_count_per_protocol",
    }),
    "outer_stage_prerequisite": frozenset({
        "require_probe_paper_gate_passed",
        "require_probe_paper_to_pilot_paper_transition_decision",
        "require_pilot_paper_gate_passed",
        "require_pilot_paper_to_full_paper_transition_decision",
        "require_full_paper_to_submission_freeze_transition_decision",
        "require_full_paper_to_submission_freeze_transition_decision_scope",
        "submission_freeze_allowed_after_full_paper_checker_pass",
    }),
    "profile_documentation": frozenset({
        "paper_protocol_difference_from_probe_paper",
        "paper_protocol_difference_from_pilot_paper",
        "paper_protocol_difference_from_full_paper",
        "required_runtime_attack_protocol_note",
    }),
}

PAPER_PROFILE_ONLY_FIELDS = frozenset().union(
    *PAPER_PROFILE_ONLY_FIELD_CATEGORIES.values()
)

PAPER_PROFILE_INVARIANT_METADATA_FIELDS = frozenset({
    COMMON_CONTRACT_PATH_FIELD,
    COMMON_CONTRACT_DIGEST_FIELD,
})

PAPER_PROFILE_EXPECTED_IDENTITY: dict[str, dict[str, Any]] = {
    "probe_paper": {
        "paper_profile_names": ["probe_paper"],
        "stage_id": "probe_paper_generative_probe_gate",
    },
    "pilot_paper": {
        "paper_profile_names": ["pilot_paper"],
        "stage_id": "pilot_paper_generative_probe_gate",
    },
    "full_paper": {
        "paper_profile_names": ["full_paper"],
        "stage_id": "full_paper_generative_probe_gate",
    },
}

PAPER_PROFILE_EXPECTED_TARGET_FPR = {
    "probe_paper": 0.1,
    "pilot_paper": 0.01,
    "full_paper": 0.001,
}

PAPER_PROFILE_EXPECTED_BLOCKED_TARGET_FPR = {
    "probe_paper": 0.01,
    "pilot_paper": 0.001,
    "full_paper": None,
}

PAPER_PROFILE_REQUIRED_SCALE_FIELDS = (
    PAPER_PROFILE_ONLY_FIELD_CATEGORIES["target_fpr_and_statistical_scale"]
    - {"blocked_target_fpr"}
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"paper profile JSON 顶层必须是对象: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    """计算文件原始字节摘要, 避免 JSON 重排后仍被误认为同一冻结契约。"""

    return sha256(path.read_bytes()).hexdigest()


def _resolve_repository_path(raw_path: str | Path) -> Path:
    """把仓库相对路径解析到当前安装或抽离包的 repository root。"""

    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (REPOSITORY_ROOT / path).resolve()


def _canonical_contract() -> dict[str, Any]:
    """读取并验证仓库唯一可信的 paper profile 公共契约。"""

    if not CANONICAL_COMMON_CONTRACT_PATH.is_file():
        raise FileNotFoundError(
            f"缺少 canonical paper profile 公共契约: {CANONICAL_COMMON_CONTRACT_PATH}"
        )
    observed_digest = _sha256_file(CANONICAL_COMMON_CONTRACT_PATH)
    if observed_digest != CANONICAL_COMMON_CONTRACT_SHA256:
        raise ValueError(
            "canonical paper profile 公共契约内容摘要不匹配: "
            f"expected={CANONICAL_COMMON_CONTRACT_SHA256}, observed={observed_digest}"
        )
    contract = _read_json(CANONICAL_COMMON_CONTRACT_PATH)
    if contract.get("paper_profile_common_contract_id") != CANONICAL_COMMON_CONTRACT_ID:
        raise ValueError(
            "canonical paper profile 公共契约 ID 不匹配: "
            f"expected={CANONICAL_COMMON_CONTRACT_ID}, "
            f"observed={contract.get('paper_profile_common_contract_id')}"
        )
    return contract


def _enforce_noncanonical_test_profile(
    profile: Mapping[str, Any],
    profile_path: str | Path,
) -> dict[str, Any]:
    """为 pytest 局部门禁 fixture 保留非正式、不可支持 claim 的配置读取能力。

    该路径只会由显式测试上下文启用。测试配置可以包含被测模块的局部旧字段,
    但不能改变正式层级冻结的目标 FPR, 且返回值会被标记为不可支持论文主张。
    正式 CLI 不具备进入此分支的能力, 因而不会削弱 canonical 配置的闭合校验。
    """

    merged = dict(profile)
    level = str(merged.get("paper_result_level") or "")
    expected_target_fpr = PAPER_PROFILE_EXPECTED_TARGET_FPR[level]
    if "target_fpr" in merged and float(merged["target_fpr"]) != expected_target_fpr:
        raise ValueError(
            f"{level} target_fpr 必须为 {expected_target_fpr}, "
            f"实际为 {merged['target_fpr']}"
        )
    merged["paper_profile_common_contract_status"] = (
        "pytest_noncanonical_override_not_claim_supporting"
    )
    merged["paper_profile_common_contract_resolved_path"] = str(profile_path)
    return merged


def enforce_paper_profile_common_contract(
    profile: Mapping[str, Any],
    profile_path: str | Path,
) -> dict[str, Any]:
    """验证 profile 未漂移并返回合并后的只读语义副本。

    profile 文件仍显式保存公共字段, 便于独立审阅。加载时逐字段与公共契约比对,
    任一差异立即失败。公共契约之外只接受本模块显式分类的 profile 身份、目标
    FPR、样本统计规模、外层阶段前置与说明字段; 任意未登记键立即失败。这样可以
    从机制上阻断 pilot 残留旧语义或用新证据开关制造单文件静默漂移。
    """

    merged = dict(profile)
    level = str(merged.get("paper_result_level") or "")
    if level not in PAPER_PROFILE_NAMES:
        return merged
    if _NONCANONICAL_TEST_PROFILE_ALLOWED.get():
        return _enforce_noncanonical_test_profile(merged, profile_path)

    resolved_profile_path = _resolve_repository_path(profile_path)
    expected_profile_path = PAPER_PROFILE_CANONICAL_PATHS[level]
    if resolved_profile_path != expected_profile_path:
        raise ValueError(
            f"{level} 正式运行只接受仓库 canonical profile: "
            f"expected={expected_profile_path}, observed={resolved_profile_path}"
        )

    raw_contract_path = merged.get(COMMON_CONTRACT_PATH_FIELD)
    if not raw_contract_path:
        raise KeyError(f"{level} 缺少 {COMMON_CONTRACT_PATH_FIELD}")
    contract_path = _resolve_repository_path(str(raw_contract_path))
    if contract_path != CANONICAL_COMMON_CONTRACT_PATH:
        raise ValueError(
            "paper profile 只能引用仓库 canonical 公共契约: "
            f"expected={CANONICAL_COMMON_CONTRACT_PATH}, observed={contract_path}"
        )
    declared_digest = str(merged.get(COMMON_CONTRACT_DIGEST_FIELD) or "")
    if declared_digest != CANONICAL_COMMON_CONTRACT_SHA256:
        raise ValueError(
            f"{level} {COMMON_CONTRACT_DIGEST_FIELD} 不匹配: "
            f"expected={CANONICAL_COMMON_CONTRACT_SHA256}, observed={declared_digest}"
        )
    contract = _canonical_contract()
    unknown_fields = sorted(
        set(merged)
        - set(contract)
        - set(PAPER_PROFILE_ONLY_FIELDS)
        - set(PAPER_PROFILE_INVARIANT_METADATA_FIELDS)
    )
    if unknown_fields:
        raise ValueError(
            "paper profile 包含未登记的层级差异字段: "
            + json.dumps(unknown_fields, ensure_ascii=False)
        )
    drift = {
        key: {"expected": expected, "observed": merged.get(key)}
        for key, expected in contract.items()
        if merged.get(key) != expected
    }
    if drift:
        raise ValueError(
            "paper profile 公共机制配置漂移: "
            + json.dumps(drift, ensure_ascii=False, sort_keys=True)
        )
    identity_drift = {
        key: {"expected": expected, "observed": merged.get(key)}
        for key, expected in PAPER_PROFILE_EXPECTED_IDENTITY[level].items()
        if merged.get(key) != expected
    }
    if identity_drift:
        raise ValueError(
            "paper profile 身份字段漂移: "
            + json.dumps(identity_drift, ensure_ascii=False, sort_keys=True)
        )
    missing_scale_fields = sorted(
        PAPER_PROFILE_REQUIRED_SCALE_FIELDS - set(merged)
    )
    if missing_scale_fields:
        raise ValueError(
            "paper profile 缺少必填统计规模字段: "
            + json.dumps(missing_scale_fields, ensure_ascii=False)
        )
    expected_target_fpr = PAPER_PROFILE_EXPECTED_TARGET_FPR[level]
    if float(merged["target_fpr"]) != expected_target_fpr:
        raise ValueError(
            f"{level} target_fpr 必须为 {expected_target_fpr}, "
            f"实际为 {merged['target_fpr']}"
        )
    expected_blocked_fpr = PAPER_PROFILE_EXPECTED_BLOCKED_TARGET_FPR[level]
    observed_blocked_fpr = merged.get("blocked_target_fpr")
    if (
        expected_blocked_fpr is None
        and observed_blocked_fpr is not None
    ) or (
        expected_blocked_fpr is not None
        and (
            observed_blocked_fpr is None
            or float(observed_blocked_fpr) != expected_blocked_fpr
        )
    ):
        raise ValueError(
            f"{level} blocked_target_fpr 必须为 {expected_blocked_fpr}, "
            f"实际为 {observed_blocked_fpr}"
        )
    invalid_difference_fields = sorted(
        field_name
        for field_name in (
            "paper_protocol_difference_from_probe_paper",
            "paper_protocol_difference_from_pilot_paper",
            "paper_protocol_difference_from_full_paper",
        )
        if field_name in merged
        and merged[field_name] != "sample_scale_and_target_fpr_only"
    )
    if invalid_difference_fields:
        raise ValueError(
            "paper profile 层级差异声明只能是样本规模与目标 FPR: "
            + json.dumps(invalid_difference_fields, ensure_ascii=False)
        )
    invalid_scale_fields = sorted(
        field_name
        for field_name in PAPER_PROFILE_REQUIRED_SCALE_FIELDS
        if field_name != "target_fpr"
        and (
            isinstance(merged.get(field_name), bool)
            or not isinstance(merged.get(field_name), int)
            or int(merged[field_name]) <= 0
        )
    )
    if invalid_scale_fields:
        raise ValueError(
            "paper profile 统计规模字段必须为正整数: "
            + json.dumps(invalid_scale_fields, ensure_ascii=False)
        )
    expected_heldout_cluster_count = int(merged["minimum_test_unique_video_count"])
    heldout_cluster_drift = {
        field_name: {
            "expected": expected_heldout_cluster_count,
            "observed": merged.get(field_name),
        }
        for field_name in (
            "minimum_heldout_posterior_positive_cluster_count",
            "minimum_heldout_posterior_negative_cluster_count",
        )
        if int(merged[field_name]) != expected_heldout_cluster_count
    }
    attack_cluster_count = int(merged["minimum_heldout_posterior_attack_cluster_count"])
    attack_event_count = int(merged["minimum_attack_event_count_per_attack"])
    if attack_cluster_count != attack_event_count:
        heldout_cluster_drift["minimum_heldout_posterior_attack_cluster_count"] = {
            "expected": attack_event_count,
            "observed": attack_cluster_count,
        }
    elif attack_cluster_count > expected_heldout_cluster_count:
        heldout_cluster_drift["minimum_heldout_posterior_attack_cluster_count"] = {
            "expected_maximum": expected_heldout_cluster_count,
            "observed": attack_cluster_count,
        }
    if heldout_cluster_drift:
        raise ValueError(
            "held-out posterior 簇数必须由 held-out 视频容量与逐攻击统计规模派生: "
            + json.dumps(heldout_cluster_drift, ensure_ascii=False, sort_keys=True)
        )
    merged.update(contract)
    merged["paper_profile_common_contract_resolved_path"] = str(contract_path)
    merged["paper_profile_common_contract_observed_sha256"] = (
        CANONICAL_COMMON_CONTRACT_SHA256
    )
    merged["paper_profile_common_contract_status"] = "matched"
    return merged
