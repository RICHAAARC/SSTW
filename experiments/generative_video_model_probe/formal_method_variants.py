"""把论文 baseline 与内部消融组合映射到参数化 SSTW 核心 API。

该模块属于项目特定的实验设计层。`main/` 只实现速度场约束、状态观测、
状态空间后验和固定 FPR 检测等通用原语；本模块才负责解释论文中的
`method_variant` 名称，并把每个名称转换为明确的核心机制参数或观测变换。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from main.methods.state_space_watermark.flow_velocity_runtime import (
    FlowVelocityRuntimeMechanismConfig,
)
from main.methods.state_space_watermark.formal_detector import (
    FrozenFlowDetectorCalibration,
    FlowDetectorMechanismConfig,
    apply_frozen_flow_detector as apply_core_frozen_flow_detector,
    fit_flow_detector_calibration as fit_core_flow_detector_calibration,
    flow_evidence_observation_sequence_from_mappings,
    frozen_flow_detector_calibration_from_dict,
)


FORMAL_METHOD_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "trajectory_only_score",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
    "without_replay_uncertainty_weighting",
    "without_flow_state_admissibility",
    "generic_ssm_baseline",
)

# 只有以下变体改变 scheduler 嵌入轨迹, 因而必须真实重新生成视频。
GENERATION_METHOD_VARIANTS = (
    "sstw_full_method",
    "endpoint_only_control",
    "without_velocity_constraint",
    "without_endpoint_aware_control",
)

# 以下变体只改变同一视频、同一 replay 观测的检测方式。重复生成视频既浪费
# GPU, 又会把生成随机性混入消融效应, 因此必须复用 full-method 证据。
DETECTOR_ONLY_METHOD_VARIANTS = (
    "trajectory_only_score",
    "without_replay_uncertainty_weighting",
    "without_flow_state_admissibility",
    "generic_ssm_baseline",
)

if set(GENERATION_METHOD_VARIANTS) & set(DETECTOR_ONLY_METHOD_VARIANTS):
    raise RuntimeError("生成级变体与检测器级变体不得重叠")
if set(GENERATION_METHOD_VARIANTS) | set(DETECTOR_ONLY_METHOD_VARIANTS) != set(
    FORMAL_METHOD_VARIANTS
):
    raise RuntimeError("正式方法变体必须完整且唯一地归入生成级或检测器级")

# Claim-2 的检测器消融复用 full-method 视频和 calibration records, 不重新生成视频。
# 这样比较只移除路径观测, 不会同时改变速度约束、replay 或状态空间后验。
CLAIM2_PATH_NESTED_ABLATION_VARIANT = "without_path_evidence"
FORMAL_DETECTOR_VARIANTS = (
    *FORMAL_METHOD_VARIANTS,
    CLAIM2_PATH_NESTED_ABLATION_VARIANT,
)


@dataclass(frozen=True)
class FormalMethodVariantDefinition:
    """描述一个论文实验变体如何组合核心机制。"""

    method_variant: str
    velocity_runtime: FlowVelocityRuntimeMechanismConfig
    detector_mechanism: FlowDetectorMechanismConfig = FlowDetectorMechanismConfig()
    observation_transform: str = "identity"


_FULL_VELOCITY = FlowVelocityRuntimeMechanismConfig()
_VARIANT_DEFINITIONS = {
    "sstw_full_method": FormalMethodVariantDefinition(
        method_variant="sstw_full_method",
        velocity_runtime=_FULL_VELOCITY,
    ),
    "endpoint_only_control": FormalMethodVariantDefinition(
        method_variant="endpoint_only_control",
        velocity_runtime=FlowVelocityRuntimeMechanismConfig(
            velocity_constraint_enabled=False,
            endpoint_control_enabled=False,
            terminal_endpoint_perturbation_enabled=True,
        ),
        observation_transform="endpoint_only",
    ),
    "trajectory_only_score": FormalMethodVariantDefinition(
        method_variant="trajectory_only_score",
        velocity_runtime=_FULL_VELOCITY,
        observation_transform="trajectory_only",
    ),
    "without_path_evidence": FormalMethodVariantDefinition(
        method_variant="without_path_evidence",
        velocity_runtime=_FULL_VELOCITY,
        observation_transform="without_path_evidence",
    ),
    "without_velocity_constraint": FormalMethodVariantDefinition(
        method_variant="without_velocity_constraint",
        velocity_runtime=FlowVelocityRuntimeMechanismConfig(
            velocity_constraint_enabled=False,
        ),
    ),
    "without_endpoint_aware_control": FormalMethodVariantDefinition(
        method_variant="without_endpoint_aware_control",
        velocity_runtime=FlowVelocityRuntimeMechanismConfig(
            endpoint_control_enabled=False,
        ),
    ),
    "without_replay_uncertainty_weighting": FormalMethodVariantDefinition(
        method_variant="without_replay_uncertainty_weighting",
        velocity_runtime=_FULL_VELOCITY,
        observation_transform="unit_replay_reliability",
    ),
    "without_flow_state_admissibility": FormalMethodVariantDefinition(
        method_variant="without_flow_state_admissibility",
        velocity_runtime=_FULL_VELOCITY,
        detector_mechanism=FlowDetectorMechanismConfig(
            enforce_state_admissibility=False,
        ),
    ),
    "generic_ssm_baseline": FormalMethodVariantDefinition(
        method_variant="generic_ssm_baseline",
        velocity_runtime=_FULL_VELOCITY,
        observation_transform="key_agnostic_state_observation",
    ),
}


def formal_method_variant_definition(
    method_variant: str,
) -> FormalMethodVariantDefinition:
    """读取实验变体定义，并对未知名称执行 fail-closed。"""

    try:
        return _VARIANT_DEFINITIONS[str(method_variant)]
    except KeyError as exc:
        raise ValueError(f"未注册的正式 method variant: {method_variant}") from exc


def velocity_runtime_mechanism_for_method_variant(
    method_variant: str,
) -> FlowVelocityRuntimeMechanismConfig:
    """把生成实验名称转换为核心 scheduler 机制参数。

    clean reference 是实验层的无水印样本，不属于正式检测变体，因此在此处
    单独映射为关闭全部水印注入的核心运行配置。
    """

    if method_variant == "sstw_clean_unwatermarked_reference":
        return FlowVelocityRuntimeMechanismConfig(
            velocity_constraint_enabled=False,
            endpoint_control_enabled=False,
        )
    if method_variant == "key_conditioned_state_space_with_trajectory":
        return _FULL_VELOCITY
    return formal_method_variant_definition(method_variant).velocity_runtime


def _transform_phase_observation(
    phase: Mapping[str, Any],
    *,
    transform: str,
) -> dict[str, Any]:
    """对单个真实 phase 观测执行预注册的实验变换。"""

    result = dict(phase)
    if transform == "identity":
        return result
    if transform == "endpoint_only":
        result.update(
            velocity_score=0.0,
            S_velocity=0.0,
            path_score=0.0,
            S_path_inv=0.0,
            path_endpoint_consistency=0.0,
            replay_log_likelihood_ratio=0.0,
            replay_log_likelihood_ratio_mean=0.0,
            replay_reliability=1.0,
            replay_reliability_weight=1.0,
            time_grid_reliability=1.0,
        )
        return result
    if transform == "trajectory_only":
        result["endpoint_score"] = 0.0
        return result
    if transform == "without_path_evidence":
        result.update(
            path_score=0.0,
            S_path_inv=0.0,
            path_endpoint_consistency=0.0,
        )
        return result
    if transform == "unit_replay_reliability":
        # 正式 replay 序列同时保存未加权路径投影, 因而该消融可以真正移除
        # 不确定性权重, 而不是只把一个旁路 reliability 特征改成常数。
        if result.get("path_score_unweighted") is not None:
            result["path_score"] = float(result["path_score_unweighted"])
            result["S_path_inv"] = float(result["path_score_unweighted"])
        if result.get("path_endpoint_consistency_unweighted") is not None:
            result["path_endpoint_consistency"] = float(
                result["path_endpoint_consistency_unweighted"]
            )
        result.update(
            replay_reliability=1.0,
            replay_reliability_weight=1.0,
            replay_step_reliability_weight=1.0,
            time_grid_reliability=1.0,
        )
        return result
    if transform == "key_agnostic_state_observation":
        result.update(
            endpoint_score=float(result.get("key_agnostic_endpoint_energy") or 0.0),
            velocity_score=float(result.get("key_agnostic_velocity_energy") or 0.0),
            S_velocity=float(result.get("key_agnostic_velocity_energy") or 0.0),
            path_score=float(result.get("key_agnostic_path_energy") or 0.0),
            S_path_inv=float(result.get("key_agnostic_path_energy") or 0.0),
            path_endpoint_consistency=float(
                result.get("path_velocity_consistency")
                or result.get("path_endpoint_consistency")
                or 0.0
            ),
            replay_log_likelihood_ratio=0.0,
            replay_log_likelihood_ratio_mean=0.0,
        )
        return result
    raise ValueError(f"未知观测变换: {transform}")


def transform_flow_evidence_record_for_method_variant(
    record: Mapping[str, Any],
    *,
    method_variant: str,
) -> dict[str, Any]:
    """生成实验专用记录副本，原始 governed record 不会被原地修改。"""

    definition = formal_method_variant_definition(method_variant)
    transformed = deepcopy(dict(record))
    sequence = transformed.get("flow_state_observation_sequence")
    if not isinstance(sequence, list) or len(sequence) < 2:
        raise ValueError("正式 method variant 必须消费至少2个真实 phase 观测")
    transformed["flow_state_observation_sequence"] = [
        _transform_phase_observation(row, transform=definition.observation_transform)
        for row in sequence
    ]
    return transformed


def observation_sequence_for_method_variant(
    record: Mapping[str, Any],
    *,
    method_variant: str,
) -> list[Any]:
    """返回实验变换后的核心状态空间观测，供验证变换语义复用。"""

    transformed = transform_flow_evidence_record_for_method_variant(
        record,
        method_variant=method_variant,
    )
    raw_sequence = transformed.get("flow_state_observation_sequence")
    if not isinstance(raw_sequence, list):
        raise TypeError("governed flow evidence record 的 phase 序列必须为列表")
    return flow_evidence_observation_sequence_from_mappings(raw_sequence)


def fit_flow_evidence_calibration(
    calibration_records: Iterable[Mapping[str, Any]],
    *,
    method_variant: str,
    target_fpr: float,
) -> FrozenFlowDetectorCalibration:
    """验证 governed calibration records 并调用纯核心拟合 API。

    该函数属于项目特定的实验适配层。这里负责解释数据分区、样本角色和
    统计簇；`main/` 只接收已经提取好的观测序列、二元标签与簇标识。
    """

    definition = formal_method_variant_definition(method_variant)
    rows = [dict(record) for record in calibration_records]
    invalid_splits = sorted({
        str(row.get("split") or "missing")
        for row in rows
        if row.get("split") != "calibration"
    })
    if invalid_splits:
        raise ValueError(
            "概率后验与 fixed-FPR 阈值只能使用 calibration split, "
            f"收到: {invalid_splits}"
        )
    allowed_sample_roles = {
        "attacked_positive",
        "clean_negative",
        "controlled_negative",
    }
    invalid_sample_roles = sorted({
        str(row.get("sample_role") or "missing")
        for row in rows
        if row.get("sample_role") not in allowed_sample_roles
    })
    if invalid_sample_roles:
        raise ValueError(
            "概率后验 calibration 包含未知 sample_role: "
            f"{invalid_sample_roles}"
        )
    cluster_ids: list[str] = []
    for row in rows:
        cluster_id = row.get("statistical_cluster_id")
        if not cluster_id:
            raise KeyError("正式 calibration record 缺少 statistical_cluster_id")
        cluster_ids.append(str(cluster_id))
    return fit_core_flow_detector_calibration(
        [
            observation_sequence_for_method_variant(
                record,
                method_variant=method_variant,
            )
            for record in rows
        ],
        [
            1 if row.get("sample_role") == "attacked_positive" else 0
            for row in rows
        ],
        cluster_ids,
        target_fpr=target_fpr,
        mechanism_config=definition.detector_mechanism,
        detector_configuration_id=method_variant,
    )


def apply_frozen_flow_detector(
    record: Mapping[str, Any],
    calibration: FrozenFlowDetectorCalibration,
) -> dict[str, Any]:
    """适配 governed record, 并补充正式实验所需的来源字段。"""

    method_variant = calibration.detector_configuration_id
    core_decision = apply_core_frozen_flow_detector(
        observation_sequence_for_method_variant(
            record,
            method_variant=method_variant,
        ),
        calibration,
    )
    return {
        **core_decision,
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
        "metric_status": "measured_formal",
    }


def frozen_flow_detector_calibration_artifact(
    calibration: FrozenFlowDetectorCalibration,
) -> dict[str, Any]:
    """把纯核心 calibration 转写为受治理的正式阈值 artifact。"""

    return {
        **calibration.as_dict(),
        "threshold_source_split": "calibration",
        "test_time_threshold_update_blocked": True,
    }


def frozen_flow_detector_calibration_from_governed_artifact(
    payload: Mapping[str, Any],
) -> FrozenFlowDetectorCalibration:
    """验证正式阈值来源后, 重建只读核心检测器。"""

    if payload.get("threshold_source_split") != "calibration":
        raise ValueError("正式阈值 artifact 必须来自 calibration split")
    if payload.get("test_time_threshold_update_blocked") is not True:
        raise ValueError("正式阈值 artifact 必须禁止测试时更新")
    return frozen_flow_detector_calibration_from_dict(payload)
