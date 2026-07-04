# Field Registry

## 文档定位

本文档是项目中 governed fields 的登记表。它只记录“当前项目实际使用或模板预留的字段实例”, 不重复解释字段治理规则。

字段 category、后缀要求和清理规则见:

```text
docs/placeholder_random_governance.md
docs/intermediate_state_governance.md
docs/artifact_rebuild.md
```

## 何时需要登记

新增字段只要进入下列任一位置, 就应先登记到本表:

```text
配置文件
records
manifests
tables
reports
Python dict key
测试 fixture
Markdown 示例
Notebook 与 repository module 的跨边界数据
```

函数内部一次性局部变量不需要登记。跨函数、跨文件、跨进程或跨 Notebook 边界保存的中间状态字段需要登记。

## 字段登记表

| field_name | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- |
| project_stage | governance | none | true | false | false | 当前项目语义阶段。 |
| target_construction_phase | governance | none | true | false | false | 当前构建目标。 |
| run_id | protocol | none | true | false | false | 一次运行的稳定标识。 |
| record_id | protocol | none | true | false | false | 单条记录的稳定标识。 |
| split | protocol | none | true | false | false | 数据或事件划分。 |
| method_name | protocol | none | true | false | false | 实验记录中的方法名称。 |
| metric_name | protocol | none | true | false | false | 实验记录中的指标名称。 |
| metric_value | protocol | none | true | false | false | 实验记录中的指标数值。 |
| artifact_id | artifact | none | false | false | false | 受治理论文产物的稳定标识。 |
| artifact_type | artifact | none | false | false | false | 受治理论文产物类型, 例如 table、figure、report 或 manifest。 |
| input_paths | artifact | none | false | false | false | 产物重建所需输入路径。 |
| output_paths | artifact | none | false | false | false | 产物重建生成输出路径。 |
| config_digest | artifact | none | false | false | false | 产物重建配置摘要。 |
| code_version | artifact | none | false | false | false | 产物重建所用代码版本。 |
| rebuild_command | artifact | none | false | false | false | 产物重建命令。 |
| claim_id | claim | none | false | true | false | claim 审计表中的声明标识。 |
| evidence_path | claim | none | false | true | false | claim 绑定的证据路径。 |
| backend_placeholder | placeholder | _placeholder | true | false | true | Bootstrap 阶段的占位 backend 字段。 |
| example_digest_random | random | _digest_random | true | false | false | 可复现随机轨迹的 digest 字段。 |
| example_state_intermediate | intermediate | _intermediate | true | false | true | 跨步骤保存的示例中间状态字段, 正式产物生成前需要清理或迁移。 |
| example_artifact_temporary | temporary | _temporary | false | false | true | 可清理的示例临时产物标记。 |
| example_result_cache | cache | _cache | false | false | false | 可由输入、配置和代码重建的示例缓存标记。 |
| record_version | protocol | none | true | false | false | B1 event record schema version. |
| sample_id | protocol | none | true | false | false | Synthetic sample identifier. |
| sample_role | protocol | none | true | false | false | Sample role. |
| method_variant | protocol | none | true | false | false | Controlled method or baseline variant. |
| attack_name | protocol | none | true | false | false | Synthetic attack name. |
| attack_strength | protocol | none | true | false | false | Synthetic attack strength parameter. |
| key_id | protocol | none | true | false | false | Watermark key identifier. |
| content_id | protocol | none | true | false | false | Synthetic content identifier. |
| prompt_id_placeholder | placeholder | _placeholder | true | false | true | B1 placeholder for later prompt identifier. |
| seed_id | protocol | none | true | false | false | Deterministic synthetic seed identifier. |
| generation_model_id_placeholder | placeholder | _placeholder | true | false | true | B1 placeholder for later generation model identifier. |
| backend_id | protocol | none | true | false | false | Runtime backend identifier. |
| tubelet_length | method | none | true | false | false | Tubelet temporal length. |
| tubelet_spatial_patch | method | none | true | false | false | Tubelet spatial patch size. |
| tubelet_stride_t | method | none | true | false | false | Tubelet temporal stride. |
| tubelet_stride_xy | method | none | true | false | false | Tubelet spatial stride. |
| watermark_alpha | method | none | true | false | false | Projection margin used by B1 synthetic embedding proxy. |
| payload_code_id | method | none | true | false | false | Payload code configuration identifier. |
| sync_code_id | method | none | true | false | false | Synchronization code configuration identifier. |
| joint_code_mode | method | none | true | false | false | Joint payload and synchronization code mode. |
| embedding_mode | method | none | true | false | false | Embedding mode identifier. |
| state_model_id | method | none | true | false | false | State model identifier. |
| state_dim | method | none | true | false | false | State vector dimension. |
| key_condition_mode | method | none | true | false | false | How key conditioning is injected. |
| filter_mode | method | none | true | false | false | State filtering mode. |
| smoother_enabled | method | none | true | false | false | Whether smoother is enabled. |
| phase_state_proxy | method | none | true | false | false | B1 proxy for phase state. |
| evidence_state_proxy | method | none | true | false | false | B1 proxy for evidence state. |
| confidence_state_proxy | method | none | true | false | false | B1 proxy for confidence state. |
| disturbance_state_proxy | method | none | true | false | false | B1 proxy for disturbance state. |
| state_entropy | metric | none | true | false | false | State uncertainty score. |
| state_coverage_ratio | metric | none | true | false | false | State coverage ratio. |
| state_matched_count | metric | none | true | false | false | Number of matched state elements. |
| state_transition_residual | metric | none | true | false | false | State transition residual. |
| S_payload_raw | metric | none | true | false | false | Raw payload score. |
| S_payload_state | metric | none | true | false | false | State-aligned payload score. |
| S_state_posterior | metric | none | true | false | false | State posterior score. |
| S_trajectory_observation_placeholder | placeholder | _placeholder | true | false | true | B1 placeholder for trajectory observation score. |
| S_final | metric | none | true | false | false | Final detector statistic. |
| payload_state_gain | metric | none | true | false | false | Difference between state payload and raw payload score. |
| key_state_admissibility_status | metric | none | true | false | false | Admissibility gate status. |
| negative_state_over_threshold_count | metric | none | true | false | false | Negative state rescue count above threshold. |
| target_fpr | protocol | none | true | false | false | Fixed low-FPR target. |
| target_fpr_source_config_path | protocol | none | true | false | false | 当前 target_fpr 的来源 protocol config 路径。 |
| protocol_target_fpr | protocol | none | false | false | false | Notebook workflow profile 解析时从 protocol config 合并得到的 target_fpr。 |
| threshold_id | protocol | none | true | false | false | Threshold identifier. |
| threshold_source_split | protocol | none | true | false | false | Split used to calibrate threshold. |
| threshold_value | protocol | none | true | false | false | Calibrated threshold value. |
| decision | protocol | none | true | false | false | Detector decision. |
| decision_reason | protocol | none | true | false | false | Decision provenance reason. |
| test_time_threshold_update_blocked | protocol | none | true | false | false | Whether test-time threshold update is blocked. |
| trajectory_trace_placeholder | placeholder | _placeholder | true | false | true | B1 placeholder for trajectory trace. |
| real_video_quality_metrics_placeholder | placeholder | _placeholder | true | false | true | B1 placeholder for real-video quality metrics. |
| semantic_consistency_placeholder | placeholder | _placeholder | true | false | true | B1 placeholder for semantic consistency metric. |
| placeholder_reason | protocol | none | true | false | false | Reason explaining placeholder presence. |
| replacement_stage | protocol | none | true | false | false | Stage expected to replace placeholder. |
| replacement_field_name | protocol | none | true | false | false | Concrete field expected to replace placeholder. |
| source_video_id | protocol | none | true | false | false | B2 source video identifier. |
| dataset_id | protocol | none | true | false | false | B2 dataset identifier. |
| video_fps | protocol | none | true | false | false | Processed video fps. |
| video_num_frames | protocol | none | true | false | false | Processed video frame count. |
| video_resolution | protocol | none | true | false | false | Processed video resolution. |
| video_duration_sec | protocol | none | true | false | false | Processed video duration. |
| frame_sample_status | protocol | none | true | false | false | Frame sampling status. |
| fps_normalizer_status | protocol | none | true | false | false | FPS normalization status. |
| vae_chain_status | protocol | none | true | false | false | VAE encode-decode-reencode chain status. |
| vae_backend_id | protocol | none | true | false | false | VAE backend identifier. |
| vae_model_name | protocol | none | true | false | false | VAE model name. |
| vae_model_version | protocol | none | true | false | false | VAE model version. |
| vae_encode_dtype | protocol | none | true | false | false | VAE encode dtype. |
| vae_decode_dtype | protocol | none | true | false | false | VAE decode dtype. |
| vae_reconstruction_psnr | metric | none | true | false | false | VAE reconstruction PSNR proxy. |
| vae_reconstruction_ssim | metric | none | true | false | false | VAE reconstruction SSIM proxy. |
| vae_reconstruction_lpips_status | metric | none | true | false | false | VAE reconstruction LPIPS status. |
| vae_reconstruction_lpips_status_reason | protocol | none | true | false | false | Reason for VAE LPIPS status. |
| attack_config_id | protocol | none | true | false | false | Attack configuration identifier. |
| attack_seed | protocol | none | true | false | false | Attack seed identifier. |
| attack_runtime_sec | metric | none | true | false | false | Attack runtime in seconds. |
| runtime_attack_decision | governance | none | true | false | false | Runtime video-file attack runner decision. |
| runtime_attack_record_count | metric | none | true | false | false | Number of runtime attack records. |
| runtime_attack_ready_count | metric | none | true | false | false | Number of runtime attack records that produced attacked videos. |
| runtime_attack_count | metric | none | true | false | false | Number of distinct runtime attacks that produced attacked videos. |
| source_video_path | artifact | none | true | false | false | Source generated video path used by runtime attack runner. |
| source_video_sha256 | artifact | none | true | false | false | Source generated video sha256 used by runtime attack runner. |
| attacked_video_path | artifact | none | true | false | false | Attacked video path produced by runtime attack runner. |
| attacked_video_sha256 | artifact | none | true | false | false | Attacked video sha256 produced by runtime attack runner. |
| source_frame_count | metric | none | true | false | false | Number of decoded source frames used by runtime attack runner. |
| attacked_frame_count | metric | none | true | false | false | Number of frames written to attacked video. |
| attack_transform | protocol | none | true | false | false | Runtime attack transform description. |
| runtime_attack_expected_effect | protocol | none | true | false | false | Expected effect of a runtime attack transform. |
| attack_failure_status | protocol | none | true | false | false | Attack failure status. |
| attack_failure_reason | protocol | none | true | false | false | Attack failure reason. |
| quality_psnr | metric | none | true | false | false | B2 quality PSNR proxy. |
| quality_ssim | metric | none | true | false | false | B2 quality SSIM proxy. |
| quality_lpips | metric | none | true | false | false | B2 quality LPIPS value or null. |
| quality_metric_status | metric | none | true | false | false | Quality metric status. |
| quality_metric_failure_reason | protocol | none | true | false | false | Quality metric failure reason. |
| quality_not_collapsed | metric | none | true | false | false | Quality gate decision. |
| temporal_flicker_score | metric | none | true | false | false | Temporal flicker proxy score. |
| temporal_consistency_not_collapsed | metric | none | true | false | false | Temporal consistency gate decision. |
| motion_consistency_score_placeholder | placeholder | _placeholder | true | false | true | B2 placeholder for motion consistency score. |
| motion_consistency_status | protocol | none | true | false | false | Motion consistency metric status. |
| motion_consistency_reason | protocol | none | true | false | false | Motion consistency status reason. |
| formal_state_schema_version | protocol | none | true | false | false | B3 formal state schema version. |
| state_transition_model_id | method | none | true | false | false | B3 transition model identifier. |
| state_observation_model_id | method | none | true | false | false | B3 observation model identifier. |
| key_conditioner_id | method | none | true | false | false | B3 key conditioner identifier. |
| smoother_mode | method | none | true | false | false | B3 smoother mode. |
| state_entropy_gate_threshold | method | none | true | false | false | B3 entropy gate threshold. |
| state_entropy_gate_status | metric | none | true | false | false | B3 entropy gate status. |
| state_allowed_to_affect_final_score | metric | none | true | false | false | Whether state may affect final score. |
| trajectory_enabled | protocol | none | true | false | false | Whether trajectory observation is enabled. |
| trajectory_status | protocol | none | true | false | false | Trajectory status, explicit disabled in B3. |
| trajectory_state_adapter_placeholder | placeholder | _placeholder | true | false | true | B3 placeholder for trajectory state adapter. |
| ablation_family | ablation | none | true | false | false | Ablation family. |
| ablation_name | ablation | none | true | false | false | Ablation variant name. |
| ablation_removed_component | ablation | none | true | false | false | Removed component in ablation. |
| ablation_expected_effect | ablation | none | true | false | false | Expected ablation effect. |
| ablation_observed_delta_tpr | ablation | none | true | false | false | Observed TPR delta or lightweight proxy delta. |
| ablation_observed_delta_fpr | ablation | none | true | false | false | Observed FPR delta. |
| ablation_status | ablation | none | true | false | false | Ablation status. |
| ablation_failure_reason | ablation | none | true | false | false | Ablation failure reason. |
| generalization_axis | generalization | none | true | false | false | Generalization axis. |
| train_condition_id | generalization | none | true | false | false | Training or calibration condition. |
| test_condition_id | generalization | none | true | false | false | Test condition. |
| unseen_key_status | generalization | none | true | false | false | Unseen key status. |
| unseen_attack_status | generalization | none | true | false | false | Unseen attack status. |
| generalization_delta_tpr | generalization | none | true | false | false | Generalization TPR delta. |
| generalization_delta_fpr | generalization | none | true | false | false | Generalization FPR delta. |
| trajectory_source | trajectory | none | true | false | false | Trajectory trace source. |
| trajectory_source_status | trajectory | none | true | false | false | Trajectory source status. |
| trajectory_status_reason | trajectory | none | true | false | false | Trajectory source status reason. |
| trajectory_trace_id | trajectory | none | true | false | false | Trajectory trace identifier. |
| trajectory_time_grid_id | trajectory | none | true | false | false | Trajectory time grid identifier. |
| trajectory_num_steps | trajectory | none | true | false | false | Number of trajectory steps. |
| trajectory_time_points | trajectory | none | true | false | false | Trajectory time point summary. |
| trajectory_scheduler_id_placeholder | placeholder | _placeholder | true | false | true | B4 placeholder for scheduler identifier. |
| velocity_estimator_id | trajectory | none | true | false | false | Velocity estimator identifier. |
| velocity_projection_operator_id | trajectory | none | true | false | false | Velocity projection operator identifier. |
| trajectory_runtime_sec | metric | none | true | false | false | Trajectory runtime in seconds. |
| trajectory_runtime_status | metric | none | true | false | false | Trajectory runtime status. |
| trajectory_reconstruction_status | trajectory | none | true | false | false | Trajectory reconstruction status. |
| trajectory_state_adapter_status | trajectory | none | true | false | false | Trajectory-state adapter status. |
| S_trajectory_observation | metric | none | true | false | false | Trajectory observation score. |
| S_traj_state | metric | none | true | false | false | State score after trajectory observation. |
| trajectory_state_gain | metric | none | true | false | false | Score gain from trajectory-state adapter. |
| trajectory_gain_over_state_space | metric | none | true | false | false | Gain over state-space inference. |
| trajectory_negative_leakage_delta | metric | none | true | false | false | Negative tail leakage delta after trajectory. |
| trajectory_payload_correlation | metric | none | true | false | false | Correlation between trajectory and payload evidence. |
| trajectory_state_correlation | metric | none | true | false | false | Correlation between trajectory and state posterior. |
| trajectory_control_suppression_status | metric | none | true | false | false | Trajectory control suppression status. |
| trajectory_control_failure_reason | metric | none | true | false | false | Trajectory control failure reason. |
| control_type | trajectory_control | none | true | false | false | Trajectory control type. |
| control_expected_effect | trajectory_control | none | true | false | false | Expected control effect. |
| control_observed_score | trajectory_control | none | true | false | false | Observed control score. |
| control_delta_vs_main | trajectory_control | none | true | false | false | Control score delta versus main trajectory. |
| control_status | trajectory_control | none | true | false | false | Control suppression status. |
| control_not_run_reason | trajectory_control | none | true | false | false | Control not-run reason. |
| sampling_constraint_placeholder | placeholder | _placeholder | true | false | true | B4 placeholder for B6 sampling constraint config. |
| correlation_threshold | metric | none | true | false | false | B4 correlation threshold. |
| correlation_status | metric | none | true | false | false | B4 correlation audit status. |
| top_conference_trajectory_gate | governance | none | true | false | false | B4 top conference trajectory gate decision. |
| generation_model_id | protocol | none | true | false | false | B5 generation model identifier. |
| generation_model_name | protocol | none | true | false | false | B5 generation model name. |
| generation_model_family | protocol | none | true | false | false | B5 generation model family. |
| primary_sstw_tc_model_id | protocol | none | true | false | false | Primary model identifier for SSTW-TC Flow Matching evaluation. |
| primary_sstw_tc_model_status | governance | none | true | false | false | Whether the runtime model matches the configured SSTW-TC primary model. |
| generation_model_version | protocol | none | true | false | false | B5 generation model version. |
| generation_model_role | protocol | none | true | false | false | Role assigned to a generation model in the SSTW-TC evaluation plan. |
| generation_model_commit_or_hash | protocol | none | true | false | false | B5 generation model commit or hash. |
| generation_model_license_status | protocol | none | true | false | false | B5 generation model license audit status. |
| generation_backend_id | protocol | none | true | false | false | B5 generation backend identifier. |
| generation_backend_status | protocol | none | true | false | false | B5 generation backend status. |
| generation_backend_reason | protocol | none | true | false | false | B5 generation backend status reason. |
| trajectory_capture_mode | trajectory | none | true | false | false | B5 trajectory capture mode. |
| trajectory_availability_status | trajectory | none | true | false | false | B5 trajectory availability status. |
| trajectory_capture_status | trajectory | none | true | false | false | B5 trajectory capture status. |
| trajectory_capture_failure_reason | trajectory | none | true | false | false | B5 trajectory capture failure reason. |
| latent_capture_status | protocol | none | true | false | false | B5 latent capture status. |
| latent_capture_failure_reason | protocol | none | true | false | false | B5 latent capture failure reason. |
| prompt_id | protocol | none | true | false | false | B5 prompt identifier. |
| prompt_text_hash | protocol | none | true | false | false | B5 prompt text digest. |
| prompt_category | protocol | none | true | false | false | B5 prompt category. |
| scheduler_id | protocol | none | true | false | false | B5 scheduler identifier. |
| trajectory_scheduler_id | trajectory | none | true | false | false | B5 trajectory scheduler identifier. |
| num_inference_steps | protocol | none | true | false | false | B5 inference step count. |
| guidance_scale | protocol | none | true | false | false | B5 guidance scale. |
| video_length_frames | protocol | none | true | false | false | B5 generated video length in frames. |
| fps | protocol | none | true | false | false | B5 generated video fps. |
| heldout_prompt_status | generalization | none | true | false | false | B5 heldout prompt status. |
| heldout_seed_status | generalization | none | true | false | false | B5 heldout seed status. |
| gpu_validation_status | governance | none | true | false | false | B5 local GPU validation status. |
| gpu_validation_reason | governance | none | true | false | false | B5 local GPU validation reason. |
| generation_model_runnable_status | governance | none | true | false | false | B5 generation model runnable status. |
| generation_model_not_run_reason | governance | none | true | false | false | B5 generation model not run reason. |
| visual_quality_score | metric | none | true | false | false | B5 visual quality score. |
| motion_consistency_score | metric | none | true | false | false | B5 motion consistency score. |
| motion_artifact_score | metric | none | true | false | false | B5 motion artifact score. |
| motion_metric_status | metric | none | true | false | false | B5 motion metric status. |
| semantic_consistency_score | metric | none | true | false | false | B5 semantic consistency score. |
| semantic_metric_name | metric | none | true | false | false | B5 semantic metric name. |
| semantic_metric_status | metric | none | true | false | false | B5 semantic metric status. |
| metric_failure_reason | protocol | none | true | false | false | B5 metric failure reason. |
| external_baseline_name | protocol | none | true | false | false | B5 external baseline name. |
| external_baseline_version | protocol | none | true | false | false | B5 external baseline version. |
| external_baseline_runnable_status | governance | none | true | false | false | B5 external baseline runnable status. |
| external_baseline_not_run_reason | governance | none | true | false | false | B5 external baseline not run reason. |
| external_baseline_protocol_gap | governance | none | true | false | false | B5 external baseline protocol gap. |
| external_baseline_result_used_for_claim | claim | none | true | true | false | Whether B5 external baseline is used for a claim. |
| external_baseline_command_config_status | governance | none | true | false | false | 现代 external baseline 官方命令是否已配置。 |
| external_baseline_command_env_var | provenance | none | true | false | false | 现代 external baseline adapter 使用的命令环境变量名称。 |
| external_baseline_colab_preflight_decision | governance | none | true | false | false | Colab 真实 GPU 运行前现代 external baseline command preflight 的 PASS / FAIL 决策。 |
| external_baseline_colab_preflight_status | governance | none | true | false | false | Colab 真实 GPU 运行前现代 external baseline command preflight 的阻断或通过状态。 |
| external_baseline_colab_preflight_required_env_vars | provenance | none | true | false | false | Colab preflight 要求配置的现代 external baseline command 环境变量列表。 |
| external_baseline_colab_preflight_configured_env_vars | provenance | none | true | false | false | Colab preflight 已配置的现代 external baseline command 环境变量列表。 |
| external_baseline_colab_preflight_missing_env_vars | governance | none | true | false | false | Colab preflight 缺失的现代 external baseline command 环境变量列表。 |
| external_baseline_colab_preflight_required_env_var_count | metric | none | true | false | false | Colab preflight 要求配置的现代 external baseline command 环境变量数量。 |
| external_baseline_colab_preflight_configured_env_var_count | metric | none | true | false | false | Colab preflight 已配置的现代 external baseline command 环境变量数量。 |
| external_baseline_colab_preflight_missing_env_var_count | metric | none | true | false | false | Colab preflight 缺失的现代 external baseline command 环境变量数量。 |
| paper_gate_profile | protocol | none | true | false | false | 当前 Colab profile 是否属于 validation_scale / pilot_paper 这类 paper gate profile。 |
| require_modern_baseline_commands_for_paper_gate | protocol | none | true | false | false | Colab paper gate profile 是否要求现代 baseline command 在真实 GPU 运行前全部配置。 |
| run_external_baseline_source_clone | protocol | none | true | false | false | Colab 冷启动阶段是否执行可 clone 外部 baseline 官方 source 拉取。 |
| external_baseline_evidence_path_count | metric | none | true | false | false | Colab preflight 或 execution manifest 中登记的外部 baseline evidence path 数量。 |
| external_baseline_formal_ready_count | metric | none | true | false | false | external_baseline comparison 中 measured_formal record 数量。 |
| external_baseline_formal_candidate_record_count | metric | none | true | true | false | external_baseline comparison 或 pilot_paper gate 中声明 measured_formal 的候选记录数量。 |
| external_baseline_formal_incomplete_record_count | metric | none | true | true | false | external_baseline comparison 中声明 measured_formal 但缺少 anchor、clean negative 或 official evidence 的 record 数量。 |
| external_baseline_formal_measured_adapter_count | metric | none | true | false | false | external_baseline comparison 中产出 measured_formal records 的 adapter 数量。 |
| external_baseline_formal_measured_adapter_names | metric | none | true | false | false | external_baseline comparison 中产出 measured_formal records 的 adapter 名称列表。 |
| modern_external_baseline_formal_measured_adapter_count | metric | none | true | true | false | 现代视频水印 baseline 中产出 measured_formal records 的 adapter 数量。 |
| modern_external_baseline_formal_measured_adapter_names | metric | none | true | true | false | 现代视频水印 baseline 中产出 measured_formal records 的 adapter 名称列表。 |
| formal_candidate_record_count | metric | none | true | true | false | external baseline self-containment 中声明 measured_formal 的候选记录数量。 |
| formal_anchor_missing_count | metric | none | true | true | false | external baseline self-containment 中缺少 prompt / seed / attack anchor 的 formal 候选记录数量。 |
| anchor_ready | governance | none | true | true | false | external baseline self-containment 中当前 baseline 的 prompt / seed / attack anchor 是否完整。 |
| missing_anchor_modern_external_baseline_names | governance | none | true | true | false | external baseline self-containment 中缺少完整 prompt / seed / attack anchor 的现代 baseline 名称列表。 |
| score_extraction_ready | governance | none | true | true | false | external baseline self-containment 中当前 baseline 的 official bundle 是否具备分数抽取口径、分数方向和协议锚点。 |
| official_score_extraction_ready_count | metric | none | true | true | false | external baseline self-containment 中通过 official 分数抽取口径检查的 bundle record 数量。 |
| missing_score_extraction_modern_external_baseline_names | governance | none | true | true | false | external baseline self-containment 中缺少 official 分数抽取口径证据的现代 baseline 名称列表。 |
| official_adapter_baseline_id | protocol | none | true | true | false | official bundle 中声明的当前 baseline adapter 身份, 用于防止跨 baseline 误用分数。 |
| official_baseline_id | protocol | none | true | true | false | official bundle 中声明的官方 baseline 身份, 必须与当前 baseline reference runner 一致。 |
| external_baseline_official_adapter_baseline_id | protocol | none | true | true | false | measured_formal external baseline record 中转写的 official_adapter_baseline_id, 用于审计正式记录是否来自对应 baseline。 |
| external_baseline_official_baseline_id | protocol | none | true | true | false | measured_formal external baseline record 中转写的 official_baseline_id, 用于审计正式记录是否来自对应官方 baseline。 |
| official_baseline_identity_ready | governance | none | true | true | false | external baseline self-containment 行中 measured_formal record 是否保留完整 official adapter 和 official baseline 身份。 |
| official_baseline_identity_ready_count | metric | none | true | true | false | external baseline self-containment 中保留完整 official baseline 身份的 measured_formal record 数量。 |
| missing_official_identity_modern_external_baseline_names | governance | none | true | true | false | external baseline self-containment 中缺少完整 official baseline 身份的现代 baseline 名称列表。 |
| runtime_comparison_unit_id | protocol | none | true | true | false | prompt / seed / attack comparison unit 的稳定 ID, 用于对齐 SSTW 与 external baseline 的同锚点比较。 |
| official_score_extraction_policy | protocol | none | true | true | false | official bundle 中记录的逐 baseline 官方分数抽取策略。 |
| official_reference_protocol_anchor | protocol | none | true | true | false | official bundle 中记录的 prompt / seed / attack comparison unit 锚点。 |
| attack_protocol_status | protocol | none | true | true | false | official bundle 中记录的攻击协议映射或执行状态。 |
| external_baseline_detected | metric | none | true | false | false | 外部 baseline 官方 detector 是否给出 detected 判定。 |
| external_baseline_bit_accuracy | metric | none | true | false | false | 外部 baseline 官方 detector 给出的 bit accuracy 或等价 payload accuracy。 |
| external_baseline_threshold | protocol | none | true | false | false | 外部 baseline 官方 detector 使用或输出的阈值。 |
| generation_model_main_table_ready | governance | none | true | false | false | B5 main table readiness status. |
| trajectory_observation_gain_confirmed | metric | none | true | false | false | B5 trajectory gain confirmation status. |
| fixed_low_fpr_audit_pass | metric | none | true | false | false | B5 fixed low-FPR audit status. |
| quality_motion_semantic_consistency_pass | metric | none | true | false | false | B5 quality motion semantic gate status. |
| cross_prompt_generalization_pass | generalization | none | true | false | false | B5 cross prompt generalization status. |
| cross_seed_generalization_pass | generalization | none | true | false | false | B5 cross seed generalization status. |
| cross_motion_generalization_pass | generalization | none | true | false | false | B5 cross motion generalization status. |
| cross_length_generalization_pass | generalization | none | true | false | false | B5 cross length generalization status. |
| cross_prompt_seed_generalization_pass | generalization | none | true | false | false | B5 combined prompt seed generalization status. |
| generalization_failure_reason | generalization | none | true | false | false | B5 generalization failure reason. |
| formal_claim_status | claim | none | true | true | false | B5 formal claim status. |
| top_conference_b5_gate | governance | none | true | false | false | B5 top conference gate decision. |
| threshold_status | protocol | none | true | false | false | B5 threshold computation status. |
| threshold_not_run_reason | protocol | none | true | false | false | B5 threshold not run reason. |
| prompt_suite_id | protocol | none | true | false | false | B5 Colab prompt suite identifier. |
| prompt_suite_role | protocol | none | true | false | false | B5 prompt or seed role inside prompt suite. |
| prompt_suite_digest | artifact | none | true | false | false | B5 prompt suite digest. |
| dataset_construction_status | governance | none | true | false | false | B5 input dataset construction status. |
| dataset_source | protocol | none | true | false | false | B5 input dataset source description. |
| prompt_negative_text | protocol | none | false | false | false | B5 prompt negative text kept in input dataset, not formal result records. |
| colab_runtime_profile | protocol | none | true | false | false | B5 Colab runtime profile. |
| config_kind | governance | none | true | false | false | 配置或 artifact 的语义类型。 |
| config_version | governance | none | true | false | false | 配置或 artifact 的版本标识。 |
| default_drive_project_root | protocol | none | false | false | false | Colab workflow 统一配置中的默认 Google Drive 项目根目录。 |
| default_dataset_root_relative | protocol | none | false | false | false | Colab workflow 统一配置中的默认 dataset 相对目录。 |
| default_prompt_suite_path_relative | protocol | none | false | false | false | Colab workflow 统一配置中的默认 prompt suite 相对路径。 |
| default_workflow_profile_by_notebook_role | protocol | none | false | false | false | Colab workflow 统一配置中每类 Notebook 的默认 workflow profile 映射。 |
| workflow_profile_aliases | protocol | none | false | false | false | Colab workflow 统一配置中旧 profile 名称到规范 profile 名称的别名映射。 |
| workflow_profiles | protocol | none | false | false | false | Colab workflow 统一配置中所有结果层级 profile 的配置映射。 |
| workflow_profile | protocol | none | true | false | false | Colab workflow 的规范 profile 名称, 用于区分 motion_calibration、validation_scale、pilot_paper 和未来 full_paper。 |
| requested_workflow_profile | protocol | none | true | false | false | 用户或环境变量请求的原始 workflow profile 名称。 |
| canonical_workflow_profile | protocol | none | true | false | false | 经过 alias 解析后的规范 workflow profile 名称。 |
| profile_alias_applied | governance | none | true | false | false | requested_workflow_profile 是否经过 alias 映射。 |
| profile_status | governance | none | true | false | false | workflow profile 的实现或阻断状态。 |
| enabled_for_run | governance | none | true | false | false | workflow profile 当前是否允许作为可运行入口。 |
| enabled_for_claim | governance | none | true | false | false | workflow profile 当前是否允许支撑论文 claim。 |
| runtime_profile | protocol | none | true | false | false | workflow profile 映射到 experiments runner 的 runtime profile。 |
| result_tier | protocol | none | true | false | false | 当前结果层级, 例如 validation_scale、pilot_paper 或 future full_paper。 |
| notebook_role | protocol | none | true | false | false | Colab Notebook 的职责角色, 用于从统一 workflow 配置读取 stage plan。 |
| notebook_roles | protocol | none | false | false | false | Colab workflow 统一配置中 Notebook role 到路径、允许 profile 和 stage plan 的映射。 |
| notebook_path | artifact | none | true | false | false | Notebook role 对应的 Colab Notebook 文件路径。 |
| notebook_path_examples | artifact | none | true | false | false | 当某个 Notebook role 没有单独入口时, 对应该 role 的具体 Notebook 示例路径列表。 |
| entrypoint_status | governance | none | true | false | false | Notebook role 是否保留独立入口, 或是否仅由更细粒度 Notebook 复用该 role。 |
| allowed_workflow_profiles | protocol | none | true | false | false | 某个 Notebook role 允许使用的 workflow profile 列表。 |
| workflow_stage_plan | protocol | none | true | false | false | 某个 Notebook role 在指定 workflow profile 下应执行的语义阶段列表。 |
| disabled_stage_names | protocol | none | true | false | false | 某个 workflow profile 在通用 stage plan 中禁用的阶段名称列表。 |
| protocol_config_path | protocol | none | true | false | false | workflow profile 对应的 protocol gate 配置路径。 |
| drive_run_root_relative | artifact | none | false | false | false | workflow profile 对应的 Google Drive run_root 相对路径。 |
| drive_package_dir_relative | artifact | none | false | false | false | workflow profile 对应的 Google Drive package 目录相对路径。 |
| drive_log_dir_relative | artifact | none | false | false | false | workflow profile 对应的 Google Drive log 目录相对路径。 |
| motion_threshold_artifact_run_root_relative | artifact | none | false | false | false | workflow profile 复用 motion threshold calibration artifact 的 Google Drive run_root 相对路径。 |
| motion_threshold_artifact_run_root | artifact | none | true | false | false | 当前 workflow profile 复用 motion threshold calibration artifact 的绝对 run_root。 |
| method_sample_count | protocol | none | true | false | false | workflow profile 计划中的主方法样本数量。 |
| baseline_sample_count | protocol | none | true | false | false | workflow profile 计划中的 baseline 评价样本数量。 |
| max_content_records | governance | none | true | false | false | workflow profile 计划或审计中允许读取的最大内容记录数量。 |
| max_source_records | governance | none | true | false | false | workflow profile 计划或审计中允许读取的最大 source record 数量。 |
| minimum_clean_negative_count | protocol | none | true | false | false | workflow profile 或 gate 要求的 clean negative 最小数量。 |
| bootstrap_iteration_count | protocol | none | true | false | false | workflow profile 计划中的 bootstrap 迭代次数。 |
| profile_switching_note | governance | none | false | false | false | workflow profile 切换时的人类可读边界说明。 |
| cross_model_role | generalization | none | true | false | false | B5 model role for cross-model validation. |
| generation_status | protocol | none | true | false | false | B5 generation execution status. |
| generation_failure_reason | protocol | none | true | false | false | B5 generation failure reason. |
| generation_runtime_sec | metric | none | true | false | false | B5 generation runtime in seconds. |
| video_path | artifact | none | true | false | false | B5 generated video path. |
| video_sha256 | artifact | none | true | false | false | B5 generated video hash. |
| trajectory_step_index | trajectory | none | true | false | false | B5 trajectory callback step index. |
| trajectory_timestep | trajectory | none | true | false | false | B5 trajectory callback timestep. |
| latent_norm | metric | none | true | false | false | B5 latent tensor norm from trajectory callback. |
| latent_mean | metric | none | true | false | false | B5 latent tensor mean from trajectory callback. |
| latent_std | metric | none | true | false | false | B5 latent tensor standard deviation from trajectory callback. |
| cross_model_validation_status | generalization | none | true | false | false | B5 cross model validation status. |
| external_baseline_comparison_status | governance | none | true | false | false | B5 external baseline comparison status. |
| drive_project_root | artifact | none | true | false | false | Google Drive SSTW project root used by Colab workflow. |
| drive_dataset_root | artifact | none | true | false | false | Google Drive dataset output directory for B5 Colab workflow. |
| drive_run_root | artifact | none | true | false | false | Google Drive run output directory for B5 Colab workflow. |
| drive_package_dir | artifact | none | true | false | false | Google Drive package output directory for B5 Colab workflow. |
| drive_log_dir | artifact | none | true | false | false | Google Drive log output directory for B5 Colab workflow. |
| run_root | artifact | none | true | false | false | Run root packaged by Drive packager. |
| archive_path | artifact | none | true | false | false | Archive path created by Drive packager. |
| package_manifest_path | artifact | none | true | false | false | Package manifest path created by Drive packager. |
| include_videos | protocol | none | true | false | false | Whether generated videos are included in Drive package. |
| created_at | protocol | none | true | false | false | Creation timestamp for package manifest. |
| decision_summary | governance | none | true | false | false | Summary of stage decision embedded in package manifest. |
| generation_manifest_status | governance | none | true | false | false | Status showing whether generation manifest was present during packaging. |
| hf_token_status | governance | none | true | false | false | Whether HF_TOKEN was provided to Colab runtime; token value is never recorded. |
| implementation_evidence_status | governance | none | true | false | false | B5 Colab result checker implementation evidence status. |
| mechanism_evidence_status | governance | none | true | false | false | B5 Colab result checker mechanism evidence status. |
| missing_mechanism_requirements | governance | none | true | false | false | B5 missing mechanism requirements list. |
| successful_generation_count | metric | none | true | false | false | Number of successful generation records. |
| external_baseline_runnable_count | metric | none | true | false | false | Number of runnable external baselines. |
| quality_metric_ready_count | metric | none | true | false | false | Number of ready quality metric records. |
| video_checks | artifact | none | true | false | false | Video integrity check records. |
| video_exists | artifact | none | true | false | false | Whether generated video exists. |
| video_size_bytes | artifact | none | true | false | false | Generated video size in bytes. |
| video_sha256_match | artifact | none | true | false | false | Whether generated video hash matches record. |
| expected_video_sha256 | artifact | none | true | false | false | Expected generated video hash from records. |
| actual_video_sha256 | artifact | none | true | false | false | Actual generated video hash from file. |
| next_recommended_profile | governance | none | true | false | false | Recommended next Colab profile after result checking. |
| external_baseline_family | protocol | none | true | false | false | B5 external baseline method family. |
| external_baseline_source_url | protocol | none | true | false | false | B5 external baseline source URL or local source marker. |
| external_baseline_recommended_priority | governance | none | true | false | false | B5 external baseline recommendation priority. |
| external_baseline_selection_role | governance | none | true | false | false | B5 role assigned to the external baseline. |
| external_baseline_integration_status | governance | none | true | false | false | B5 external baseline integration status. |
| external_baseline_layer | governance | none | true | false | false | Comparison layer for external baseline, such as modern external baseline or explicit synchronization control. |
| external_baseline_adapter_status | governance | none | true | false | false | Whether the external baseline has a repository adapter ready for governed execution. |
| external_baseline_input_compatibility_status | governance | none | true | false | false | Whether the external baseline can consume the current project video, key, attack, and split inputs. |
| external_baseline_output_record_status | governance | none | true | false | false | Whether the external baseline produces governed records or only a governed non-run record. |
| external_baseline_threshold_policy_compatible | governance | none | true | false | false | Whether the external baseline can be evaluated under the same frozen threshold policy. |
| external_baseline_attack_manifest_compatible | governance | none | true | false | false | Whether the external baseline can be evaluated under the same attack manifest. |
| external_baseline_main_comparison_ready | governance | none | true | false | false | Whether the baseline satisfies all conditions to enter the main comparison table. |
| external_baseline_claim_support_status | claim | none | true | true | false | Claim support boundary for external baseline records. |
| external_baseline_status_decision | governance | none | true | false | false | Decision for the external baseline status audit. |
| modern_external_baseline_record_count | metric | none | true | false | false | Number of modern external baseline governed status records. |
| modern_external_baseline_status_records_ready | governance | none | true | false | false | Whether modern external baselines have governed status records or non-run records. |
| modern_external_baseline_main_comparison_ready_count | metric | none | true | false | false | Number of modern external baselines ready for main comparison. |
| external_baseline_main_comparison_ready_count | metric | none | true | false | false | Number of all external baselines ready for main comparison. |
| external_baseline_non_run_record_count | metric | none | true | false | false | Number of external baselines represented by governed non-run records. |
| external_baseline_adapter_path | artifact | none | true | false | false | Path of the repository adapter under external_baseline used to produce comparison records. |
| external_baseline_score_record_id | protocol | none | true | false | false | Stable identifier for one external baseline comparison score record. |
| external_baseline_score_status | governance | none | true | false | false | Per-record status of an external baseline score, such as measured_proxy or adapter_not_integrated. |
| external_baseline_score_source | governance | none | true | false | false | Evidence source used by an external baseline adapter score. |
| external_baseline_score_failure_reason | governance | none | true | false | false | Failure reason for an unsupported external baseline score record. |
| external_baseline_official_output_path | artifact | none | true | true | false | Path to a persisted official external baseline output JSON generated by the command adapter. |
| external_baseline_official_stdout_path | artifact | none | true | true | false | Path to persisted stdout from an official external baseline command. |
| external_baseline_official_stderr_path | artifact | none | true | true | false | Path to persisted stderr from an official external baseline command. |
| external_baseline_official_command_manifest_path | artifact | none | true | true | false | Path to the governed manifest for one official external baseline command execution. |
| external_baseline_official_result_provenance | governance | none | true | true | false | 现代 external baseline official bundle 是否来自项目内第三方官方代码运行链路的 provenance 标记。 |
| external_baseline_official_result_bundle_path | artifact | none | true | true | false | 项目内 official reference 流程生成的单条 baseline 结果 bundle JSON 路径。 |
| external_baseline_official_execution_manifest_path | artifact | none | true | true | false | 项目内 official reference 流程生成的 baseline execution manifest 路径。 |
| external_baseline_official_score_extraction_policy | protocol | none | true | true | false | measured_formal external baseline record 中从 official bundle 继承的官方分数抽取策略。 |
| external_baseline_official_reference_protocol_anchor | protocol | none | true | true | false | measured_formal external baseline record 中从 official bundle 继承的 prompt / seed / attack comparison unit 锚点。 |
| external_baseline_attack_protocol_status | protocol | none | true | true | false | measured_formal external baseline record 中从 official bundle 继承的攻击协议映射或执行状态。 |
| external_baseline_reference_sequence_length | metric | none | true | false | false | Reference trajectory sequence length consumed by an external synchronization baseline adapter. |
| external_baseline_observed_sequence_length | metric | none | true | false | false | Observed trajectory or runtime metadata sequence length consumed by an external synchronization baseline adapter. |
| external_baseline_distance | metric | none | true | false | false | Distance or cost reported by an external synchronization baseline adapter. |
| external_baseline_score | metric | none | true | false | false | Proxy score reported by an external baseline adapter under the common comparison protocol. |
| external_baseline_comparison_decision | governance | none | true | false | false | Decision for external_baseline adapter comparison output readiness. |
| external_baseline_comparison_record_count | metric | none | true | false | false | Number of records written to external_baseline_score_records.jsonl. |
| external_baseline_comparison_ready_count | metric | none | true | false | false | Number of measured external baseline comparison score records. |
| external_baseline_measured_adapter_count | metric | none | true | false | false | Number of external_baseline adapters with measured proxy comparison records. |
| external_baseline_measured_adapter_names | protocol | none | true | false | false | Names of external_baseline adapters with measured proxy comparison records. |
| external_baseline_unsupported_adapter_count | metric | none | true | false | false | Number of unsupported external baseline comparison records or rows. |
| external_baseline_comparison_table_status | governance | none | true | false | false | Whether the external baseline comparison table was rebuilt from governed records. |
| validation_external_baseline_comparison_records_ready | governance | none | true | false | false | Whether validation-scale gate found external_baseline comparison records and enough measured adapters. |
| minimum_external_baseline_measured_adapter_count | protocol | none | true | false | false | Minimum measured external_baseline adapter count required by validation-scale gate. |
| external_baseline_source_intake_decision | governance | none | true | false | false | Decision for external baseline source intake manifest readiness. |
| external_baseline_source_intake_ready_count | metric | none | true | false | false | Number of external baseline sources that are local, present, or command-configured. |
| external_baseline_source_intake_missing_count | metric | none | true | false | false | Number of external baseline sources still requiring official source or command configuration. |
| external_baseline_source_intake_manifest_path | artifact | none | true | false | false | Path to the generated source intake manifest for external baseline governance. |
| source_intake_status | governance | none | true | false | false | Per-baseline source intake state, such as source snapshot available or manual command required. |
| source_intake_action_required | governance | none | true | false | false | Next action required before the external baseline can run formally. |
| source_cloneable | governance | none | true | false | false | Whether the registered source URL can be handled by git clone. |
| source_dir_exists | artifact | none | true | false | false | Whether the expected external baseline source directory exists. |
| source_dir_file_count | metric | none | true | false | false | Number of auditable files found under an external baseline source directory. |
| source_inspection_status | governance | none | true | false | false | Whether a source directory was inspected or is missing. |
| source_inspection_decision | governance | none | true | false | false | Decision for the source inspection manifest. |
| clone_operation_status | governance | none | true | false | false | Status of a planned or executed external baseline source clone operation. |
| clone_failure_reason | governance | none | true | false | false | Failure or non-execution reason for external baseline source clone operation. |
| clone_results_decision | governance | none | true | false | false | Decision for external baseline clone results manifest. |
| external_baseline_execution_manifest_status | governance | none | true | false | false | Whether external_baseline_execution_manifest.json is present in a packaged run. |
| formal_evidence_status | governance | none | true | true | false | Whether measured_formal external baseline rows are bound to explicit evidence paths. |
| evidence_paths | artifact | none | true | true | false | Existing log, config, output, or source-evidence paths bound to an execution manifest. |
| evidence_path_count | metric | none | true | true | false | Number of evidence paths bound to an execution manifest. |
| manifest_kind | artifact | none | false | false | false | Kind of manifest written by a governance runner or source intake tool. |
| baseline_source_count | metric | none | false | false | false | Number of external baseline source entries in the source intake manifest. |
| modern_external_baseline_source_count | metric | none | false | false | false | Number of modern external baseline source entries in source intake. |
| source_intake_ready_count | metric | none | false | false | false | Number of source intake rows that are ready through local code, source snapshot, or configured command. |
| source_intake_missing_count | metric | none | false | false | false | Number of source intake rows still missing source or official command configuration. |
| modern_external_baseline_source_ready_count | metric | none | false | false | false | Number of modern external baseline rows ready through source snapshot or configured command. |
| adapter_exists | governance | none | false | false | false | Whether the registered adapter path exists in the repository. |
| source_dir_top_level_entries | artifact | none | false | false | false | Top-level files or folders found in an external baseline source directory. |
| source_inspection_record_count | metric | none | false | false | false | Number of source inspection rows written for external baselines. |
| source_inspection_ready_count | metric | none | false | false | false | Number of external baseline source directories inspected successfully. |
| source_inspection_missing_count | metric | none | false | false | false | Number of external baseline source directories missing during inspection. |
| source_inspections | artifact | none | false | false | false | Source inspection row list in external baseline source inspection manifest. |
| metadata_files | artifact | none | false | false | false | Candidate metadata, dependency, README, or license files discovered in an external source tree. |
| entrypoint_candidate_files | artifact | none | false | false | false | Candidate run, eval, detect, or infer entrypoint files discovered in an external source tree. |
| clone_result_count | metric | none | false | false | false | Number of clone result rows in external baseline clone manifest. |
| clone_executed_count | metric | none | false | false | false | Number of clone or update operations actually executed. |
| clone_failed_count | metric | none | false | false | false | Number of failed clone or update operations. |
| clone_results | artifact | none | false | false | false | Clone result row list for external baseline source preparation. |
| execute_clone | protocol | none | false | false | false | Whether the source intake command was allowed to execute git clone or fetch. |
| planned_repository_url | artifact | none | false | false | false | Git URL that would be cloned when execute_clone is enabled. |
| git_results | artifact | none | false | false | false | Structured stdout, stderr, and return code summaries from git operations. |
| stdout_tail | artifact | none | false | false | false | Tail of stdout captured from a governed external command. |
| stderr_tail | artifact | none | false | false | false | Tail of stderr captured from a governed external command. |
| return_code | metric | none | false | false | false | Process return code captured from a governed external command. |
| table_plan_path | artifact | none | false | false | false | Path to external baseline table plan generated by source intake. |
| method_count | metric | none | false | false | false | Number of methods listed in an external baseline table plan. |
| modern_external_baseline_count | metric | none | false | false | false | Number of modern external baselines listed in a table plan. |
| explicit_synchronization_control_count | metric | none | false | false | false | Number of explicit synchronization controls listed in a table plan. |
| comparison_layer | governance | none | false | false | false | External baseline comparison layer, such as modern external baseline or synchronization control. |
| claim_boundary | claim | none | false | true | false | Claim boundary assigned to an external baseline table plan method. |
| local_source_root | artifact | none | false | false | false | Local source root path for an external baseline. |
| formal_result_claim | claim | none | false | true | false | Whether an execution manifest declares formal result evidence. |
| execution_boundary | governance | none | false | true | false | Boundary describing how external baseline outputs were produced and bound to evidence. |
| comparison_unit_count | metric | none | true | false | false | Number of comparable runtime detection units covered by a non-run external baseline comparison row. |
| comparison_record_count | metric | none | true | false | false | Number of records aggregated into one comparison table row. |
| comparison_attack_count | metric | none | true | false | false | Number of attacks represented by one comparison table row. |
| comparison_scope | governance | none | true | false | false | Scope of a method or baseline comparison table row. |
| method_id | protocol | none | true | false | false | Stable identifier for a method or baseline in comparison tables. |
| method_role | protocol | none | true | false | false | Role of a method or baseline in comparison tables. |
| metric_status | governance | none | true | false | false | Whether a comparison row or record contains measured proxy metrics or remains unsupported. |
| official_source_dir | artifact | none | true | false | false | VideoSeal 或其他 external baseline 官方源码目录。 |
| official_source_dir_exists | governance | none | true | false | false | 官方源码目录是否存在。 |
| official_source_layout_status | governance | none | true | false | false | 官方源码运行布局状态, 例如 VideoSeal root config 是否可解析。 |
| official_source_layout_audit | artifact | none | false | false | false | 官方源码运行布局审计对象, 用于说明 Notebook/Colab 运行前置条件。 |
| official_source_runtime_cwd | artifact | none | true | false | false | 调用官方 API 时临时使用的官方源码工作目录。 |
| official_video_io_backend | artifact | none | true | false | false | official baseline wrapper 使用的视频文件 I/O 后端。 |
| colab_torch_stack_policy | governance | none | false | false | false | Colab 中是否保持预装 torch / torchvision 运行栈不被 requirements 覆盖。 |
| stage_package_handoff_mode | governance | none | true | false | false | Colab Notebook 是否启用本地 workspace 与阶段 zip 交接, 例如 local_zip。 |
| stage_package_id | artifact | none | true | false | false | Notebook 阶段 zip 交接包的稳定语义 ID。 |
| stage_package_dir | artifact | none | true | false | false | 阶段 zip 交接包在 Google Drive 中的归档目录。 |
| local_stage_workspace_root | artifact | none | true | false | false | Colab 中用于热路径读写的本地 workspace 根目录。 |
| local_stage_package_cache_root | artifact | none | true | false | false | Colab 中缓存从 Drive 复制来的阶段 zip 的本地目录。 |
| stage_package_restore_status | governance | none | true | false | false | 前置阶段 zip 是否已恢复到本地 workspace。 |
| stage_package_source_kind | governance | none | false | false | false | 恢复阶段包时使用的是新阶段 zip 还是 legacy Drive package zip。 |
| stage_package_source_workflow_profile | governance | none | false | false | false | 恢复阶段包时实际读取的来源 workflow profile, 用于表达 motion calibration 等跨 profile 复用规则。 |
| stage_package_target_workflow_profile | governance | none | false | false | false | 当前 Notebook 正在运行的目标 workflow profile, 用于审计跨 profile 阶段包恢复。 |
| stage_package_publish_status | governance | none | true | false | false | 当前阶段 zip 是否已发布到 Google Drive。 |
| stage_package_archive_sha256 | artifact | none | true | false | false | 阶段 zip 文件的 sha256 摘要, 用于校验交接包完整性。 |
| stage_package_entry_count | metric | none | true | false | false | 阶段 zip 中归档的文件条目数量。 |
| stage_package_source_root_count | metric | none | true | false | false | 生成阶段 zip 时纳入的本地源目录数量。 |
| stage_package_source_roots | artifact | none | false | false | false | 生成阶段 zip 时纳入的本地源目录和 zip 内归档根路径列表。 |
| local_stage_package_zip | artifact | none | true | false | false | 阶段 zip 在 Colab 本地缓存中的路径。 |
| drive_stage_package_zip | artifact | none | true | false | false | 阶段 zip 在 Google Drive 中的时间戳归档路径。 |
| latest_drive_stage_package_zip | artifact | none | true | false | false | 历史兼容字段; 2026-07-01 后正式阶段包默认使用时间戳 zip, 该字段应为空。 |
| stage_package_manifest_path | artifact | none | true | false | false | 阶段 zip manifest 在 Google Drive 中的时间戳归档路径。 |
| latest_stage_package_manifest_path | artifact | none | true | false | false | 历史兼容字段; 2026-07-01 后正式阶段包默认使用时间戳 manifest, 该字段应为空。 |
| stage_package_file_stem | artifact | none | true | false | false | 阶段 zip 的规范文件名前缀, 格式为 `<workflow_profile>_<stage_package_id>_<YYYYMMDD_HHMMSS>_<git_short_commit>`。 |
| external_baseline_resource_package_restore_status | governance | none | false | false | false | Colab 是否从 Drive resources zip 包恢复了本地 external baseline 资源根目录。 |
| resource_package_count | metric | none | false | false | false | 本次从 Drive resources 中检测到并解包的资源 zip 数量。 |
| extracted_resource_file_count | metric | none | false | false | false | 本次从资源 zip 解压出的文件数量。 |
| required_stage_package_ids | governance | none | false | false | false | 当前 Notebook 必须先恢复的前置阶段包 ID 列表。 |
| optional_stage_package_ids | governance | none | false | false | false | 当前 Notebook 可恢复但不强制存在的阶段包 ID 列表。 |
| restored_stage_package_count | metric | none | false | false | false | 当前 Notebook 已成功恢复到本地 workspace 的阶段包数量。 |
| video_io_backend | artifact | none | false | false | false | 视频张量 I/O helper 返回的实际后端名称。 |
| required_working_directory | artifact | none | false | false | false | 第三方官方代码加载模型或配置时要求的工作目录。 |
| runtime_cwd_policy | governance | none | false | false | false | 第三方官方代码加载阶段的临时 cwd 切换策略。 |
| source_root_config_paths | artifact | none | false | false | false | 官方源码根目录下需要被相对路径解析的配置文件列表。 |
| package_fallback_config_paths | artifact | none | false | false | false | 官方包目录下可作为后备解析位置的配置文件列表。 |
| missing_required_config_paths | artifact | none | false | false | false | 官方源码运行布局审计中缺失的必要配置路径。 |
| config_relative_path | artifact | none | false | false | false | 官方源码运行布局审计中配置文件相对路径。 |
| config_path | artifact | none | false | false | false | 官方源码运行布局审计中配置文件完整路径。 |
| config_file_exists | governance | none | false | false | false | 官方源码运行布局审计中配置文件是否存在。 |
| layout_decision | governance | none | false | false | false | 官方源码运行布局审计 PASS/FAIL 判定。 |
| layout_status | governance | none | false | false | false | 官方源码运行布局审计的具体状态。 |
| proposed_method_score_mean | metric | none | true | false | false | Mean SSTW runtime detection proxy score in an external baseline comparison table. |
| external_baseline_score_mean | metric | none | true | false | false | Mean external baseline adapter proxy score in a comparison table. |
| external_baseline_distance_mean | metric | none | true | false | false | Mean external baseline adapter distance in a comparison table. |
| baseline_score_margin_mean | metric | none | true | false | false | Mean score margin between SSTW runtime proxy score and external baseline adapter proxy score. |
| validation_scale_gate_decision | governance | none | true | false | false | Decision for validation-scale generative probe gate before pilot_paper full-protocol run. |
| validation_scale_result_level | governance | none | true | true | false | package manifest 中记录的 validation_scale 结果级别。 |
| validation_scale_target_fpr | protocol | none | true | false | false | package manifest 中记录的 validation_scale protocol config target_fpr 摘要。 |
| missing_validation_requirements | governance | none | true | false | false | Validation-scale requirements that are not yet satisfied. |
| validation_missing_requirement_count | metric | none | true | false | false | Count of missing validation-scale requirements. |
| validation_scale_hard_required_config_missing | governance | none | true | false | false | validation_scale 阶段不可通过配置关闭的公平比较硬前置缺口列表。 |
| validation_scale_hard_required_config_missing_count | metric | none | true | false | false | validation_scale 公平比较硬前置配置缺口数量。 |
| validation_profile_names | protocol | none | true | false | false | Runtime profile names accepted as validation-scale generation records. |
| validation_generation_record_count | metric | none | true | false | false | Number of successful generation records produced by validation-scale profiles. |
| validation_prompt_count | metric | none | true | false | false | Number of prompts covered by validation-scale generation records. |
| validation_seed_per_prompt_min | metric | none | true | false | false | Minimum seed count per prompt in validation-scale generation records. |
| minimum_prompt_count | protocol | none | true | false | false | Minimum prompt count required by a gate. |
| minimum_seed_per_prompt | protocol | none | true | false | false | Minimum seed count per prompt required by a gate. |
| internal_ablation_record_count | metric | none | true | false | false | Number of internal ablation records available to a validation or full-paper gate. |
| internal_ablation_status | governance | none | true | false | false | Internal ablation readiness or claim boundary status. |
| validation_internal_ablation_decision | governance | none | true | false | false | Decision for validation-scale internal ablation proxy runner. |
| validation_internal_ablation_variant_count | metric | none | true | false | false | Number of method variants covered by validation-scale internal ablation records. |
| validation_internal_ablation_attack_count | metric | none | true | false | false | Number of attacks covered by validation-scale internal ablation records. |
| validation_internal_ablation_score_margin | metric | none | true | false | false | Mean score margin between full method and ablated proxy variants in validation-scale. |
| validation_internal_ablation_evidence_level | governance | none | true | false | false | Evidence level for validation-scale internal ablation records. |
| ablation_runtime_profile | protocol | none | true | false | false | Internal ablation record 对应的 runtime profile, 用于区分 validation_scale 与 pilot_paper 覆盖。 |
| validation_internal_ablation_profile_counts | metric | none | true | false | false | Internal ablation records 按 runtime profile 汇总的数量映射。 |
| pilot_paper_internal_ablation_record_count | metric | none | true | true | false | Internal ablation records 中属于 pilot_paper profile 的数量。 |
| validation_ablation_evidence_level | governance | none | true | false | false | Per-record evidence level for validation-scale ablation proxy records. |
| validation_ablation_source_score | metric | none | true | false | false | Source runtime detection proxy score used by a validation-scale ablation record. |
| validation_ablation_proxy_score | metric | none | true | false | false | Derived validation-scale ablation proxy score. |
| adaptive_attack_record_count | metric | none | true | false | false | Number of Flow-specific adaptive attack records available to a validation or full-paper gate. |
| adaptive_attack_status | governance | none | true | false | false | Adaptive attack readiness or claim boundary status. |
| adaptive_attack_decision | governance | none | true | false | false | Adaptive attack validation runner 决策。 |
| adaptive_attack_name | protocol | none | true | false | false | Adaptive attack 名称。 |
| adaptive_attack_family | protocol | none | true | false | false | Adaptive attack 家族。 |
| adaptive_attack_strength | metric | none | true | false | false | Adaptive attack 强度。 |
| adaptive_attack_budget | protocol | none | true | false | false | Adaptive attack 预算。 |
| attack_knowledge_level | protocol | none | true | false | false | 攻击者知识层级。 |
| targeted_evidence_layer | protocol | none | true | false | false | Adaptive attack 目标证据层。 |
| endpoint_preservation_status | governance | none | true | false | false | Endpoint 保持状态。 |
| path_response_suppression_score | metric | none | true | false | false | Path response suppression proxy 分数。 |
| velocity_projection_suppression_score | metric | none | true | false | false | Velocity projection suppression proxy 分数。 |
| adaptive_residual_proxy_score | metric | none | true | false | false | Adaptive attack 后残余 proxy 分数。 |
| replay_signature_mismatch_status | governance | none | true | false | false | Replay signature mismatch 状态。 |
| trajectory_sketch_tamper_status | governance | none | true | false | false | Trajectory sketch tamper 状态。 |
| quality_guard_status | governance | none | true | false | false | Quality guard 状态。 |
| semantic_projection_status | governance | none | true | false | false | Semantic projection 状态。 |
| adaptive_negative_fpr | metric | none | true | false | false | Adaptive negative FPR。 |
| adaptive_negative_fpr_status | governance | none | true | false | false | Adaptive negative FPR 可用状态。 |
| adaptive_attack_success_status | governance | none | true | false | false | Adaptive attack success 或 proxy 状态。 |
| adaptive_attack_claim_support_status | claim | none | true | false | false | Adaptive attack claim 支撑状态。 |
| adaptive_attack_name_count | metric | none | true | false | false | Adaptive attack 名称覆盖数量。 |
| adaptive_attack_family_count | metric | none | true | false | false | Adaptive attack family 覆盖数量。 |
| adaptive_attack_knowledge_level_count | metric | none | true | false | false | Adaptive attack 攻击者知识层级覆盖数量。 |
| adaptive_attack_targeted_layer_count | metric | none | true | false | false | Adaptive attack 目标证据层覆盖数量。 |
| adaptive_attack_missing_names | governance | none | true | false | false | 缺失的 adaptive attack 名称列表。 |
| adaptive_attack_score_mean | metric | none | true | false | false | Adaptive residual proxy 平均分数。 |
| adaptive_robustness_claim_allowed | governance | none | true | false | false | 是否允许 adaptive robustness 强 claim。 |
| adaptive_attack_evidence_level | governance | none | true | false | false | Adaptive attack evidence 等级。 |
| replay_or_sketch_status | governance | none | true | false | false | Replay/sketch readiness status or explicit Claim-3 downgrade status. |
| claim3_downgrade_decision | governance | none | true | false | false | Claim-3 降级门禁决策。 |
| claim3_downgraded | governance | none | true | false | false | 是否已经显式将 Claim-3 降级。 |
| claim3_original_scope | claim | none | true | false | false | Claim-3 原始强主张范围。 |
| claim3_allowed_scope | claim | none | true | false | false | 当前证据允许的 Claim-3 范围。 |
| claim3_downgrade_reason | governance | none | true | false | false | Claim-3 降级或未降级的原因。 |
| claim3_full_support_allowed | governance | none | true | false | false | 是否允许把 Claim-3 写成强 supported claim。 |
| claim3_missing_replay_requirement_count | metric | none | true | false | false | Claim-3 replay/sketch 缺失要求数量。 |
| claim3_missing_replay_requirements | governance | none | true | false | false | Claim-3 replay/sketch 缺失要求列表。 |
| authenticated_trajectory_sketch_status | governance | none | true | false | false | Authenticated trajectory sketch 就绪状态。 |
| trajectory_sketch_verification_status | governance | none | true | false | false | Trajectory sketch 验证状态。 |
| replay_uncertainty_records_ready | governance | none | true | false | false | Replay uncertainty records 是否就绪。 |
| wrong_sampler_replay_records_ready | governance | none | true | false | false | Wrong sampler replay records 是否就绪。 |
| wrong_prompt_replay_records_ready | governance | none | true | false | false | Wrong prompt replay records 是否就绪。 |
| confidence_interval_status | governance | none | true | false | false | Statistical confidence interval report readiness status. |
| statistical_confidence_interval_decision | governance | none | true | false | false | Decision for validation-scale statistical confidence interval reporter. |
| statistical_confidence_interval_family | governance | none | true | false | false | Metric family covered by a statistical confidence interval record. |
| ci_record_count | metric | none | true | false | false | Number of confidence interval records. |
| ci_success_count | metric | none | true | false | false | Count of successful events used in a confidence interval. |
| ci_total_count | metric | none | true | false | false | Total event count used in a confidence interval. |
| ci_point_estimate | metric | none | true | false | false | Point estimate for a confidence interval. |
| ci_wilson_lower | metric | none | true | false | false | Wilson lower bound for a binomial confidence interval. |
| ci_wilson_upper | metric | none | true | false | false | Wilson upper bound for a binomial confidence interval. |
| ci_confidence_level | protocol | none | true | false | false | Confidence level used by a confidence interval record. |
| ci_evidence_level | governance | none | true | false | false | Evidence level for a validation-scale confidence interval record. |
| cluster_by_video_interval_status | governance | none | true | false | false | Status of cluster-by-video confidence interval availability. |
| paper_low_fpr_ci_status | governance | none | true | false | false | Status of paper-level low-FPR confidence interval availability. |
| artifact_rebuild_status | governance | none | true | false | false | Artifact rebuild dry-run readiness status. |
| validation_artifact_rebuild_dry_run_decision | governance | none | true | false | false | Decision for validation-scale artifact rebuild dry-run. |
| artifact_rebuild_check_record_count | metric | none | true | false | false | Number of artifact rebuild dry-run check records. |
| artifact_rebuild_missing_count | metric | none | true | false | false | Number of missing artifacts in a rebuild dry-run. |
| artifact_rebuild_missing_paths | artifact | none | true | false | false | List of missing artifact paths from a rebuild dry-run. |
| artifact_rebuild_scope | governance | none | true | false | false | Scope of an artifact rebuild dry-run. |
| artifact_role | artifact | none | true | false | false | Whether an artifact is required input or required output. |
| artifact_rebuild_check_scope | governance | none | true | false | false | Per-record scope of artifact rebuild dry-run. |
| artifact_relative_path | artifact | none | true | false | false | Artifact path relative to run root checked by rebuild dry-run. |
| artifact_exists | artifact | none | true | false | false | Whether an artifact exists during rebuild dry-run. |
| artifact_size_bytes | metric | none | true | false | false | Artifact size observed during rebuild dry-run. |
| artifact_status | governance | none | true | false | false | Per-artifact rebuild dry-run status. |
| full_paper_allowed | governance | none | true | false | false | Whether current gate allows full-paper result package execution. |
| full_paper_next_gate | governance | none | true | false | false | Next gate required before full-paper result production. |
| selection_policy | governance | none | true | false | false | B5 external baseline selection policy block. |
| primary_selection_rule | governance | none | true | false | false | B5 external baseline primary selection rule. |
| claim_rule | claim | none | true | true | false | B5 external baseline claim usage rule. |
| fallback_rule | governance | none | true | false | false | B5 external baseline fallback rule. |
| internal_mechanism_baselines | protocol | none | true | false | false | B5 internal mechanism baseline list paired with external explicit synchronization baselines. |
| mechanism_score_source | protocol | none | true | false | false | Source used for B5 mechanism postprocess score records. |
| baseline_score_margin | metric | none | true | false | false | Score margin between key-conditioned trajectory score and the compared baseline score. |
| latent_norm_range | metric | none | true | false | false | Range of latent norm values in a captured generation trajectory. |
| latent_norm_total_variation | metric | none | true | false | false | Total variation of latent norm values across a captured generation trajectory. |
| latent_directed_norm_drop | metric | none | true | false | false | Directed latent norm drop used by B5 trajectory proxy scoring. |
| latent_mean_range | metric | none | true | false | false | Range of latent mean values in a captured generation trajectory. |
| latent_std_range | metric | none | true | false | false | Range of latent standard deviation values in a captured generation trajectory. |
| trajectory_observation_proxy_score | metric | none | true | false | false | Proxy trajectory observation score derived from Colab latent callback records. |
| control_name | protocol | none | true | false | false | Controlled negative trajectory control name used by B5 postprocess. |
| controlled_negative_count | metric | none | true | false | false | Number of controlled negative records used by B5 postprocess thresholding. |
| controlled_negative_false_positive_count | metric | none | true | false | false | Number of controlled negatives above the postprocess threshold. |
| controlled_negative_fpr | metric | none | true | false | false | Controlled negative false positive rate for B5 postprocess. |
| fixed_low_fpr_proxy_pass | metric | none | true | false | false | Whether B5 postprocess controlled negative FPR satisfies target FPR. |
| visual_quality_proxy_score | metric | none | true | false | false | Lightweight visual quality proxy score for generated video outputs. |
| visual_quality_proxy_status | governance | none | true | false | false | Status for lightweight visual quality proxy records. |
| motion_consistency_proxy_score | metric | none | true | false | false | Lightweight motion consistency proxy score from trajectory statistics. |
| motion_consistency_proxy_status | governance | none | true | false | false | Status for lightweight motion consistency proxy records. |
| semantic_consistency_proxy_status | governance | none | true | false | false | Status for semantic consistency proxy records. |
| video_file_local_status | artifact | none | true | false | false | Whether a generated video path is locally available during postprocess. |
| mechanism_postprocess_decision | governance | none | true | false | false | B5 mechanism postprocess decision based on proxy records. |
| mechanism_score_record_count | metric | none | true | false | false | Number of B5 mechanism score records produced by postprocess. |
| controlled_negative_record_count | metric | none | true | false | false | Number of controlled negative records produced by postprocess. |
| quality_proxy_record_count | metric | none | true | false | false | Number of quality motion semantic proxy records produced by postprocess. |
| key_conditioned_score_mean | metric | none | true | false | false | Mean key-conditioned trajectory proxy score across generated positives. |
| best_baseline_score_mean | metric | none | true | false | false | Best mean baseline proxy score across compared B5 baselines. |
| trajectory_gain_over_best_baseline | metric | none | true | false | false | Mean proxy gain over the strongest compared baseline. |
| trajectory_gain_confirmed_by_proxy | metric | none | true | false | false | Whether B5 postprocess proxy scores show trajectory gain. |
| quality_motion_semantic_proxy_pass | metric | none | true | false | false | Whether lightweight quality and motion proxies are ready; semantic remains proxy status. |
| formal_quality_semantic_ready | governance | none | true | false | false | Whether formal quality and semantic metrics are ready for positive claim use. |
| mechanism_postprocess_status | governance | none | true | false | false | Result checker status for B5 mechanism postprocess artifacts. |
| postprocess_stage_id | governance | none | true | false | false | Stage id reported by B5 mechanism postprocess decision. |
| postprocess_mechanism_decision | governance | none | true | false | false | Formal mechanism decision reported by postprocess artifacts. |
| runtime_mechanism_decision | governance | none | true | false | false | Raw mechanism decision reported by the runtime generation stage before postprocess or pilot gate aggregation. |
| effective_mechanism_decision | governance | none | true | false | false | Package-level mechanism decision after applying governed postprocess and small-scale pilot evidence precedence. |
| mechanism_decision_source | governance | none | true | false | false | Artifact source used to derive the package-level effective mechanism decision. |
| video_decode_status | governance | none | true | false | false | Decode status for generated mp4 files used by B5 formal metrics. |
| video_metric_failure_reason | governance | none | true | false | false | Failure reason for generated video file metric extraction. |
| decoded_frame_count | metric | none | true | false | false | Number of decoded frames sampled from a generated video. |
| sampled_frame_count | metric | none | true | false | false | Number of frames sampled for B5 video file metrics. |
| mean_brightness | metric | none | true | false | false | Mean brightness over sampled generated video frames. |
| mean_contrast | metric | none | true | false | false | Mean contrast over sampled generated video frames. |
| dark_pixel_ratio | metric | none | true | false | false | Ratio of near-black pixels over sampled generated video frames. |
| bright_pixel_ratio | metric | none | true | false | false | Ratio of near-white pixels over sampled generated video frames. |
| motion_delta_score | metric | none | true | false | false | Mean adjacent-frame absolute difference over sampled generated frames. |
| motion_delta_p90_score | metric | none | true | false | false | Mean 90th percentile adjacent-frame absolute difference over sampled frames. |
| motion_delta_top10_mean_score | metric | none | true | false | false | Mean high-difference region adjacent-frame score over sampled frames. |
| motion_delta_focus_score | metric | none | true | false | false | Local-motion-oriented score computed as high-difference region mean minus median frame difference. |
| motion_delta_focus_to_mean_ratio | metric | none | true | false | false | Ratio between focus motion score and full-frame mean motion score. |
| motion_calibration_score | metric | none | true | false | false | Motion score selected for threshold calibration, preferring motion_delta_focus_score when available. |
| motion_calibration_score_name | protocol | none | true | false | false | Name of the metric used as the primary motion calibration score. |
| visual_brightness_min | metric | none | true | false | false | Minimum mean brightness threshold used by formal visual quality gate. |
| visual_brightness_max | metric | none | true | false | false | Maximum mean brightness threshold used by formal visual quality gate. |
| visual_contrast_min | metric | none | true | false | false | Minimum mean contrast threshold used by formal visual quality gate. |
| visual_extreme_pixel_ratio_max | metric | none | true | false | false | Maximum near-black or near-white pixel ratio used by formal visual quality gate. |
| motion_delta_threshold | metric | none | true | false | false | Minimum adjacent-frame delta threshold used by formal motion consistency gate. |
| temporal_flicker_threshold | metric | none | true | false | false | Maximum temporal flicker threshold used by formal motion consistency gate. |
| visual_quality_metric_status | governance | none | true | false | false | Status of formal video file visual quality metric. |
| motion_consistency_metric_status | governance | none | true | false | false | Status of formal video file motion consistency metric. |
| visual_quality_failure_reason | governance | none | true | false | false | Explicit visual quality gate failure reason for generated video file metrics. |
| motion_consistency_failure_reason | governance | none | true | false | false | Explicit motion consistency gate failure reason for generated video file metrics. |
| formal_visual_quality_ready | governance | none | true | false | false | Whether formal video file visual quality metric is ready. |
| formal_motion_consistency_ready | governance | none | true | false | false | Whether formal video file motion consistency metric is ready. |
| motion_claim_role | protocol | none | true | false | false | Role-aware motion claim category used by formal motion gate. |
| formal_motion_gate_policy | governance | none | true | false | false | Role-aware policy used to interpret file-level motion metric. |
| formal_motion_gate_failure_reason | governance | none | true | false | false | Role-aware failure reason emitted by formal motion gate. |
| low_motion_expected_for_role | governance | none | true | false | false | Whether low motion is expected for the sample role and should not block boundary evidence. |
| semantic_model_id | metric | none | true | false | false | CLIP or VLM model identifier used for formal semantic consistency metric. |
| semantic_prompt_source | artifact | none | true | false | false | Prompt suite path or source status used by formal semantic metric. |
| semantic_consistency_mean_score | metric | none | true | false | false | Mean CLIP text-video similarity across sampled frames. |
| semantic_consistency_max_score | metric | none | true | false | false | Maximum CLIP text-video similarity across sampled frames. |
| semantic_consistency_threshold | metric | none | true | false | false | Threshold used to decide formal semantic consistency readiness. |
| semantic_sampled_frame_count | metric | none | true | false | false | Number of video frames sampled by formal semantic metric. |
| semantic_frame_limit | protocol | none | true | false | false | Maximum number of frames requested for formal semantic metric. |
| semantic_metric_device | governance | none | true | false | false | Device used by formal semantic metric inference. |
| semantic_metric_failure_reason | governance | none | true | false | false | Failure reason for semantic consistency metric. |
| formal_semantic_consistency_ready | governance | none | true | false | false | Whether formal semantic consistency metric is ready. |
| formal_metric_result_used_for_claim | claim | none | true | true | false | Whether formal quality motion semantic metrics are used for a positive claim. |
| formal_metric_blocking_reason | governance | none | true | false | false | Per-record blocking reason for formal quality motion semantic claim gate. |
| formal_metric_record_count | metric | none | true | false | false | Number of formal quality motion semantic metric records. |
| formal_visual_quality_ready_count | metric | none | true | false | false | Count of records with ready formal visual quality metrics. |
| formal_motion_consistency_ready_count | metric | none | true | false | false | Count of records with ready formal motion consistency metrics. |
| formal_semantic_consistency_ready_count | metric | none | true | false | false | Count of records with ready formal semantic consistency metrics. |
| formal_visual_quality_blocked_count | metric | none | true | false | false | Count of records blocked by formal visual quality metrics. |
| formal_motion_consistency_blocked_count | metric | none | true | false | false | Count of records blocked by formal motion consistency metrics. |
| formal_semantic_consistency_blocked_count | metric | none | true | false | false | Count of records blocked by formal semantic consistency metrics. |
| formal_visual_motion_ready | governance | none | true | false | false | Whether formal visual and motion metrics are ready for all records. |
| formal_semantic_ready | governance | none | true | false | false | Whether formal semantic metrics are ready for all records. |
| formal_quality_motion_semantic_ready | governance | none | true | false | false | Whether all formal quality, motion, and semantic metrics are ready. |
| formal_metric_claim_status | claim | none | true | true | false | Claim readiness status for formal quality motion semantic metrics. |
| gpu_name | governance | none | true | false | false | GPU name captured by Colab runtime. |
| gpu_memory_mb | governance | none | true | false | false | GPU memory in MB captured by Colab runtime. |
| sampling_constraint_enabled | protocol | none | true | false | false | Whether sampling-time weak constraint is enabled. |
| sampling_constraint_config_id | protocol | none | true | false | false | Sampling-time weak constraint configuration identifier. |
| constraint_projection_operator_id | protocol | none | true | false | false | Projection operator used by sampling-time constraint. |
| constraint_key_id | protocol | none | true | false | false | Key identifier used by sampling-time constraint. |
| constraint_payload_code_id | protocol | none | true | false | false | Payload code identifier used by sampling-time constraint. |
| constraint_tubelet_selector_id | protocol | none | true | false | false | Tubelet selector used by sampling-time constraint. |
| constraint_admissibility_enabled | protocol | none | true | false | false | Whether admissibility filtering is enabled for constraint variant. |
| constraint_key_condition_enabled | protocol | none | true | false | false | Whether key conditioning is enabled for constraint variant. |
| lambda_schedule_id | protocol | none | true | false | false | Lambda schedule identifier for sampling-time constraint. |
| lambda_max | protocol | none | true | false | false | Maximum lambda value for sampling-time constraint. |
| lambda_time_window | protocol | none | true | false | false | Normalized time window where sampling-time constraint is active. |
| lambda_values | protocol | none | true | false | false | Bounded lambda values used by preflight sampling-time constraint. |
| constraint_apply_steps | metric | none | true | false | false | Number of sampling steps where constraint is active. |
| constraint_norm_budget | protocol | none | true | false | false | Norm budget for weak velocity projection. |
| constraint_runtime_overhead_sec | metric | none | true | false | false | Runtime overhead proxy for sampling-time constraint. |
| S_trajectory_observation_before_constraint | metric | none | true | false | false | Trajectory observation score before sampling-time constraint. |
| S_trajectory_observation_after_constraint | metric | none | true | false | false | Trajectory observation score after sampling-time constraint. |
| trajectory_constraint_gain | metric | none | true | false | false | Trajectory observation gain caused by sampling-time constraint. |
| attacked_positive_TPR_before_constraint | metric | none | true | false | false | Attacked positive TPR before sampling-time constraint. |
| attacked_positive_TPR_after_constraint | metric | none | true | false | false | Attacked positive TPR after sampling-time constraint. |
| attacked_negative_FPR_before_constraint | metric | none | true | false | false | Attacked negative FPR before sampling-time constraint. |
| attacked_negative_FPR_after_constraint | metric | none | true | false | false | Attacked negative FPR after sampling-time constraint. |
| quality_delta_after_constraint | metric | none | true | false | false | Visual quality delta after sampling-time constraint. |
| motion_delta_after_constraint | metric | none | true | false | false | Motion consistency delta after sampling-time constraint. |
| semantic_delta_after_constraint | metric | none | true | false | false | Semantic consistency delta after sampling-time constraint. |
| constraint_quality_status | governance | none | true | false | false | Quality gate status for sampling-time constraint. |
| constraint_motion_status | governance | none | true | false | false | Motion gate status for sampling-time constraint. |
| constraint_semantic_status | governance | none | true | false | false | Semantic gate status for sampling-time constraint. |
| constraint_main_claim_status | claim | none | true | true | false | Claim scope status for sampling-time constraint evidence. |
| constraint_threshold_value | metric | none | true | false | false | Decision threshold used by sampling-time constraint preflight. |
| sampling_time_constraint_preflight_decision | governance | none | true | false | false | B6 sampling-time constraint preflight decision. |
| trajectory_constraint_gain_mean | metric | none | true | false | false | Mean trajectory constraint gain across preflight records. |
| trajectory_constraint_gain_over_unconstrained | metric | none | true | false | false | Constraint gain compared with unconstrained trajectory baseline. |
| attacked_positive_tpr_gain | metric | none | true | false | false | TPR gain after sampling-time constraint. |
| quality_motion_semantic_constraint_gate | governance | none | true | false | false | Joint quality, motion, and semantic gate for sampling-time constraint. |
| lambda_schedule_ablation_supports_mid_stage | governance | none | true | false | false | Whether lambda schedule ablation supports mid-stage constraint. |
| strong_lambda_quality_block_detected | governance | none | true | false | false | Whether strong lambda ablation is blocked by quality or consistency gates. |
| submission_claim_policy | governance | none | true | false | false | Policy describing whether preflight records support final submission claims. |
| constraint_trace_id | protocol | none | true | false | false | Trace identifier for sampling-time constraint callback records. |
| constraint_apply_status | governance | none | true | false | false | Whether sampling-time constraint was applied at a callback step. |
| constraint_apply_reason | governance | none | true | false | false | Reason describing why sampling-time constraint was or was not applied. |
| lambda_value | metric | none | true | false | false | Lambda value used at one sampling callback step. |
| latent_alignment_before_constraint | metric | none | true | false | false | Latent alignment with constraint direction before callback update. |
| latent_alignment_after_constraint | metric | none | true | false | false | Latent alignment with constraint direction after callback update. |
| latent_alignment_gain | metric | none | true | false | false | Alignment gain caused by one sampling-time callback update. |
| latent_norm_before_constraint | metric | none | true | false | false | Latent norm before sampling-time callback update. |
| latent_norm_after_constraint | metric | none | true | false | false | Latent norm after sampling-time callback update. |
| latent_constraint_delta_norm | metric | none | true | false | false | Norm of the explicit sampling-time constraint delta vector applied in callback. |
| flow_matching_backbone_claim_status | claim | none | true | true | false | Claim boundary status for Wan2.1 Flow Matching backbone evidence. |
| flow_velocity_proxy_available | governance | none | true | false | false | Whether adjacent callback latent displacement is available as a flow velocity proxy. |
| flow_velocity_proxy_source | protocol | none | true | false | false | Source used to derive flow velocity proxy records. |
| flow_velocity_proxy_norm_before_constraint | metric | none | true | false | false | Flow velocity proxy norm before sampling-time callback update. |
| flow_velocity_proxy_norm_after_constraint | metric | none | true | false | false | Flow velocity proxy norm after sampling-time callback update. |
| flow_velocity_alignment_before_constraint | metric | none | true | false | false | Alignment between flow velocity proxy and keyed constraint direction before callback update. |
| flow_velocity_alignment_after_constraint | metric | none | true | false | false | Alignment between flow velocity proxy and keyed constraint direction after callback update. |
| flow_velocity_alignment_gain | metric | none | true | false | false | Alignment gain showing whether velocity / flow trajectory participates in watermark synchronization. |
| flow_velocity_alignment_gain_mean | metric | none | true | false | false | Mean flow velocity alignment gain for one method variant. |
| flow_velocity_proxy_record_count | metric | none | true | false | false | Number of constraint records with available flow velocity proxy. |
| keyed_flow_velocity_alignment_gain_mean | metric | none | true | false | false | Mean flow velocity alignment gain for keyed SSTW-TC constraint records. |
| baseline_flow_velocity_alignment_gain_mean | metric | none | true | false | false | Mean flow velocity alignment gain for the unconstrained baseline. |
| flow_velocity_gain_over_unconstrained | metric | none | true | false | false | Flow velocity alignment gain of keyed constraint over the unconstrained baseline. |
| flow_velocity_proxy_ready | governance | none | true | false | false | Whether flow velocity proxy evidence is ready for mechanism audit. |
| primary_flow_matching_model_ready | governance | none | true | false | false | Whether the checked run uses Wan2.1 as the primary Flow Matching backbone. |
| constraint_variant_summary_records | artifact | none | true | false | false | Aggregated B6 Colab constraint variant summary records. |
| keyed_constraint_alignment_gain_mean | metric | none | true | false | false | Mean alignment gain for keyed sampling constraint variant. |
| baseline_alignment_gain_mean | metric | none | true | false | false | Mean alignment gain for unconstrained trajectory baseline. |
| sampling_time_constraint_colab_probe | governance | none | true | false | false | Stage identifier for B6 real sampling callback probe. |
| sampling_time_constraint_colab_postprocess | governance | none | true | false | false | Stage identifier for B6 Colab postprocess artifacts. |

| evidence_stage_id | governance | none | true | false | false | Stage identifier summarized by submission freeze preparation evidence records. |
| evidence_decision | governance | none | true | false | false | PASS or FAIL decision summarized for one evidence stage. |
| supporting_artifact_paths | artifact | none | true | false | false | Governed artifact paths supporting a claim or evidence record. |
| evidence_details | governance | none | true | false | false | Bounded details copied from stage evidence for submission preparation. |
| claim_text | claim | none | true | true | false | Human-readable claim text audited against governed artifacts. |
| claim_scope | claim | none | true | true | false | Claim scope such as main or exploratory. |
| claim_status | claim | none | true | true | false | Claim audit status such as supported, unsupported, or needs_downgrade. |
| downgrade_reason | claim | none | true | true | false | Reason a claim is downgraded instead of used as a main claim. |
| supporting_stage_ids | claim | none | true | true | false | Evidence stage identifiers supporting a claim audit record. |
| supported_by_governed_artifacts | claim | none | true | true | false | Whether a claim maps to governed artifacts. |
| claim_audit_record_count | metric | none | true | false | false | Number of claim audit records in submission freeze preparation. |
| supported_claim_count | metric | none | true | false | false | Number of supported claims in submission freeze preparation. |
| needs_downgrade_claim_count | metric | none | true | false | false | Number of downgraded claims in submission freeze preparation. |
| unsupported_claim_count | metric | none | true | false | false | Number of unsupported claims in submission freeze preparation. |
| sstw_t_submission_preparation_status | governance | none | true | false | false | Whether SSTW-T can enter submission preparation. |
| sstw_tc_submission_freeze_status | governance | none | true | false | false | Whether SSTW-TC can be used as a final submission-freeze claim. |
| claim_boundary_status | governance | none | true | false | false | Whether claim boundary and downgrade policy pass. |
| release_package_rebuildable | governance | none | true | false | false | Release package rebuildability status for submission freeze preparation. |

| package_dir | artifact | none | true | false | false | Directory where a governed package is written. |
| package_file_count | metric | none | true | false | false | Number of files included in a governed package. |
| package_size_bytes | metric | none | true | false | false | Size of a governed package archive in bytes. |
| included_subdirs | artifact | none | true | false | false | Governed subdirectories included in a package. |
| excluded_asset_policy | governance | none | true | false | false | Policy describing assets intentionally excluded from a package. |
| file_records | artifact | none | true | false | false | File-level package manifest records. |
| relative_path | artifact | none | true | false | false | File path relative to the packaged run root. |
| archive_name | artifact | none | true | false | false | File path inside a package archive. |
| size_bytes | metric | none | true | false | false | File size in bytes recorded by package manifest. |
| sha256 | artifact | none | true | false | false | File sha256 digest recorded by package manifest. |

| submission_readiness_decision | governance | none | true | false | false | Overall submission readiness decision derived from claim audit records. |
| main_submission_variant | governance | none | true | false | false | Method variant currently eligible for main submission narrative. |
| exploratory_variants | governance | none | true | false | false | Method variants allowed only as exploratory or appendix content. |
| main_text_ready_claim_count | metric | none | true | false | false | Number of supported claims ready for main text. |
| exploratory_ready_claim_count | metric | none | true | false | false | Number of supported exploratory claims. |
| downgraded_claim_count | metric | none | true | false | false | Number of claims downgraded from main claim status. |
| blocked_claim_count | metric | none | true | false | false | Number of blocked claims in readiness summary. |
| package_ready | governance | none | true | false | false | Whether package digest and package artifacts are available. |
| remaining_submission_tasks | governance | none | true | false | false | Remaining governed tasks before paper submission. |
| claim_boundary_statement | claim | none | true | true | false | Reader-facing statement of allowed claim boundary. |
| readiness_bucket | governance | none | true | false | false | Claim readiness bucket used in submission readiness claim table. |

| table_id | artifact | none | true | false | false | Semantic identifier of a rebuilt submission table. |
| stage_label | governance | none | true | false | false | Human-readable stage label used in submission tables. |
| primary_metrics | metric | none | true | false | false | Bounded key metrics serialized from governed stage evidence details. |
| output_tables | artifact | none | true | false | false | Table paths generated by a table manifest. |
| stage_evidence_row_count | metric | none | true | false | false | Number of rows in the stage evidence main table. |
| main_claim_row_count | metric | none | true | false | false | Number of rows in the main claim table. |
| exploratory_boundary_row_count | metric | none | true | false | false | Number of rows in the exploratory boundary table. |
| table_rebuild_status | governance | none | true | false | false | Whether submission main tables were rebuilt from governed records. |
| main_tables_rebuild_status | governance | none | true | false | false | Main tables rebuild status copied into submission preparation decision. |
| allowed_paper_location | governance | none | true | false | false | Allowed paper location for an exploratory or downgraded claim. |

| negative_family | protocol | none | true | false | false | Negative sample family or calibration tail family used for fixed-FPR and replay controls. |
| sampler_signature_placeholder | placeholder | _placeholder | true | false | true | Placeholder for governed sampler signature hash before preflight validates stable sampler metadata capture. |
| trajectory_source_level | protocol | none | true | false | false | Evidence level for trajectory source, such as synthetic proxy, callback latent trace, replay trace, or unavailable. |
| S_path_inv | metric | none | true | false | false | Time reparameterization invariant path evidence score derived from trajectory records. |
| S_velocity | metric | none | true | false | false | Velocity or latent displacement trajectory evidence score. |
| S_final_conservative | metric | none | true | false | false | Conservative final score combining endpoint, path, and velocity evidence without allowing a single evidence layer to dominate. |
| path_marginal_gain_at_fixed_fpr | metric | none | true | false | false | Marginal gain from path evidence measured at fixed false positive rate. |
| replay_uncertainty_mean | metric | none | true | false | false | Mean replay uncertainty used to down-weight or block uncertain reconstructed trajectories. |
| flow_state_admissibility_status | governance | none | true | false | false | Status showing whether flow trajectory state evidence passed admissibility constraints. |
| claim_support_status | claim | none | true | true | false | Claim support boundary status mapped to governed records, tables, figures, reports, or manifests. |
| formal_motion_claim_status | claim | none | true | true | false | Claim readiness status derived from formal motion-consistency records for positive motion samples. |
| motion_claim_eligible_generation_count | metric | none | true | false | false | Number of successful generation records that passed formal visual, motion, and semantic gates for positive motion claim use. |
| motion_claim_excluded_generation_count | metric | none | true | false | false | Number of successful generation records excluded from motion or trajectory claim use by formal metric gates. |
| motion_claim_runtime_attack_ready_count | metric | none | true | false | false | Number of runtime attack records counted after formal motion claim eligibility filtering. |
| motion_claim_runtime_detection_ready_count | metric | none | true | false | false | Number of runtime detection records counted after formal motion claim eligibility filtering. |
| pilot_gate_decision | governance | none | true | false | false | Small-scale claim pilot gate decision. |
| missing_pilot_requirements | governance | none | true | false | false | List of missing requirements blocking small-scale claim pilot progression. |
| pilot_missing_requirement_count | metric | none | true | false | false | Count of missing small-scale claim pilot requirements. |
| pilot_matrix_record_count | metric | none | true | false | false | Number of small-scale claim pilot matrix proxy records. |
| pilot_matrix_attack_count | metric | none | true | false | false | Number of attacks covered by small-scale claim pilot matrix proxy records. |
| pilot_matrix_method_variant_count | metric | none | true | false | false | Number of method variants covered by small-scale claim pilot matrix proxy records. |
| pilot_matrix_negative_family_count | metric | none | true | false | false | Number of negative families covered by small-scale claim pilot matrix proxy records. |
| pilot_matrix_postprocess_decision | governance | none | true | false | false | Postprocess decision for small-scale claim pilot matrix proxy records. |
| pilot_evidence_level | governance | none | true | false | false | Evidence level of pilot records, such as proxy postprocess or runtime attack. |
| attack_matrix_evidence_level | governance | none | true | false | false | Evidence level used to support attack matrix coverage. |
| negative_family_evidence_level | governance | none | true | false | false | Evidence level used to support negative family coverage. |
| prompt_count | metric | none | true | false | false | Number of unique successful prompts observed by a pilot gate. |
| seed_per_prompt_min | metric | none | true | false | false | Minimum number of successful seeds per prompt. |
| attack_count | metric | none | true | false | false | Number of distinct non-no-op attacks observed by a pilot gate. |
| negative_family_count | metric | none | true | false | false | Number of distinct negative families observed by a pilot gate. |
| method_variant_count | metric | none | true | false | false | Number of distinct method variants observed by a pilot gate. |
| wrong_key_score_separation_passed | metric | none | true | false | false | Whether wrong-key controls are separated from the matched-key trajectory score. |
| wrong_sampler_replay_control_not_equivalent | metric | none | true | false | false | Whether wrong-sampler replay control cannot forge the matched sampler trajectory. |
| wrong_key_status | governance | none | true | false | false | Status describing wrong-key separation evidence for a pilot or detection record. |
| wrong_sampler_replay_status | governance | none | true | false | false | Status describing whether wrong-sampler replay was rejected or not applicable. |
| pilot_paper_gate_decision | governance | none | true | false | false | pilot_paper fixed-FPR gate 的 PASS / FAIL 判定。 |
| missing_pilot_paper_requirements | governance | none | true | false | false | 阻断 pilot_paper fixed-FPR claim 的缺失条件列表。 |
| pilot_paper_missing_requirement_count | metric | none | true | false | false | pilot_paper fixed-FPR gate 的缺失条件数量。 |
| pilot_paper_hard_required_config_missing | governance | none | true | false | false | pilot_paper 不可通过配置关闭的 validation_scale 与公平比较硬前置缺口列表。 |
| pilot_paper_hard_required_config_missing_count | metric | none | true | false | false | pilot_paper 硬前置配置缺口数量。 |
| paper_result_level | governance | none | true | true | false | 结果包的论文级别, 例如 pilot_paper 或 full_paper。 |
| paper_protocol_level | governance | none | true | true | false | 结果包采用的协议级别, 例如 paper_grade_protocol。 |
| paper_protocol_difference_from_full_paper | governance | none | true | true | false | 当前结果级别与 full_paper 的协议差异说明。 |
| pilot_paper_protocol_matches_full_paper | governance | none | true | true | false | pilot_paper 是否与 full_paper 使用同构协议。 |
| pilot_paper_claim_allowed | governance | none | true | true | false | 当前结果是否允许报告 pilot_paper 级论文主张。 |
| threshold_protocol | protocol | none | true | false | false | 固定 FPR 阈值构造协议, 例如 calibration split 到 frozen threshold 再到 held-out test split。 |
| blocked_target_fpr | protocol | none | true | false | false | 当前阶段明确禁止报告的更低 FPR 目标。 |
| fpr_threshold_value | protocol | none | true | false | false | calibration negative split 冻结得到的 FPR 阈值。 |
| calibration_negative_fpr_at_threshold | metric | none | true | false | false | calibration negative split 在冻结阈值下的观测 FPR。 |
| calibration_negative_false_positive_count_at_threshold | metric | none | true | false | false | calibration negative split 在冻结阈值下的 false positive 数量。 |
| heldout_negative_fpr_at_threshold | metric | none | true | false | false | held-out test negative split 在冻结阈值下的观测 FPR。 |
| heldout_negative_false_positive_count_at_threshold | metric | none | true | false | false | held-out test negative split 在冻结阈值下的 false positive 数量。 |
| observed_negative_fpr_at_threshold | metric | none | true | false | false | 对外摘要使用的 negative FPR, 在 pilot_paper 中等于 held-out test negative FPR。 |
| tpr_at_target_fpr | metric | none | true | false | false | 当前 protocol config 指定 target_fpr 下的 held-out attacked positive TPR。 |
| tpr_at_fpr_01 | metric | none | true | false | false | 冻结 FPR=0.01 阈值下的 held-out attacked positive TPR。 |
| true_positive_count_at_threshold | metric | none | true | false | false | held-out attacked positive split 在冻结阈值下的 true positive 数量。 |
| target_fpr_claim_allowed | governance | none | true | false | false | 当前 protocol config 指定 target_fpr 下的 claim 是否允许报告。 |
| tpr_at_fpr_01_pilot_claim_allowed | governance | none | true | false | false | 当前 pilot_paper 是否允许报告 pilot_paper 级 TPR@FPR=0.01 结论。 |
| blocked_target_fpr_claim_allowed | governance | none | true | false | false | 当前阶段对 blocked_target_fpr 级 claim 是否允许报告, pilot_paper 中必须为 false。 |
| tpr_at_fpr_001_claim_allowed | governance | none | true | false | false | 当前结果是否允许报告 TPR@FPR=0.001 结论。 |
| generation_record_count | metric | none | true | false | false | 当前 run_root 中读取到的 generation record 总数。 |
| pilot_paper_generation_record_count | metric | none | true | false | false | 属于 pilot_paper profile 的 generation record 数量。 |
| pilot_paper_motion_claim_eligible_generation_count | metric | none | true | false | false | 通过 formal motion claim 过滤的 pilot_paper generation record 数量。 |
| pilot_paper_prompt_count | metric | none | true | false | false | pilot_paper 中 motion-eligible prompt 数量。 |
| pilot_paper_seed_per_prompt_min | metric | none | true | false | false | pilot_paper 中每个 prompt 的最小 seed 覆盖数。 |
| pilot_paper_calibration_seed_per_prompt_min | metric | none | true | false | false | pilot_paper calibration split 中每个 prompt 的最小 seed 覆盖数。 |
| pilot_paper_test_seed_per_prompt_min | metric | none | true | false | false | pilot_paper test split 中每个 prompt 的最小 seed 覆盖数。 |
| pilot_paper_unique_video_count | metric | none | true | false | false | pilot_paper motion-eligible unique video 数量。 |
| pilot_paper_calibration_unique_video_count | metric | none | true | false | false | pilot_paper calibration split unique video 数量。 |
| pilot_paper_test_unique_video_count | metric | none | true | false | false | pilot_paper held-out test split unique video 数量。 |
| calibration_negative_event_count | metric | none | true | false | false | calibration split 中用于冻结阈值的 negative event 数量。 |
| heldout_test_negative_event_count | metric | none | true | false | false | held-out test split 中用于报告 FPR 的 negative event 数量。 |
| heldout_attacked_positive_event_count | metric | none | true | false | false | held-out test split 中用于报告 TPR 的 attacked positive event 数量。 |
| heldout_negative_event_count | metric | none | true | false | false | held-out negative event 数量摘要别名。 |
| attacked_positive_event_count | metric | none | true | false | false | attacked positive event 数量摘要别名。 |
| calibration_negative_family_count | metric | none | true | false | false | calibration split 中 negative family 覆盖数量。 |
| heldout_negative_family_count | metric | none | true | false | false | held-out test split 中 negative family 覆盖数量。 |
| calibration_negative_event_count_per_family_min | metric | none | true | false | false | calibration split 中每个 negative family 的最小事件数。 |
| heldout_negative_event_count_per_family_min | metric | none | true | false | false | held-out test split 中每个 negative family 的最小事件数。 |
| negative_event_count_per_family_min | metric | none | true | false | false | negative family 最小事件数摘要别名。 |
| calibration_negative_family_event_counts | metric | none | true | false | false | calibration split 中各 negative family 的事件数映射。 |
| heldout_negative_family_event_counts | metric | none | true | false | false | held-out test split 中各 negative family 的事件数映射。 |
| attack_event_counts | metric | none | true | false | false | held-out attacked positive split 中各 attack 的事件数映射。 |
| negative_tail_status | governance | none | true | false | false | negative score tail 是否未膨胀的审计状态。 |
| minimum_unique_video_count | protocol | none | true | false | false | gate 要求的最小 unique video 数量。 |
| minimum_calibration_negative_event_count | protocol | none | true | false | false | gate 要求的最小 calibration negative event 数量。 |
| minimum_heldout_test_negative_event_count | protocol | none | true | false | false | gate 要求的最小 held-out test negative event 数量。 |
| minimum_heldout_attacked_positive_event_count | protocol | none | true | false | false | gate 要求的最小 held-out attacked positive event 数量。 |
| minimum_calibration_negative_event_count_per_family | protocol | none | true | false | false | gate 要求的 calibration split 每个 negative family 最小事件数。 |
| minimum_heldout_negative_event_count_per_family | protocol | none | true | false | false | gate 要求的 held-out split 每个 negative family 最小事件数。 |
| minimum_attack_event_count_per_attack | protocol | none | true | false | false | gate 要求的每个 attack 最小 held-out positive event 数量。 |
| require_external_baseline_comparison_ready | protocol | none | true | false | false | pilot_paper gate 是否要求 external_baseline adapter comparison 已完成。 |
| require_modern_external_baseline_formal_results | protocol | none | true | false | false | pilot_paper gate 是否要求现代视频水印 baseline 使用正式 adapter measured_formal 结果。 |
| require_internal_ablation_matrix_ready | protocol | none | true | false | false | pilot_paper gate 是否要求内部消融矩阵已完成。 |
| required_external_baseline_adapter_names | protocol | none | true | false | false | pilot_paper gate 要求出现的 external_baseline adapter 名称列表。 |
| required_modern_external_baseline_adapter_names | protocol | none | true | false | false | pilot_paper gate 要求产出 measured_formal records 的现代视频水印 baseline adapter 名称列表。 |
| required_internal_ablation_variants | protocol | none | true | false | false | pilot_paper gate 要求出现的内部消融 method variant 列表。 |
| minimum_pilot_paper_external_baseline_trace_count | protocol | none | true | false | false | pilot_paper external baseline comparison 要求覆盖的 held-out trace 最小数量。 |
| minimum_pilot_paper_internal_ablation_trace_count | protocol | none | true | false | false | pilot_paper internal ablation 每个必需变体要求覆盖的 held-out trace 最小数量。 |
| minimum_internal_ablation_variant_count | protocol | none | true | false | false | pilot_paper gate 要求的内部消融变体最小数量。 |
| minimum_modern_external_baseline_formal_adapter_count | protocol | none | true | false | false | pilot_paper gate 要求的现代视频水印 measured_formal adapter 最小数量。 |
| pilot_paper_external_baseline_comparison_ready | governance | none | true | true | false | pilot_paper gate 中 external_baseline comparison 是否满足完整协议预演要求。 |
| pilot_paper_internal_ablation_matrix_ready | governance | none | true | true | false | pilot_paper gate 中 internal ablation matrix 是否满足完整协议预演要求。 |
| pilot_paper_external_baseline_trace_count | metric | none | true | true | false | pilot_paper held-out trace 中已有任一 measured external_baseline comparison 的数量。 |
| pilot_paper_external_baseline_trace_count_min | metric | none | true | true | false | pilot_paper held-out trace 中每个必需 external_baseline adapter 的最小覆盖数量。 |
| pilot_paper_external_baseline_trace_counts | metric | none | true | false | false | pilot_paper held-out trace 中各必需 external_baseline adapter 的覆盖数量映射。 |
| pilot_paper_internal_ablation_trace_count_min | metric | none | true | true | false | pilot_paper held-out trace 中每个必需内部消融变体的最小覆盖数量。 |
| pilot_paper_internal_ablation_trace_counts | metric | none | true | false | false | pilot_paper held-out trace 中各内部消融变体的覆盖数量映射。 |
| missing_external_baseline_adapter_names | governance | none | true | true | false | pilot_paper gate 中缺失的 required external_baseline adapter 名称列表。 |
| missing_modern_external_baseline_formal_adapter_names | governance | none | true | true | false | pilot_paper gate 中缺失 measured_formal 结果的现代视频水印 baseline adapter 名称列表。 |
| missing_internal_ablation_variants | governance | none | true | true | false | pilot_paper gate 中缺失的 required internal ablation variant 名称列表。 |
| next_allowed_action | governance | none | true | false | false | 当前 gate 后允许执行的下一步动作。 |
| next_forbidden_action | governance | none | true | false | false | 当前 gate 后明确禁止执行的动作。 |
| pilot_paper_claim_support_status | claim | none | true | true | false | package manifest 中记录的 pilot_paper claim 支撑状态摘要。 |
| validation_scale_claim_support_status | claim | none | true | true | false | package manifest 或 pilot_paper gate 中记录的 validation-scale claim 支撑状态摘要。 |
| pilot_paper_result_level | governance | none | true | true | false | package manifest 中记录的 pilot_paper 结果级别。 |
| pilot_paper_protocol_level | governance | none | true | true | false | package manifest 中记录的 pilot_paper 协议级别。 |
| pilot_paper_protocol_difference_from_full_paper | governance | none | true | true | false | package manifest 中记录的 pilot_paper 与 full_paper 差异。 |
| pilot_paper_missing_external_baseline_adapter_names | governance | none | true | true | false | package manifest 中记录的 pilot_paper gate 缺失 external_baseline adapter 名称列表。 |
| pilot_paper_missing_modern_external_baseline_formal_adapter_names | governance | none | true | true | false | package manifest 中记录的 pilot_paper gate 缺失现代视频水印 formal adapter 名称列表。 |
| pilot_paper_modern_external_baseline_formal_measured_adapter_count | metric | none | true | true | false | package manifest 中记录的 pilot_paper 现代视频水印 measured_formal adapter 数量。 |
| pilot_paper_missing_internal_ablation_variants | governance | none | true | true | false | package manifest 中记录的 pilot_paper gate 缺失内部消融变体列表。 |
| pilot_paper_threshold_protocol | protocol | none | true | false | false | package manifest 中记录的 pilot_paper threshold protocol 摘要。 |
| pilot_paper_threshold_source_split | protocol | none | true | false | false | package manifest 中记录的 pilot_paper 阈值来源 split 摘要。 |
| pilot_paper_test_time_threshold_update_blocked | protocol | none | true | false | false | package manifest 中记录的 pilot_paper test-time 阈值更新阻断状态。 |
| pilot_paper_target_fpr | protocol | none | true | false | false | package manifest 中记录的 pilot_paper protocol config target_fpr 摘要。 |
| pilot_paper_tpr_at_target_fpr | metric | none | true | false | false | package manifest 中记录的当前 target_fpr 下 pilot_paper TPR 摘要。 |
| pilot_paper_target_fpr_claim_allowed | governance | none | true | false | false | package manifest 中记录的当前 target_fpr claim 允许状态。 |
| pilot_paper_blocked_target_fpr | protocol | none | true | false | false | package manifest 中记录的 pilot_paper blocked_target_fpr 摘要。 |
| pilot_paper_blocked_target_fpr_claim_allowed | governance | none | true | false | false | package manifest 中记录的 blocked_target_fpr claim 禁止状态。 |
| pilot_paper_tpr_at_fpr_01 | metric | none | true | false | false | package manifest 中记录的 pilot_paper 级 TPR@FPR=0.01 摘要。 |
| pilot_paper_calibration_negative_fpr_at_threshold | metric | none | true | false | false | package manifest 中记录的 calibration negative FPR 摘要。 |
| pilot_paper_heldout_negative_fpr_at_threshold | metric | none | true | false | false | package manifest 中记录的 held-out negative FPR 摘要。 |
| pilot_paper_observed_negative_fpr_at_threshold | metric | none | true | false | false | package manifest 中记录的 observed negative FPR 摘要。 |
| pilot_paper_calibration_negative_event_count | metric | none | true | false | false | package manifest 中记录的 calibration negative event 数量摘要。 |
| pilot_paper_heldout_test_negative_event_count | metric | none | true | false | false | package manifest 中记录的 held-out test negative event 数量摘要。 |
| pilot_paper_heldout_negative_event_count | metric | none | true | false | false | package manifest 中记录的 held-out negative event 数量摘要别名。 |
| pilot_paper_heldout_attacked_positive_event_count | metric | none | true | false | false | package manifest 中记录的 held-out attacked positive event 数量摘要。 |
| pilot_paper_attacked_positive_event_count | metric | none | true | false | false | package manifest 中记录的 attacked positive event 数量摘要别名。 |
| pilot_paper_tpr_at_fpr_01_pilot_claim_allowed | governance | none | true | false | false | package manifest 中记录的 pilot_paper 级 TPR@FPR=0.01 claim 允许状态。 |
| pilot_paper_tpr_at_fpr_001_claim_allowed | governance | none | true | false | false | package manifest 中记录的 TPR@FPR=0.001 claim 禁止状态。 |
| motion_threshold_id | protocol | none | true | false | false | Identifier for the motion threshold used by formal motion gate. |
| motion_threshold_source_split | protocol | none | true | false | false | Source split or heuristic source used to define the motion threshold. |
| motion_threshold_calibration_required | governance | none | true | false | false | Whether motion threshold calibration remains required before final claim use. |
| sampler_signature_id | protocol | none | true | false | false | Governed sampler signature identifier derived from model and scheduler metadata. |
| sampler_signature_sha256 | artifact | none | true | false | false | SHA256 digest of governed sampler signature metadata. |
| sampler_class_name | protocol | none | true | false | false | Scheduler or sampler class name recorded during adapter preflight. |
| l4_memory_sufficient | governance | none | true | false | false | Whether detected GPU memory satisfies the minimum L4 smoke preflight requirement. |
| adapter_preflight_decision | governance | none | true | false | false | PASS or FAIL decision for Wan2.1 Flow adapter preflight. |
| adapter_preflight_failure_reason | governance | none | true | false | false | Failure reason for Wan2.1 Flow adapter preflight. |
| model_load_status | governance | none | true | false | false | Whether the Wan2.1 pipeline loaded during adapter preflight. |
| callback_latent_capture_status | governance | none | true | false | false | Whether callback latents were captured during adapter preflight. |
| callback_latent_count | metric | none | true | false | false | Number of callback steps with available latents during adapter preflight. |
| time_grid_capture_status | governance | none | true | false | false | Whether the sampler time grid was captured during adapter preflight. |
| time_grid | protocol | none | true | false | false | Bounded sampler timestep grid captured during adapter preflight. |
| sampler_signature_status | governance | none | true | false | false | Whether governed sampler signature metadata was captured. |
| velocity_proxy_status | governance | none | true | false | false | Whether velocity or latent displacement proxy was captured. |
| velocity_proxy_count | metric | none | true | false | false | Number of callback steps with velocity proxy records. |
| runtime_sec | metric | none | true | false | false | Wall-clock runtime in seconds for a governed stage. |

| package_batch_id | artifact | none | true | false | false | Package batch identifier in <utc_time>_<short_commit> format shared by archive and manifest files. |
| package_utc_time | artifact | none | true | false | false | UTC timestamp token used in package file names. |
| package_short_commit | artifact | none | true | false | false | Short Git commit token used in package file names. |

| wrong_key_control_enabled | protocol | none | true | false | false | Whether the sampling constraint record is a wrong-key trajectory control. |
| constraint_application_direction_status | protocol | none | true | false | false | Direction used to perturb latents during sampling-time constraint callback. |
| constraint_evidence_direction_status | protocol | none | true | false | false | Direction used to score keyed evidence after applying or skipping the constraint. |
| key_separation_gain_over_control | metric | none | true | false | false | Keyed path gain over the strongest without-key or wrong-key control. |
| key_separation_flow_velocity_gain_over_control | metric | none | true | false | false | Keyed velocity gain over the strongest without-key or wrong-key control. |
| without_key_alignment_gain_mean | metric | none | true | false | false | Mean alignment gain for the without-key control variant. |
| wrong_key_alignment_gain_mean | metric | none | true | false | false | Mean alignment gain for the wrong-key control variant. |
| without_key_flow_velocity_alignment_gain_mean | metric | none | true | false | false | Mean flow velocity alignment gain for the without-key control variant. |
| wrong_key_flow_velocity_alignment_gain_mean | metric | none | true | false | false | Mean flow velocity alignment gain for the wrong-key control variant. |
| application_evidence_direction_cosine | metric | none | true | false | false | Cosine similarity between the constraint application direction and the matched-key evidence direction. |
| latent_norm_change | metric | none | true | false | false | Signed change in latent norm after applying the sampling-time constraint. |
| minimum_key_separation_gain | metric | none | true | false | false | Minimum required keyed path gain over the strongest without-key or wrong-key control. |
| minimum_key_separation_flow_velocity_gain | metric | none | true | false | false | Minimum required keyed velocity gain over the strongest without-key or wrong-key control. |

| runtime_detection_decision | governance | none | true | false | false | Runtime attacked video detection runner decision. |
| runtime_detection_record_count | metric | none | true | false | false | Number of runtime detection records. |
| runtime_detection_ready_count | metric | none | true | false | false | Number of runtime detection records that decoded and scored attacked videos. |
| runtime_detection_detectable_count | metric | none | true | false | false | Number of runtime detection records with detectable attacked video proxy evidence. |
| runtime_detection_attack_count | metric | none | true | false | false | Number of distinct attacks covered by runtime detection records. |
| runtime_detection_score_mean | metric | none | true | false | false | Mean runtime attacked video detection proxy score. |
| runtime_detection_evidence_level | governance | none | true | false | false | Evidence level for runtime attacked video detection records. |
| runtime_detection_status | governance | none | true | false | false | Per-record runtime detection status. |
| runtime_detection_failure_reason | governance | none | true | false | false | Per-record runtime detection failure reason. |
| attacked_video_decode_status | governance | none | true | false | false | Whether attacked video file can be decoded during runtime detection. |
| attacked_video_decode_failure_reason | governance | none | true | false | false | Decode failure reason for attacked video runtime detection. |
| attacked_video_detectable | metric | none | true | false | false | Whether attacked video produced a positive runtime detection proxy. |
| attacked_video_decoded_frame_count | metric | none | true | false | false | Number of decoded frames observed by runtime detection runner. |
| source_to_attack_frame_ratio | metric | none | true | false | false | Ratio between attacked frame count and source frame count. |
| decoded_to_source_frame_ratio | metric | none | true | false | false | Ratio between decoded attacked frame count and source frame count. |
| attack_score_delta | metric | none | true | false | false | Runtime attacked video score delta against trajectory observation proxy. |
| S_runtime_attack_detection | metric | none | true | false | false | Runtime attacked video detection proxy score. |

| motion_threshold_calibration_decision | governance | none | true | false | false | Decision for formal motion threshold calibration stage. |
| motion_threshold_calibration_ready | governance | none | true | false | false | Whether calibrated motion threshold is ready for claim gating. |
| target_static_fpr | protocol | none | true | false | false | Target false positive rate for static negative motion tail calibration. |
| estimated_static_fpr | metric | none | true | false | false | Estimated false positive rate on calibration negative static tail. |
| negative_static_calibration_count | metric | none | true | false | false | Number of negative static calibration records used for motion threshold calibration. |
| positive_motion_calibration_count | metric | none | true | false | false | Number of positive motion calibration records used for threshold sanity check. |
| usable_motion_calibration_record_count | metric | none | true | false | false | Number of usable formal motion records available to calibration. |
| motion_calibration_record_count | metric | none | true | false | false | Total number of motion calibration records emitted. |
| negative_static_motion_delta_max | metric | none | true | false | false | Maximum motion delta score among negative static calibration records. |
| negative_static_motion_delta_mean | metric | none | true | false | false | Mean motion delta score among negative static calibration records. |
| conservative_motion_delta_threshold | metric | none | true | false | false | Conservative max-negative-tail motion threshold retained for audit comparison. |
| motion_threshold_selection_strategy | protocol | none | true | false | false | Strategy used to select the primary motion threshold. |
| estimated_static_fpr_including_contaminated | metric | none | true | false | false | Estimated static FPR before excluding suspected negative static contamination. |
| negative_static_contamination_status | governance | none | true | false | false | Whether negative static calibration tail contains suspected high-motion contamination. |
| negative_static_contamination_rule | protocol | none | true | false | false | Rule used to identify suspected high-motion negative static contamination. |
| negative_static_contamination_cutoff | metric | none | true | false | false | Motion delta cutoff used for suspected negative static contamination. |
| negative_static_contamination_count | metric | none | true | false | false | Count of suspected contaminated negative static calibration records. |
| negative_static_contaminated_prompt_count | metric | none | true | false | false | Count of prompts contributing suspected contaminated negative static records. |
| negative_static_clean_calibration_count | metric | none | true | false | false | Count of negative static calibration records after excluding suspected contamination. |
| negative_static_clean_motion_delta_max | metric | none | true | false | false | Maximum motion delta among clean negative static calibration records. |
| negative_static_contaminated_motion_delta_min | metric | none | true | false | false | Minimum motion delta among suspected contaminated negative static records. |
| positive_motion_delta_min | metric | none | true | false | false | Minimum motion delta score among positive motion calibration records. |
| positive_motion_delta_mean | metric | none | true | false | false | Mean motion delta score among positive motion calibration records. |
| positive_motion_pass_rate_at_threshold | metric | none | true | false | false | Positive motion pass rate under the calibrated or fallback motion threshold. |
| minimum_positive_motion_pass_rate_at_threshold | protocol | none | true | false | false | Motion calibration 中 positive_motion 在冻结阈值下的最低通过率要求。 |
| positive_negative_motion_delta_margin | metric | none | true | false | false | positive_motion 最小 motion_delta 与 clean negative_static 最大 motion_delta 的间隔。 |
| motion_threshold_calibration_recommended_action | governance | none | true | false | false | motion calibration 未通过时建议执行的下一步动作。 |
| minimum_negative_static_calibration_count | protocol | none | true | false | false | Minimum required negative static calibration record count. |
| minimum_positive_motion_calibration_count | protocol | none | true | false | false | Minimum required positive motion calibration record count. |
| motion_threshold_calibration_missing_reasons | governance | none | true | false | false | Missing reasons blocking formal motion threshold calibration. |
| motion_calibration_source_split | protocol | none | true | false | false | Source split used by a motion calibration record. |
| motion_calibration_role | protocol | none | true | false | false | Role of a calibration sample, such as negative_static or positive_motion. |
| motion_calibration_record_status | governance | none | true | false | false | Usability status of one motion calibration record. |
| motion_calibration_record_failure_reason | governance | none | true | false | false | Failure reason for a motion calibration record. |
| previous_motion_threshold_id | protocol | none | true | false | false | Previous motion threshold identifier before calibration. |
| previous_motion_delta_threshold | metric | none | true | false | false | Previous heuristic motion delta threshold before calibration. |

| ambiguous_low_motion_calibration_count | metric | none | true | false | false | Number of ambiguous low-motion calibration records used for threshold sanity analysis. |
| ambiguous_low_motion_delta_min | metric | none | true | false | false | Minimum motion delta score among ambiguous low-motion calibration records. |
| ambiguous_low_motion_delta_mean | metric | none | true | false | false | Mean motion delta score among ambiguous low-motion calibration records. |
| minimum_ambiguous_low_motion_calibration_count | protocol | none | true | false | false | Minimum required ambiguous low-motion calibration record count. |
| motion_calibration_design | protocol | none | true | false | false | Design counts for motion threshold calibration prompt and seed split. |
| negative_static_target_video_count | protocol | none | true | false | false | Target negative static video count for motion calibration design. |
| positive_motion_target_video_count | protocol | none | true | false | false | Target positive motion video count for motion calibration design. |
| ambiguous_low_motion_target_video_count | protocol | none | true | false | false | Target ambiguous low-motion video count for motion calibration design. |
| seed_suite_role | protocol | none | true | false | false | Seed role preserved separately from prompt_suite_role in generation plans. |


| prompt_contamination_status | governance | none | true | false | false | Prompt-level contamination status for motion calibration negative prompts. |
| prompt_contamination_reason | governance | none | true | false | false | Reason for prompt-level contamination decision. |
| prompt_contamination_score | metric | none | true | false | false | Motion observability score used for prompt contamination audit, not a final watermark score. |
| contamination_rule_id | protocol | none | true | false | false | Identifier for prompt-level and record-level contamination rules. |
| contamination_decision_source | protocol | none | true | false | false | Source of contamination decision; must be motion_observability_score_only. |
| excluded_from_threshold_estimation | governance | none | true | false | false | Whether a calibration record is excluded from threshold estimation. |
| included_in_contamination_audit | governance | none | true | false | false | Whether a calibration record remains in contamination audit. |
| included_in_stress_negative_eval | governance | none | true | false | false | Whether a contaminated negative sample is retained for stress negative evaluation. |
| final_detection_score_filtering_blocked | governance | none | true | false | false | Whether filtering by final watermark detection score is explicitly blocked. |
| no_final_detection_score_used_for_filtering | governance | none | true | false | false | Whether contamination filtering avoided S_final or final detection scores. |
| positive_motion_pass_rate_wilson_lower | metric | none | true | false | false | Wilson lower confidence bound for positive motion pass rate. |
| minimum_positive_motion_pass_rate_wilson_lower | protocol | none | true | false | false | Minimum Wilson lower confidence bound required for engineering calibration PASS. |
| formal_visual_quality_ready_rate | metric | none | true | false | false | Ready rate of formal visual quality records in motion calibration split. |
| formal_motion_consistency_ready_rate | metric | none | true | false | false | Ready rate of formal motion consistency records in motion calibration split. |
| motion_threshold_evidence_level | governance | none | true | false | false | Evidence level of the motion threshold, such as engineering_calibration. |
| engineering_motion_threshold_calibration_decision | governance | none | true | false | false | Engineering-stage motion calibration decision. |
| paper_fixed_fpr_calibration_decision | governance | none | true | false | false | Paper-stage fixed-FPR calibration decision. |
| paper_fixed_fpr_calibration_ready | governance | none | true | false | false | Whether paper-stage fixed-FPR calibration is ready. |
| target_static_fpr_engineering | protocol | none | true | false | false | Engineering-stage target static FPR used for motion threshold quantile. |
| threshold_quantile | protocol | none | true | false | false | Quantile used to select motion threshold from filtered negative tail. |
| not_final_paper_fpr_0_01 | governance | none | true | false | false | Whether current calibration is not a final paper-level FPR=0.01 claim. |
| prompt_contamination_audit_record_count | metric | none | true | false | false | Number of prompt contamination audit records emitted. |
| threshold_stability_audit | artifact | none | true | false | false | Artifact summarizing threshold stability, bootstrap CI, and prompt dominance. |
| trajectory_sketch_digest_random | random_trace | none | true | false | true | Stable digest for authenticated trajectory sketch verification; suffix marks digest random governance boundary. |
| replay_uncertainty_weight | metric | none | true | false | false | Weight derived from replay uncertainty proxy for validation replay records. |
| replay_scheduler_id | governance | none | true | false | false | Scheduler identifier used by replay or sketch validation records. |
| replay_time_grid_id | governance | none | true | false | false | Time-grid identifier used by replay or sketch validation records. |
| wrong_prompt_replay_control | governance | none | true | false | false | Control label for wrong-prompt replay validation records. |
| replay_and_sketch_gate_decision | governance | none | true | true | false | Decision status for replay and authenticated sketch gate. |
| replay_and_sketch_evidence_level | governance | none | true | true | false | Evidence level for replay and authenticated sketch gate outputs. |
| replay_control_status | governance | none | true | false | false | Replay control acceptance or rejection status. |
| replay_and_sketch_missing_requirements | governance | none | true | false | false | Missing requirement names for replay and authenticated sketch gate. |

| official_runtime_closure_decision | governance | none | true | false | false | Decision for the modern external baseline runtime closure requirements preflight. |
| official_runtime_closure_status | governance | none | true | false | false | Status explaining whether selected modern external baselines are ready to attempt formal reference execution. |
| runtime_closure_ready_count | metric | none | true | false | false | Number of selected modern external baselines with source, requirements, runtime inputs, and one formal execution path available. |
| runtime_closure_blocked_count | metric | none | true | false | false | Number of selected modern external baselines blocked by missing runtime requirements. |
| runtime_closure_ready_baselines | governance | none | true | false | false | Baseline identifiers ready to attempt formal reference execution. |
| runtime_closure_blocked_baselines | governance | none | true | false | false | Baseline identifiers blocked by missing runtime requirements. |
| runtime_closure_missing_requirements | governance | none | true | false | false | Per-baseline list of missing runtime closure requirements. |
| official_runtime_closure_requirements | artifact | none | true | false | false | Artifact describing source, requirements, runtime input, resource, command, and official bundle readiness for modern external baselines. |
| requirements_file_path | artifact | none | true | false | false | Path to a baseline-specific requirements file used by runtime closure preflight. |
| requirements_file_exists | governance | none | true | false | false | Whether a baseline-specific requirements file exists. |
| official_source_ready | governance | none | true | false | false | Whether a baseline official source directory contains required source files. |
| required_resource_ready | governance | none | true | false | false | Whether required baseline resource environment variables or default Drive resource files are available. |
| environment_updates | artifact | none | true | false | false | Environment variable updates emitted by a preflight artifact for the Notebook parent process. |

| external_baseline_self_containment_decision | governance | none | true | true | false | External baseline 是否完成项目内 clone/build/run/adapt/record 闭环的阶段判定。 |
| external_baseline_self_containment_ready_count | metric | none | true | true | false | validation_scale gate 重新审计到的 self-contained modern baseline 数量。 |
| external_baseline_self_containment_required_count | metric | none | true | true | false | validation_scale gate 要求 self-contained 的 modern baseline 数量。 |
| external_baseline_self_containment_gate_missing_requirements | governance | none | true | true | false | validation_scale gate 对 self-containment artifact 复查得到的缺口列表。 |
| self_contained_modern_external_baseline_count | metric | none | true | true | false | 已完成项目内 official bundle 执行闭环的现代 external baseline 数量。 |
| missing_self_contained_modern_external_baseline_names | governance | none | true | false | false | 尚未完成项目内 official bundle 执行闭环的现代 external baseline 名称列表。 |
| missing_self_containment_requirements | governance | none | true | false | false | External baseline self-containment 阶段仍缺失的要求列表。 |
| source_clone_ready | governance | none | true | false | false | External baseline self-containment 行中表示当前 checkout 或 clone manifest 是否已经提供源码克隆证据。 |
| repository_generated_official_bundle_ready | governance | none | true | false | false | External baseline self-containment 行中表示项目内生成的 official bundle 与 execution manifest 是否足以证明 clone/build/run 证据。 |
| missing_repository_generated_official_bundle_modern_external_baseline_names | governance | none | true | false | false | External baseline self-containment 中缺少项目内 official bundle 执行闭环的现代 baseline 名称列表。 |
| official_bundle_record_count | artifact | none | true | false | false | 单个 baseline self-containment 行中绑定的 official bundle record 数量。 |
| official_bundle_record_ok_count | artifact | none | true | false | false | 单个 baseline self-containment 行中通过 provenance 与 execution manifest 校验的 official bundle record 数量。 |
| official_execution_manifest_count | artifact | none | true | false | false | 单个 baseline self-containment 行中绑定的 official execution manifest 数量。 |
| official_execution_manifest_ok_count | artifact | none | true | false | false | 单个 baseline self-containment 行中通过项目内执行闭环校验的 official execution manifest 数量。 |
| materialized_official_bundle_path_count | artifact | none | true | false | false | 单个 baseline self-containment 行中实际落盘可读取的 official bundle 路径数量。 |
| materialized_official_execution_manifest_path_count | artifact | none | true | false | false | 单个 baseline self-containment 行中实际落盘可读取的 official execution manifest 路径数量。 |

| sstw_measured_formal_record_id | protocol | none | true | false | false | SSTW 本方法 measured_formal record 的稳定标识。 |
| sstw_measured_formal_decision | governance | none | true | true | false | SSTW 本方法 measured_formal 转写阶段的门禁判定。 |
| sstw_measured_formal_status | governance | none | true | false | false | 单条 SSTW measured_formal record 的可用状态。 |
| sstw_score | metric | none | true | true | false | SSTW 本方法在同协议攻击样本上的检测分数。 |
| sstw_detected | metric | none | true | true | false | SSTW 本方法在同协议攻击样本上的检测布尔结果。 |
| sstw_clean_negative_score | metric | none | true | true | false | SSTW 本方法 clean negative 校准样本上的检测分数。 |
| sstw_clean_negative_score_semantics | protocol | none | true | false | false | SSTW clean negative 分数语义, 必须与 sstw_score_semantics 对齐。 |
| sstw_score_orientation | protocol | none | true | false | false | SSTW 分数方向, 公平比较阶段只接受 higher_is_more_watermarked。 |
| sstw_detection_score_field | governance | none | true | false | false | SSTW measured_formal 转写所使用的源检测分数字段。 |
| source_runtime_detection_record_index | artifact | none | true | false | false | SSTW measured_formal record 对应的 runtime_detection_records 源记录序号。 |
| source_controlled_negative_record_index | artifact | none | true | false | false | SSTW measured_formal clean negative record 对应的 controlled_negative_records 源记录序号。 |
| clean_negative_unit_id | protocol | none | true | false | false | clean negative 校准单元标识, 用于避免同一 prompt / seed 下多个负样本控制项被误去重。 |
| clean_negative_evidence_level | governance | none | true | false | false | clean negative 分数的证据来源层级。 |
| clean_negative_source_record_family | governance | none | true | false | false | clean negative 分数来自的源 record 文件族。 |
| sstw_measured_formal_record_count | metric | none | true | true | false | SSTW 本方法 measured_formal records 数量。 |
| sstw_measured_formal_ready_count | metric | none | true | false | false | SSTW 本方法 measured_formal ready records 数量。 |
| sstw_measured_formal_positive_record_count | metric | none | true | true | false | SSTW measured_formal 中 attacked positive record 数量。 |
| sstw_measured_formal_clean_negative_record_count | metric | none | true | true | false | SSTW measured_formal 中 clean negative record 数量。 |
| sstw_measured_formal_clean_negative_score_count | metric | none | true | true | false | SSTW measured_formal 中可用于公平校准的 clean negative 分数数量。 |
| sstw_measured_formal_prompt_count | metric | none | true | false | false | SSTW 本方法 measured_formal records 覆盖的 prompt 数量。 |
| sstw_measured_formal_attack_count | metric | none | true | false | false | SSTW 本方法 measured_formal records 覆盖的 attack 数量。 |
| sstw_measured_formal_detected_count | metric | none | true | true | false | SSTW 本方法 measured_formal records 中检测为 positive 的数量。 |
| sstw_measured_formal_detectable_rate | metric | none | true | true | false | SSTW 本方法 measured_formal records 的 positive rate。 |
| sstw_measured_formal_score_mean | metric | none | true | true | false | SSTW 本方法 measured_formal records 的平均检测分数。 |
| sstw_measured_formal_clean_negative_score_mean | metric | none | true | true | false | SSTW measured_formal clean negative 分数均值。 |
| sstw_measured_formal_metric_status | governance | none | true | false | false | SSTW 本方法 measured_formal 转写阶段的 metric_status 汇总。 |
| missing_sstw_measured_formal_requirements | governance | none | true | false | false | SSTW measured_formal 转写阶段仍缺失的要求。 |
| sstw_measured_formal_missing_requirement_count | metric | none | true | false | false | SSTW measured_formal 转写阶段缺失要求数量。 |
| require_sstw_measured_formal_records | protocol | none | true | false | false | validation_scale gate 是否要求 SSTW 本方法 measured_formal records 已落盘。 |
| allow_effect_size_claims | protocol | none | true | false | false | 当前 protocol profile 是否允许效果大小 claim。 |
| formal_method_baseline_comparison_decision | governance | none | true | true | false | SSTW 与现代 external baseline 同协议 measured_formal 比较表阶段判定。 |
| formal_method_baseline_comparison_target_fpr | protocol | none | true | false | false | pilot_paper gate 读取到的 formal method baseline comparison decision 的 target_fpr。 |
| formal_method_baseline_comparison_status | governance | none | true | false | false | pilot_paper gate 汇总的 formal method baseline comparison 产物状态。 |
| formal_comparison_required_method_count | metric | none | true | true | false | 同协议 formal 比较表要求覆盖的方法数量。 |
| formal_comparison_ready_method_count | metric | none | true | true | false | 同协议 formal 比较表已产出 measured_formal 的方法数量。 |
| formal_comparison_modern_baseline_ready_count | metric | none | true | true | false | 同协议 formal 比较表中已产出 measured_formal 的现代 external baseline 数量。 |
| formal_comparison_sstw_ready | governance | none | true | true | false | 同协议 formal 比较表中 SSTW 本方法 measured_formal 是否可用。 |
| formal_comparison_missing_method_ids | governance | none | true | true | false | 同协议 formal 比较表缺失 measured_formal records 的方法 ID 列表。 |
| formal_comparison_missing_method_count | metric | none | true | true | false | 同协议 formal 比较表缺失 measured_formal records 的方法数量。 |
| comparison_score_field | protocol | none | true | false | false | 同协议 formal 比较表聚合时使用的源分数字段。 |
| comparison_prompt_count | metric | none | true | false | false | 同协议 formal 比较行覆盖的 prompt 数量。 |
| comparison_positive_count | metric | none | true | true | false | 同协议 formal 比较行中 positive 检测数量。 |
| comparison_positive_rate | metric | none | true | true | false | 同协议 formal 比较行中 positive 检测比例。 |
| comparison_score_mean | metric | none | true | true | false | 同协议 formal 比较行的平均分数。 |
| comparison_missing_reason | governance | none | true | false | false | 同协议 formal 比较行缺失 measured_formal records 的原因。 |
| require_formal_method_baseline_comparison | protocol | none | true | false | false | validation_scale gate 是否要求 SSTW 与 5 个 baseline 的同协议 measured_formal 比较表。 |
| formal_baseline_difference_interval_decision | governance | none | true | true | false | SSTW 相对现代 external baseline 的差值置信区间阶段判定。 |
| formal_baseline_difference_interval_target_fpr | protocol | none | true | false | false | pilot_paper gate 读取到的 formal baseline difference interval decision 的 target_fpr。 |
| formal_baseline_difference_interval_status | governance | none | true | false | false | pilot_paper gate 汇总的 formal baseline difference interval 产物状态。 |
| reference_method_id | protocol | none | true | false | false | 差值比较中的参考方法 ID。 |
| baseline_method_id | protocol | none | true | false | false | 差值比较中的 baseline 方法 ID。 |
| difference_metric_name | metric | none | true | false | false | 差值置信区间所对应的指标名称。 |
| reference_score_field | protocol | none | true | false | false | 差值统计中参考方法使用的源分数字段。 |
| baseline_score_field | protocol | none | true | false | false | 差值统计中 baseline 使用的源分数字段。 |
| reference_record_count | metric | none | true | false | false | 差值统计中参考方法的 measured_formal 分数数量。 |
| baseline_record_count | metric | none | true | false | false | 差值统计中 baseline 的 measured_formal 分数数量。 |
| paired_comparison_unit_count | metric | none | true | false | false | prompt / seed / attack 锚点完全重合的配对比较单元数量。 |
| reference_score_mean | metric | none | true | true | false | 差值统计中参考方法的平均分数。 |
| baseline_score_mean | metric | none | true | true | false | 差值统计中 baseline 的平均分数。 |
| score_mean_difference | metric | none | true | true | false | 参考方法平均分数减 baseline 平均分数的差值。 |
| difference_ci_confidence_level | metric | none | true | false | false | 差值置信区间的置信水平。 |
| difference_ci_lower | metric | none | true | true | false | 差值置信区间下界。 |
| difference_ci_upper | metric | none | true | true | false | 差值置信区间上界。 |
| difference_interval_method | protocol | none | true | false | false | 差值置信区间使用的统计方法。 |
| difference_interval_status | governance | none | true | false | false | 单条差值置信区间 record 是否可用。 |
| significance_claim_status | governance | none | true | true | false | 当前差值区间是否允许支撑显著性 claim 的状态。 |
| difference_interval_record_count | metric | none | true | true | false | 差值置信区间 records 数量。 |
| difference_interval_ready_count | metric | none | true | true | false | 已可用差值置信区间 records 数量。 |
| difference_interval_missing_baseline_ids | governance | none | true | true | false | 缺失差值置信区间的 baseline 方法 ID 列表。 |
| difference_interval_missing_baseline_count | metric | none | true | true | false | 缺失差值置信区间的 baseline 数量。 |
| require_formal_baseline_difference_interval | protocol | none | true | false | false | validation_scale gate 是否要求 SSTW 相对 5 个 baseline 的差值置信区间报告。 |
| validation_scale_gate_missing_validation_requirements | governance | none | true | false | false | pilot_paper gate 读取到的 validation_scale gate 原始缺失要求列表。 |
| validation_scale_gate_missing_requirement_count | metric | none | true | false | false | pilot_paper gate 读取到的 validation_scale gate 原始缺失要求数量。 |
| validation_scale_gate_fairness_missing_requirements | governance | none | true | false | false | pilot_paper gate 复核 validation_scale 公平比较闭环时发现的缺失要求。 |
| validation_scale_fair_detection_calibration_ready_count | metric | none | true | true | false | validation_scale gate 中已通过 clean negative 公平校准的方法数量。 |
| validation_scale_formal_method_baseline_comparison_ready_count | metric | none | true | true | false | validation_scale gate 中同协议 method-baseline 比较已 ready 的方法数量。 |
| validation_scale_formal_baseline_difference_interval_ready_count | metric | none | true | true | false | validation_scale gate 中 SSTW 相对 baseline 差值区间已 ready 的 baseline 数量。 |
| validation_scale_transition_claim_support_status | governance | none | true | false | false | pilot_paper gate 读取到的 validation_scale -> pilot_paper 跳转 claim_support_status。 |
| validation_scale_transition_source_gate_passed | governance | none | true | false | false | validation_scale -> pilot_paper 跳转记录中的 source gate 是否已通过。 |
| validation_scale_transition_missing_requirements | governance | none | true | false | false | validation_scale -> pilot_paper 跳转记录中的原始缺失要求列表。 |
| validation_scale_transition_missing_requirement_count | metric | none | true | false | false | validation_scale -> pilot_paper 跳转记录中的原始缺失要求数量。 |
| validation_scale_transition_allowed_next_result_profiles | governance | none | true | false | false | validation_scale 通过后允许进入的下一结果 profile 列表。 |
| validation_scale_transition_blocked_next_result_profiles | governance | none | true | false | false | validation_scale 通过后仍禁止跳转的后续结果 profile 列表。 |
| validation_scale_transition_fairness_missing_requirements | governance | none | true | false | false | pilot_paper gate 复核 validation_scale 跳转判定完整性时发现的缺失要求。 |
| validation_scale_formal_internal_ablation_decision | governance | none | true | true | false | validation_scale 级 formal-compatible 内部消融汇总判定。 |
| formal_internal_ablation_evidence_level | governance | none | true | true | false | 内部消融汇总中某个变体的证据层级。 |
| formal_internal_ablation_source_record_family | artifact | none | true | false | false | 内部消融汇总行所使用的源 records 家族。 |
| formal_internal_ablation_record_count | metric | none | true | false | false | 内部消融汇总行聚合的源分数数量。 |
| formal_internal_ablation_score_mean | metric | none | true | true | false | 内部消融汇总行的平均分数。 |
| formal_internal_ablation_full_method_score_mean | metric | none | true | true | false | 内部消融汇总中 full-method measured_formal 平均分数。 |
| formal_internal_ablation_delta_vs_full_method | metric | none | true | true | false | 内部消融变体相对 full-method 平均分数的差值。 |
| formal_internal_ablation_variant_count | metric | none | true | true | false | validation_scale 级内部消融已覆盖变体数量。 |
| formal_internal_ablation_expected_variant_count | metric | none | true | false | false | validation_scale 级内部消融应覆盖变体数量。 |
| formal_internal_ablation_full_method_formal_ready | governance | none | true | true | false | full-method 行是否来自 SSTW measured_formal 结果。 |
| formal_internal_ablation_missing_variants | governance | none | true | true | false | validation_scale 级内部消融缺失的变体列表。 |
| formal_internal_ablation_missing_variant_count | metric | none | true | true | false | validation_scale 级内部消融缺失的变体数量。 |
| require_validation_scale_formal_internal_ablation | protocol | none | true | false | false | validation_scale gate 是否要求 formal-compatible 内部消融汇总。 |
| low_fpr_formal_statistics_decision | governance | none | true | true | false | 低 FPR 正式统计阻断记录阶段判定。 |
| current_target_fpr | protocol | none | true | false | false | 当前 validation_scale profile 使用的 target_fpr。 |
| blocked_result_profile | protocol | none | true | false | false | 当前阻断记录所对应的更高层级结果 profile。 |
| low_fpr_formal_statistics_status | governance | none | true | true | false | 低 FPR 正式统计状态或阻断状态。 |
| formal_low_fpr_claim_allowed | governance | none | true | true | false | 当前阶段是否允许低 FPR 正式 claim。 |
| observed_negative_event_count | metric | none | true | false | false | 当前 run_root 中可用于低 FPR 的 negative event 数量估计。 |
| threshold_protocol_required | protocol | none | true | false | false | 低 FPR 正式统计所需阈值协议。 |
| low_fpr_blocking_reason | governance | none | true | false | false | 低 FPR 正式统计被阻断的原因说明。 |
| low_fpr_formal_statistics_record_count | metric | none | true | true | false | 低 FPR 正式统计阻断 records 数量。 |
| low_fpr_blocked_target_fprs | governance | none | true | true | false | 当前阻断记录覆盖的低 FPR 目标列表。 |
| require_low_fpr_formal_statistics_blocking_record | protocol | none | true | false | false | validation_scale gate 是否要求低 FPR 正式统计阻断记录。 |
| motion_consistency_exclusion_decision | governance | none | true | true | false | motion consistency 阻断样本处理报告阶段判定。 |
| motion_consistency_exclusion_reason | governance | none | true | false | false | 样本被纳入或排除出 motion claim 的具体原因。 |
| excluded_from_motion_claim | governance | none | true | true | false | 样本是否被排除出 motion / trajectory claim。 |
| included_in_motion_claim | governance | none | true | true | false | 样本是否进入 motion / trajectory claim 统计。 |
| excluded_from_effect_size_claim | governance | none | true | true | false | 样本是否被排除出效果大小 claim。 |
| retained_for_audit | governance | none | true | false | false | 被阻断样本是否仍保留为审计记录。 |
| motion_consistency_exclusion_record_count | metric | none | true | true | false | motion consistency 阻断处理 records 数量。 |
| motion_consistency_included_count | metric | none | true | true | false | motion consistency 过滤后纳入 motion claim 的样本数量。 |
| motion_consistency_excluded_count | metric | none | true | true | false | motion consistency 过滤后排除出 motion claim 的样本数量。 |
| motion_consistency_exclusion_reasons | governance | none | true | false | false | motion consistency 处理报告中出现的原因集合。 |
| motion_consistency_claim_filter_applied | governance | none | true | true | false | 是否已经应用 motion consistency claim 过滤。 |
| require_motion_consistency_exclusion_report | protocol | none | true | false | false | validation_scale gate 是否要求 motion consistency 阻断样本处理报告。 |
| raw_detector_score | metric | none | true | false | false | 官方 wrapper 输出的原始水印存在性检测分数, 由 bridge 归一化为 external_baseline_raw_detector_score。 |
| payload_bit_accuracy | metric | none | true | false | false | 官方 wrapper 输出的 payload bit accuracy 辅助指标, 不作为主公平比较检测分数。 |
| score_semantics | protocol | none | true | false | false | 官方 wrapper 对 raw score 含义的显式声明。 |
| score_orientation | protocol | none | true | false | false | 官方 wrapper 对分数方向的显式声明, 当前主协议要求 higher_is_more_watermarked。 |
| external_baseline_raw_detector_score | metric | none | true | false | false | 现代 external baseline 归一化后的主检测分数, 用于后续 clean negative 阈值校准。 |
| external_baseline_score_field | protocol | none | true | false | false | 主检测分数在官方输出 JSON 中来自哪个字段。 |
| external_baseline_score_semantics | protocol | none | true | false | false | 主检测分数的语义, 用于防止混用 confidence、bit accuracy 和二值 decision。 |
| external_baseline_score_orientation | protocol | none | true | false | false | 主检测分数方向, 当前公平比较只接受 higher_is_more_watermarked。 |
| external_baseline_payload_bit_accuracy | metric | none | true | false | false | baseline payload 恢复 bit accuracy 辅助指标, 不替代主检测分数。 |
| external_baseline_clean_negative_score | metric | none | true | false | false | baseline 自身 clean negative 分布中的检测分数, 用于 target FPR 阈值校准。 |
| external_baseline_clean_negative_score_semantics | protocol | none | true | false | false | clean negative 分数语义, 必须与 external_baseline_score_semantics 对齐。 |
| external_baseline_clean_negative_video_path | artifact | none | true | false | false | baseline 自身 clean negative 视频路径, 用于审计阈值校准来源。 |
| clean_negative_ready | governance | none | true | true | false | self-containment 行中 measured_formal baseline 是否已经携带 clean negative 分数证据。 |
| clean_negative_ready_count | metric | none | true | true | false | self-containment 行中具备 clean negative 分数证据的 measured_formal record 数量。 |
| missing_clean_negative_modern_external_baseline_names | governance | none | true | false | false | self-containment 判定中缺少 clean negative 分数证据的现代 baseline 名称。 |
| official_clean_negative_source_video_path | artifact | none | true | false | false | official runtime 中未施加 runtime attack 的 baseline clean negative 视频路径。 |
| official_clean_negative_frame_array_path | artifact | none | true | false | false | VidSig official attack.py 读取的 clean negative 帧数组路径。 |
| official_clean_negative_attack_log_path | artifact | none | true | false | false | official detector 在 clean negative 视频上的日志路径。 |
| official_clean_negative_attack_stdout_path | artifact | none | true | false | false | official clean negative 检测命令 stdout 证据路径。 |
| official_clean_negative_attack_stderr_path | artifact | none | true | false | false | official clean negative 检测命令 stderr 证据路径。 |
| official_clean_negative_bit_accuracy_npz_path | artifact | none | true | false | false | SIGMark clean negative 官方 bit accuracy npz 路径。 |
| official_clean_negative_results_json_path | artifact | none | true | false | false | VideoMark clean negative 官方结果 JSON 路径。 |
| official_result_key | protocol | none | true | false | false | 官方输出文件中与当前 prompt / seed comparison unit 对应的 result key。 |
| official_temporal_attack_key | protocol | none | true | false | false | VideoMark 官方 temporal_results 中与当前 runtime attack 对齐的攻击 key。 |
| official_clean_negative_result_key | protocol | none | true | false | false | clean negative 官方输出文件中与当前 prompt / seed calibration unit 对应的 result key。 |
| official_clean_negative_output_path | artifact | none | true | false | false | 自动生成 clean negative 官方产物的输出目录。 |
| official_clean_negative_score_assignment_policy | protocol | none | true | false | false | 聚合型 official clean negative 分数如何映射到 comparison unit。 |
| official_score_assignment_policy | protocol | none | true | false | false | 正样本 official score 如何映射到当前 prompt / seed / attack comparison unit。 |
| official_detection_logic | protocol | none | true | false | false | 官方 wrapper 或 official runtime 记录的 detector / extractor 分数计算逻辑。 |
| official_clean_negative_video_io_backend | protocol | none | true | false | false | official clean negative 视频读取所使用的 I/O 后端。 |
| generate_clean_negative_reference | protocol | none | true | false | false | baseline 官方运行器是否自动生成 clean negative reference 并抽取官方分数。 |
| clean_negative_output_path | protocol | none | true | false | false | baseline 官方运行器 clean negative 输出目录配置。 |
| selected_clean_negative_bit_accuracy_npz_path | artifact | none | true | false | false | SIGMark 运行器选中的 clean negative bit accuracy npz。 |
| clean_negative_bit_accuracy_npz_candidates | artifact | none | true | false | false | SIGMark clean negative extract 阶段发现的 bit accuracy npz 候选列表。 |
| clean_negative_reference_state_manifest | artifact | none | true | false | false | SIGMark clean negative extract 复用官方 key/message 状态的复制审计。 |
| clean_negative_reference_state_status | governance | none | true | false | false | SIGMark clean negative reference 状态文件复制是否完成。 |
| copied_reference_state_count | metric | none | true | false | false | SIGMark clean negative reference 复制的官方状态文件数量。 |
| copied_reference_state_files | artifact | none | true | false | false | SIGMark clean negative reference 复制的状态文件路径列表。 |
| clean_negative_gen_command | artifact | none | true | false | false | SIGMark clean negative 官方 gen 命令。 |
| clean_negative_extract_command | artifact | none | true | false | false | SIGMark clean negative 官方 extract 命令。 |
| clean_negative_embedding_command | artifact | none | true | false | false | VideoMark clean negative 官方 embedding/extraction 命令。 |
| clean_negative_temporal_tamper_command | artifact | none | true | false | false | VideoMark clean negative 官方 temporal tamper 命令。 |
| clean_negative_video_results_json_path | artifact | none | true | false | false | VideoMark clean negative video_results.json 路径。 |
| clean_negative_temporal_results_json_path | artifact | none | true | false | false | VideoMark clean negative temporal_results.json 路径。 |
| clean_negative_temporal_results_json_exists | governance | none | true | false | false | VideoMark clean negative temporal_results.json 是否已落盘。 |
| clean_negative_attack_transform | protocol | none | true | false | false | clean negative 视频施加的 runtime attack 变换描述。 |
| clean_negative_attack_strength | protocol | none | true | false | false | clean negative 视频施加的 runtime attack 强度描述。 |
| fair_comparison_protocol | protocol | none | true | false | false | 公平比较协议名称, 当前为 method-specific clean negative calibration 到统一 target FPR。 |
| require_fair_detection_calibration | protocol | none | true | false | false | validation_scale gate 是否要求 clean negative 公平校准通过。 |
| minimum_clean_negative_count | protocol | none | true | false | false | 每个方法校准 target FPR 所需的最小 clean negative 分数数量。 |
| fair_detection_calibration_decision | governance | none | true | true | false | clean negative calibration 公平比较阶段判定。 |
| fair_detection_calibration_target_fpr | protocol | none | true | false | false | pilot_paper gate 读取到的 fair detection calibration decision 的 target_fpr。 |
| fair_detection_calibration_status | governance | none | true | false | false | pilot_paper gate 汇总的 fair detection calibration 产物状态。 |
| fair_detection_calibration_record_id | protocol | none | true | false | false | 单条公平校准记录的稳定标识。 |
| fair_comparison_status | governance | none | true | true | false | 单个方法的公平校准状态。 |
| fair_comparison_missing_reasons | governance | none | true | false | false | 单个方法未通过公平校准的原因列表。 |
| positive_score_field | protocol | none | true | false | false | attacked positive TPR 统计使用的分数字段。 |
| clean_negative_score_field | protocol | none | true | false | false | clean negative 阈值校准使用的分数字段。 |
| clean_negative_score_count | metric | none | true | true | false | clean negative 校准分数数量。 |
| attacked_positive_score_count | metric | none | true | true | false | attacked positive 检测分数数量。 |
| positive_anchor_count | metric | none | true | true | false | fair calibration 中 attacked positive 的 prompt / seed / attack anchor 数量。 |
| positive_anchor_missing_count | metric | none | true | true | false | fair calibration 中带分数但缺少 prompt_id、seed_id 或 attack_name 的 attacked positive 记录数量。 |
| positive_formal_evidence_missing_count | metric | none | true | true | false | fair calibration 中带分数但缺少 official evidence、clean negative 或完整 anchor 的 formal attacked positive 记录数量。 |
| negative_formal_evidence_missing_count | metric | none | true | true | false | fair calibration 中带 clean negative 分数但缺少 official evidence 或官方分数抽取证据的记录数量。 |
| positive_anchor_keys | protocol | none | true | false | false | fair calibration 中 attacked positive 的规范 prompt / seed / attack anchor 键集合。 |
| positive_detection_units_at_target_fpr | metric | none | true | false | false | target FPR 阈值下每个 prompt / seed / attack anchor 的检测结果列表。 |
| comparison_anchor_key | protocol | none | true | false | false | 由 prompt_id、seed_id 和 attack_name 组成的公平比较锚点键。 |
| calibrated_threshold | metric | none | true | true | false | 在方法自身 clean negative 分布上校准得到的检测阈值。 |
| threshold_selection_policy | protocol | none | true | false | false | 阈值选择策略。 |
| heldout_fpr_at_calibrated_threshold | metric | none | true | true | false | 校准阈值在 clean negative 分布上的经验 FPR。 |
| detected_positive_count_at_target_fpr | metric | none | true | true | false | target FPR 阈值下 detected attacked positive 数量。 |
| tpr_ci_confidence_level | metric | none | true | false | false | TPR 置信区间置信水平。 |
| tpr_ci_lower | metric | none | true | true | false | TPR 置信区间下界。 |
| tpr_ci_upper | metric | none | true | true | false | TPR 置信区间上界。 |
| fair_detection_calibration_method_count | metric | none | true | false | false | 公平校准要求覆盖的方法数量。 |
| fair_detection_calibration_ready_count | metric | none | true | true | false | 公平校准已通过的方法数量。 |
| fair_detection_calibration_missing_method_ids | governance | none | true | false | false | 公平校准缺失或阻断的方法列表。 |
| fair_detection_calibration_missing_method_count | metric | none | true | false | false | 公平校准缺失或阻断的方法数量。 |
| comparison_primary_metric_name | protocol | none | true | true | false | formal method baseline comparison 的主比较指标名称。 |
| comparison_primary_metric_value | metric | none | true | true | false | formal method baseline comparison 的主比较指标数值。 |
| source_fair_detection_target_fpr | protocol | none | true | true | false | formal comparison 行读取的上游 fair_detection_calibration record 的 target_fpr。 |
| comparison_anchor_count | metric | none | true | true | false | formal comparison 行中可用 prompt / seed / attack anchor 数量。 |
| reference_anchor_count | metric | none | true | true | false | 参考方法 SSTW 的 prompt / seed / attack anchor 数量。 |
| baseline_anchor_count | metric | none | true | true | false | baseline 方法的 prompt / seed / attack anchor 数量。 |
| missing_reference_anchor_count | metric | none | true | true | false | baseline 缺少的 SSTW 参考 anchor 数量。 |
| extra_anchor_count | metric | none | true | true | false | baseline 相比 SSTW 参考 anchor 多出的 anchor 数量。 |
| unpaired_reference_anchor_count | metric | none | true | true | false | 差值区间中无法与 baseline 配对的 SSTW anchor 数量。 |
| unpaired_baseline_anchor_count | metric | none | true | true | false | 差值区间中无法与 SSTW 配对的 baseline anchor 数量。 |
| comparison_anchor_alignment_status | governance | none | true | true | false | 当前方法是否与 SSTW 使用同一 prompt / seed / attack anchor 集合。 |
| reference_source_fair_detection_target_fpr | protocol | none | true | true | false | 差值区间中参考方法上游 fair_detection_calibration record 的 target_fpr。 |
| baseline_source_fair_detection_target_fpr | protocol | none | true | true | false | 差值区间中 baseline 上游 fair_detection_calibration record 的 target_fpr。 |
| reference_tpr_at_target_fpr | metric | none | true | true | false | 差值区间中参考方法的 TPR@target FPR。 |
| baseline_tpr_at_target_fpr | metric | none | true | true | false | 差值区间中 baseline 的 TPR@target FPR。 |
| tpr_at_target_fpr_difference | metric | none | true | true | false | SSTW TPR@target FPR 减 baseline TPR@target FPR 的差值。 |
