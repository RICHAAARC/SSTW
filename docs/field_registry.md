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
| record_version | protocol | none | true | false | false | synthetic_state_inference_sanity event record schema version. |
| sample_id | protocol | none | true | false | false | Synthetic sample identifier. |
| sample_role | protocol | none | true | false | false | Sample role. |
| method_variant | protocol | none | true | false | false | Controlled method or baseline variant. |
| attack_name | protocol | none | true | false | false | Synthetic attack name. |
| attack_strength | protocol | none | true | false | false | Synthetic attack strength parameter. |
| key_id | protocol | none | true | false | false | Watermark key identifier. |
| content_id | protocol | none | true | false | false | Synthetic content identifier. |
| prompt_id_placeholder | placeholder | _placeholder | true | false | true | synthetic_state_inference_sanity placeholder for later prompt identifier. |
| seed_id | protocol | none | true | false | false | Deterministic synthetic seed identifier. |
| generation_model_id_placeholder | placeholder | _placeholder | true | false | true | synthetic_state_inference_sanity placeholder for later generation model identifier. |
| backend_id | protocol | none | true | false | false | Runtime backend identifier. |
| tubelet_length | method | none | true | false | false | Tubelet temporal length. |
| tubelet_spatial_patch | method | none | true | false | false | Tubelet spatial patch size. |
| tubelet_stride_t | method | none | true | false | false | Tubelet temporal stride. |
| tubelet_stride_xy | method | none | true | false | false | Tubelet spatial stride. |
| watermark_alpha | method | none | true | false | false | Projection margin used by synthetic_state_inference_sanity synthetic embedding proxy. |
| payload_code_id | method | none | true | false | false | Payload code configuration identifier. |
| sync_code_id | method | none | true | false | false | Synchronization code configuration identifier. |
| joint_code_mode | method | none | true | false | false | Joint payload and synchronization code mode. |
| embedding_mode | method | none | true | false | false | Embedding mode identifier. |
| state_model_id | method | none | true | false | false | State model identifier. |
| state_dim | method | none | true | false | false | State vector dimension. |
| key_condition_mode | method | none | true | false | false | How key conditioning is injected. |
| filter_mode | method | none | true | false | false | State filtering mode. |
| smoother_enabled | method | none | true | false | false | Whether smoother is enabled. |
| phase_state_proxy | method | none | true | false | false | synthetic_state_inference_sanity proxy for phase state. |
| evidence_state_proxy | method | none | true | false | false | synthetic_state_inference_sanity proxy for evidence state. |
| confidence_state_proxy | method | none | true | false | false | synthetic_state_inference_sanity proxy for confidence state. |
| disturbance_state_proxy | method | none | true | false | false | synthetic_state_inference_sanity proxy for disturbance state. |
| state_entropy | metric | none | true | false | false | State uncertainty score. |
| state_coverage_ratio | metric | none | true | false | false | State coverage ratio. |
| state_matched_count | metric | none | true | false | false | Number of matched state elements. |
| state_transition_residual | metric | none | true | false | false | State transition residual. |
| S_payload_raw | metric | none | true | false | false | Raw payload score. |
| S_payload_state | metric | none | true | false | false | State-aligned payload score. |
| S_state_posterior | metric | none | true | false | false | State posterior score. |
| S_trajectory_observation_placeholder | placeholder | _placeholder | true | false | true | synthetic_state_inference_sanity placeholder for trajectory observation score. |
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
| trajectory_trace_placeholder | placeholder | _placeholder | true | false | true | synthetic_state_inference_sanity placeholder for trajectory trace. |
| real_video_quality_metrics_placeholder | placeholder | _placeholder | true | false | true | synthetic_state_inference_sanity placeholder for real-video quality metrics. |
| semantic_consistency_placeholder | placeholder | _placeholder | true | false | true | synthetic_state_inference_sanity placeholder for semantic consistency metric. |
| placeholder_reason | protocol | none | true | false | false | Reason explaining placeholder presence. |
| replacement_stage | protocol | none | true | false | false | Stage expected to replace placeholder. |
| replacement_field_name | protocol | none | true | false | false | Concrete field expected to replace placeholder. |
| source_video_id | protocol | none | true | false | false | real_video_latent_transfer_check source video identifier. |
| dataset_id | protocol | none | true | false | false | real_video_latent_transfer_check dataset identifier. |
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
| runtime_attack_names | protocol | none | true | false | false | runtime attack runner 已成功产出 attacked videos 的 attack 名称集合。 |
| attack_family | protocol | none | true | false | false | runtime attack 所属攻击家族, 例如 compression、temporal、spatial_geometry、visual_degradation 或 combined。 |
| runtime_attack_implementation_level | protocol | none | true | false | false | runtime attack 实现层级; paper profile 只能接受 formal_runtime_video_transform。 |
| runtime_attack_formal_evidence_level | governance | none | true | true | false | Runtime attack 是否来自正式视频文件级变换。 |
| runtime_attack_claim_level | governance | none | true | false | false | Runtime attack 可支持的论文协议层级。 |
| runtime_attack_proxy_free | governance | none | true | true | false | Runtime attack 是否未使用 proxy 或轻量替代实现。 |
| runtime_attack_formal_ready_count | metric | none | true | true | false | Runtime attack records 中正式视频文件级变换 ready 数量。 |
| runtime_attack_formal_missing_count | metric | none | true | true | false | Runtime attack ready records 中缺少正式视频文件级变换证据的数量。 |
| video_writer_codec | protocol | none | true | false | false | runtime attack 写出 attacked video 时请求的编码器名称。 |
| video_writer_output_params | protocol | none | true | false | false | runtime attack 写出 attacked video 时请求传递给视频编码器的参数列表。 |
| runtime_attack_observed_names | protocol | none | true | false | false | probe_paper gate 从 runtime_attack_records 观察到的 ready attack 名称集合。 |
| runtime_attack_missing_required_names | governance | none | true | false | false | probe_paper gate 中 runtime_attack_records 缺失的 required runtime attack 名称集合。 |
| runtime_attack_missing_required_count | metric | none | true | true | false | probe_paper gate 中 runtime_attack_records 缺失的 required runtime attack 数量。 |
| runtime_detection_observed_names | protocol | none | true | false | false | probe_paper gate 从 runtime_detection_records 观察到的 ready attack 名称集合。 |
| runtime_detection_missing_required_names | governance | none | true | false | false | probe_paper gate 中 runtime_detection_records 缺失的 required runtime attack 名称集合。 |
| runtime_detection_missing_required_count | metric | none | true | true | false | probe_paper gate 中 runtime_detection_records 缺失的 required runtime attack 数量。 |
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
| quality_psnr | metric | none | true | false | false | real_video_latent_transfer_check quality PSNR proxy. |
| quality_ssim | metric | none | true | false | false | real_video_latent_transfer_check quality SSIM proxy. |
| quality_lpips | metric | none | true | false | false | real_video_latent_transfer_check quality LPIPS value or null. |
| quality_metric_status | metric | none | true | false | false | Quality metric status. |
| quality_metric_failure_reason | protocol | none | true | false | false | Quality metric failure reason. |
| quality_not_collapsed | metric | none | true | false | false | Quality gate decision. |
| temporal_flicker_score | metric | none | true | false | false | Temporal flicker proxy score. |
| temporal_consistency_not_collapsed | metric | none | true | false | false | Temporal consistency gate decision. |
| motion_consistency_score_placeholder | placeholder | _placeholder | true | false | true | real_video_latent_transfer_check placeholder for motion consistency score. |
| motion_consistency_status | protocol | none | true | false | false | Motion consistency metric status. |
| motion_consistency_reason | protocol | none | true | false | false | Motion consistency status reason. |
| formal_state_schema_version | protocol | none | true | false | false | state_space_inference_formalization formal state schema version. |
| state_transition_model_id | method | none | true | false | false | state_space_inference_formalization transition model identifier. |
| state_observation_model_id | method | none | true | false | false | state_space_inference_formalization observation model identifier. |
| key_conditioner_id | method | none | true | false | false | state_space_inference_formalization key conditioner identifier. |
| smoother_mode | method | none | true | false | false | state_space_inference_formalization smoother mode. |
| state_entropy_gate_threshold | method | none | true | false | false | state_space_inference_formalization entropy gate threshold. |
| state_entropy_gate_status | metric | none | true | false | false | state_space_inference_formalization entropy gate status. |
| state_allowed_to_affect_final_score | metric | none | true | false | false | Whether state may affect final score. |
| trajectory_enabled | protocol | none | true | false | false | Whether trajectory observation is enabled. |
| trajectory_status | protocol | none | true | false | false | Trajectory status, explicit disabled in state_space_inference_formalization. |
| trajectory_state_adapter_placeholder | placeholder | _placeholder | true | false | true | state_space_inference_formalization placeholder for trajectory state adapter. |
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
| trajectory_scheduler_id_placeholder | placeholder | _placeholder | true | false | true | trajectory_observation_core_probe placeholder for scheduler identifier. |
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
| sampling_constraint_placeholder | placeholder | _placeholder | true | false | true | trajectory_observation_core_probe placeholder for sampling_time_constraint_probe sampling constraint config. |
| correlation_threshold | metric | none | true | false | false | trajectory_observation_core_probe correlation threshold. |
| correlation_status | metric | none | true | false | false | trajectory_observation_core_probe correlation audit status. |
| top_conference_trajectory_gate | governance | none | true | false | false | trajectory_observation_core_probe top conference trajectory gate decision. |
| generation_model_id | protocol | none | true | false | false | generative_video_model_probe generation model identifier. |
| generation_model_name | protocol | none | true | false | false | generative_video_model_probe generation model name. |
| generation_model_family | protocol | none | true | false | false | generative_video_model_probe generation model family. |
| primary_sstw_tc_model_id | protocol | none | true | false | false | Primary model identifier for SSTW-TC Flow Matching evaluation. |
| primary_sstw_tc_model_status | governance | none | true | false | false | Whether the runtime model matches the configured SSTW-TC primary model. |
| generation_model_version | protocol | none | true | false | false | generative_video_model_probe generation model version. |
| generation_model_role | protocol | none | true | false | false | Role assigned to a generation model in the SSTW-TC evaluation plan. |
| generation_model_requested_revision | protocol | none | true | false | false | 生成入口显式请求的 Hugging Face revision; 未指定时为空并解析默认 revision。 |
| generation_model_commit_or_hash | protocol | none | true | false | false | generative_video_model_probe generation model commit or hash. |
| generation_model_revision_source | provenance | none | true | true | false | 不可变模型 commit 来自配置解析、Hub 默认 revision 解析或离线显式 commit。 |
| generation_model_revision_resolution_status | governance | none | true | true | false | 生成与 replay 是否共同绑定到已经解析并冻结的不可变模型 commit。 |
| generation_model_provenance_decision | governance | none | true | true | false | 正式 Flow records 的注册模型家族、不可变 commit 与冻结来源是否全部通过。 |
| generation_model_provenance_failures | governance | none | true | false | false | 不满足正式生成模型 provenance 契约的记录及失败原因。 |
| generation_model_provenance_failure_reason | governance | none | true | false | false | 单条正式记录的模型注册或不可变 revision 校验失败原因。 |
| generation_model_license_status | protocol | none | true | false | false | generative_video_model_probe generation model license audit status. |
| generation_backend_id | protocol | none | true | false | false | generative_video_model_probe generation backend identifier. |
| generation_backend_status | protocol | none | true | false | false | generative_video_model_probe generation backend status. |
| generation_backend_reason | protocol | none | true | false | false | generative_video_model_probe generation backend status reason. |
| trajectory_capture_mode | trajectory | none | true | false | false | generative_video_model_probe trajectory capture mode. |
| trajectory_availability_status | trajectory | none | true | false | false | generative_video_model_probe trajectory availability status. |
| trajectory_capture_status | trajectory | none | true | false | false | generative_video_model_probe trajectory capture status. |
| trajectory_capture_failure_reason | trajectory | none | true | false | false | generative_video_model_probe trajectory capture failure reason. |
| latent_capture_status | protocol | none | true | false | false | generative_video_model_probe latent capture status. |
| latent_capture_failure_reason | protocol | none | true | false | false | generative_video_model_probe latent capture failure reason. |
| prompt_id | protocol | none | true | false | false | generative_video_model_probe prompt identifier. |
| prompt_text_hash | protocol | none | true | false | false | generative_video_model_probe prompt text digest. |
| prompt_category | protocol | none | true | false | false | generative_video_model_probe prompt category. |
| scheduler_id | protocol | none | true | false | false | generative_video_model_probe scheduler identifier. |
| trajectory_scheduler_id | trajectory | none | true | false | false | generative_video_model_probe trajectory scheduler identifier. |
| num_inference_steps | protocol | none | true | false | false | generative_video_model_probe inference step count. |
| guidance_scale | protocol | none | true | false | false | generative_video_model_probe guidance scale. |
| video_length_frames | protocol | none | true | false | false | generative_video_model_probe generated video length in frames. |
| fps | protocol | none | true | false | false | generative_video_model_probe generated video fps. |
| heldout_prompt_status | generalization | none | true | false | false | generative_video_model_probe heldout prompt status. |
| heldout_seed_status | generalization | none | true | false | false | generative_video_model_probe heldout seed status. |
| gpu_validation_status | governance | none | true | false | false | generative_video_model_probe local GPU validation status. |
| gpu_validation_reason | governance | none | true | false | false | generative_video_model_probe local GPU validation reason. |
| generation_model_runnable_status | governance | none | true | false | false | generative_video_model_probe generation model runnable status. |
| generation_model_not_run_reason | governance | none | true | false | false | generative_video_model_probe generation model not run reason. |
| visual_quality_score | metric | none | true | false | false | generative_video_model_probe visual quality score. |
| motion_consistency_score | metric | none | true | false | false | generative_video_model_probe motion consistency score. |
| motion_artifact_score | metric | none | true | false | false | generative_video_model_probe motion artifact score. |
| motion_metric_status | metric | none | true | false | false | generative_video_model_probe motion metric status. |
| semantic_consistency_score | metric | none | true | false | false | generative_video_model_probe semantic consistency score. |
| semantic_metric_name | metric | none | true | false | false | generative_video_model_probe semantic metric name. |
| semantic_metric_status | metric | none | true | false | false | generative_video_model_probe semantic metric status. |
| metric_failure_reason | protocol | none | true | false | false | generative_video_model_probe metric failure reason. |
| external_baseline_name | protocol | none | true | false | false | generative_video_model_probe external baseline name. |
| external_baseline_version | protocol | none | true | false | false | generative_video_model_probe external baseline version. |
| external_baseline_runnable_status | governance | none | true | false | false | generative_video_model_probe external baseline runnable status. |
| external_baseline_not_run_reason | governance | none | true | false | false | generative_video_model_probe external baseline not run reason. |
| external_baseline_protocol_gap | governance | none | true | false | false | generative_video_model_probe external baseline protocol gap. |
| external_baseline_result_used_for_claim | claim | none | true | true | false | Whether generative_video_model_probe external baseline is used for a claim. |
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
| paper_gate_profile | protocol | none | true | false | false | 当前 Colab profile 是否属于 probe_paper / pilot_paper / full_paper 这类 paper gate profile。 |
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
| generation_model_main_table_ready | governance | none | true | false | false | generative_video_model_probe main table readiness status. |
| trajectory_observation_gain_confirmed | metric | none | true | false | false | generative_video_model_probe trajectory gain confirmation status. |
| fixed_low_fpr_audit_pass | metric | none | true | false | false | generative_video_model_probe fixed low-FPR audit status. |
| quality_motion_semantic_consistency_pass | metric | none | true | false | false | generative_video_model_probe quality motion semantic gate status. |
| cross_prompt_generalization_pass | generalization | none | true | false | false | generative_video_model_probe cross prompt generalization status. |
| cross_seed_generalization_pass | generalization | none | true | false | false | generative_video_model_probe cross seed generalization status. |
| cross_motion_generalization_pass | generalization | none | true | false | false | generative_video_model_probe cross motion generalization status. |
| cross_length_generalization_pass | generalization | none | true | false | false | generative_video_model_probe cross length generalization status. |
| cross_prompt_seed_generalization_pass | generalization | none | true | false | false | generative_video_model_probe combined prompt seed generalization status. |
| generalization_failure_reason | generalization | none | true | false | false | generative_video_model_probe generalization failure reason. |
| formal_claim_status | claim | none | true | true | false | generative_video_model_probe formal claim status. |
| top_conference_generative_video_model_probe_gate | governance | none | true | false | false | generative_video_model_probe top conference gate decision. |
| threshold_status | protocol | none | true | false | false | generative_video_model_probe threshold computation status. |
| threshold_not_run_reason | protocol | none | true | false | false | generative_video_model_probe threshold not run reason. |
| prompt_suite_id | protocol | none | true | false | false | generative_video_model_probe Colab prompt suite identifier. |
| prompt_suite_role | protocol | none | true | false | false | generative_video_model_probe prompt or seed role inside prompt suite. |
| prompt_suite_digest | artifact | none | true | false | false | generative_video_model_probe prompt suite digest. |
| dataset_construction_status | governance | none | true | false | false | generative_video_model_probe input dataset construction status. |
| dataset_source | protocol | none | true | false | false | generative_video_model_probe input dataset source description. |
| prompt_negative_text | protocol | none | false | false | false | generative_video_model_probe prompt negative text kept in input dataset, not formal result records. |
| colab_runtime_profile | protocol | none | true | false | false | generative_video_model_probe Colab runtime profile. |
| config_kind | governance | none | true | false | false | 配置或 artifact 的语义类型。 |
| config_version | governance | none | true | false | false | 配置或 artifact 的版本标识。 |
| default_drive_project_root | protocol | none | false | false | false | Colab workflow 统一配置中的默认 Google Drive 项目根目录。 |
| default_dataset_root_relative | protocol | none | false | false | false | Colab workflow 统一配置中的默认 dataset 相对目录。 |
| default_prompt_suite_path_relative | protocol | none | false | false | false | Colab workflow 统一配置中的默认 prompt suite 相对路径。 |
| default_workflow_profile_by_notebook_role | protocol | none | false | false | false | Colab workflow 统一配置中每类 Notebook 的默认 workflow profile 映射。 |
| workflow_profile_aliases | protocol | none | false | false | false | Colab workflow 统一配置中旧 profile 名称到规范 profile 名称的别名映射。 |
| workflow_profiles | protocol | none | false | false | false | Colab workflow 统一配置中所有结果层级 profile 的配置映射。 |
| workflow_profile | protocol | none | true | false | false | Colab workflow 的规范 profile 名称, 用于区分 motion_calibration、probe_paper、pilot_paper 和 full_paper。 |
| requested_workflow_profile | protocol | none | true | false | false | 用户或环境变量请求的原始 workflow profile 名称。 |
| canonical_workflow_profile | protocol | none | true | false | false | 经过 alias 解析后的规范 workflow profile 名称。 |
| profile_alias_applied | governance | none | true | false | false | requested_workflow_profile 是否经过 alias 映射。 |
| profile_status | governance | none | true | false | false | workflow profile 的实现或阻断状态。 |
| enabled_for_run | governance | none | true | false | false | workflow profile 当前是否允许作为可运行入口。 |
| enabled_for_claim | governance | none | true | false | false | workflow profile 当前是否允许支撑论文 claim。 |
| runtime_profile | protocol | none | true | false | false | workflow profile 映射到 experiments runner 的 runtime profile。 |
| result_tier | protocol | none | true | false | false | 当前结果层级, 例如 probe_paper、pilot_paper 或 full_paper。 |
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
| cross_model_role | generalization | none | true | false | false | generative_video_model_probe model role for cross-model validation. |
| generation_status | protocol | none | true | false | false | generative_video_model_probe generation execution status. |
| generation_failure_reason | protocol | none | true | false | false | generative_video_model_probe generation failure reason. |
| generation_runtime_sec | metric | none | true | false | false | generative_video_model_probe generation runtime in seconds. |
| video_path | artifact | none | true | false | false | generative_video_model_probe generated video path. |
| video_sha256 | artifact | none | true | false | false | generative_video_model_probe generated video hash. |
| trajectory_step_index | trajectory | none | true | false | false | generative_video_model_probe trajectory callback step index. |
| trajectory_timestep | trajectory | none | true | false | false | generative_video_model_probe trajectory callback timestep. |
| latent_norm | metric | none | true | false | false | generative_video_model_probe latent tensor norm from trajectory callback. |
| latent_mean | metric | none | true | false | false | generative_video_model_probe latent tensor mean from trajectory callback. |
| latent_std | metric | none | true | false | false | generative_video_model_probe latent tensor standard deviation from trajectory callback. |
| cross_model_validation_status | generalization | none | true | false | false | generative_video_model_probe cross model validation status. |
| external_baseline_comparison_status | governance | none | true | false | false | generative_video_model_probe external baseline comparison status. |
| drive_project_root | artifact | none | true | false | false | Google Drive SSTW project root used by Colab workflow. |
| drive_dataset_root | artifact | none | true | false | false | Google Drive dataset output directory for generative_video_model_probe Colab workflow. |
| drive_run_root | artifact | none | true | false | false | Google Drive run output directory for generative_video_model_probe Colab workflow. |
| drive_package_dir | artifact | none | true | false | false | Google Drive package output directory for generative_video_model_probe Colab workflow. |
| drive_log_dir | artifact | none | true | false | false | Google Drive log output directory for generative_video_model_probe Colab workflow. |
| run_root | artifact | none | true | false | false | Run root packaged by Drive packager. |
| archive_path | artifact | none | true | false | false | Archive path created by Drive packager. |
| package_manifest_path | artifact | none | true | false | false | Package manifest path created by Drive packager. |
| include_videos | protocol | none | true | false | false | Whether generated videos are included in Drive package. |
| created_at | protocol | none | true | false | false | Creation timestamp for package manifest. |
| decision_summary | governance | none | true | false | false | Summary of stage decision embedded in package manifest. |
| generation_manifest_status | governance | none | true | false | false | Status showing whether generation manifest was present during packaging. |
| hf_token_status | governance | none | true | false | false | Whether HF_TOKEN was provided to Colab runtime; token value is never recorded. |
| implementation_evidence_status | governance | none | true | false | false | generative_video_model_probe Colab result checker implementation evidence status. |
| mechanism_evidence_status | governance | none | true | false | false | generative_video_model_probe Colab result checker mechanism evidence status. |
| missing_mechanism_requirements | governance | none | true | false | false | generative_video_model_probe missing mechanism requirements list. |
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
| external_baseline_family | protocol | none | true | false | false | generative_video_model_probe external baseline method family. |
| external_baseline_source_url | protocol | none | true | false | false | generative_video_model_probe external baseline source URL or local source marker. |
| external_baseline_recommended_priority | governance | none | true | false | false | generative_video_model_probe external baseline recommendation priority. |
| external_baseline_selection_role | governance | none | true | false | false | generative_video_model_probe role assigned to the external baseline. |
| external_baseline_integration_status | governance | none | true | false | false | generative_video_model_probe external baseline integration status. |
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
| external_baseline_score_status | governance | none | true | false | false | Per-record status of an external baseline score; formal comparison only accepts measured_formal, while adapter_not_integrated remains blocked. |
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
| external_baseline_measured_adapter_count | metric | none | true | false | false | Number of external_baseline adapters with formal measured comparison records. |
| external_baseline_measured_adapter_names | protocol | none | true | false | false | Names of external_baseline adapters with formal measured comparison records. |
| external_baseline_unsupported_adapter_count | metric | none | true | false | false | Number of unsupported external baseline comparison records or rows. |
| external_baseline_comparison_table_status | governance | none | true | false | false | Whether the external baseline comparison table was rebuilt from governed records. |
| validation_external_baseline_comparison_records_ready | governance | none | true | false | false | Whether probe-paper gate found external_baseline comparison records and enough measured adapters. |
| minimum_external_baseline_measured_adapter_count | protocol | none | true | false | false | Minimum measured external_baseline adapter count required by probe-paper gate. |
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
| metric_status | governance | none | true | false | false | 比较记录的测量状态。正式论文比较只接受 measured_formal; 历史 measured_proxy 只能作为阻断或兼容输入, 不能支持 claim。 |
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
| proposed_method_score_mean | metric | none | true | false | false | 正式外部 baseline 对比表中 SSTW 视频内容检测分数的平均值。 |
| external_baseline_score_mean | metric | none | true | false | false | 正式外部 baseline 对比表中 external baseline measured_formal 检测分数的平均值。 |
| external_baseline_distance_mean | metric | none | true | false | false | Mean external baseline adapter distance in a comparison table. |
| baseline_score_margin_mean | metric | none | true | false | false | 正式对比表中 SSTW 与 external baseline 分数或 TPR 指标的平均差值。 |
| paper_profile_gate_decision | governance | none | true | false | false | Decision for probe-paper generative probe gate before pilot_paper full-protocol run. |
| probe_paper_result_level | governance | none | true | true | false | package manifest 中记录的 probe_paper 结果级别。 |
| probe_paper_target_fpr | protocol | none | true | false | false | package manifest 中记录的 probe_paper protocol config target_fpr 摘要。 |
| missing_validation_requirements | governance | none | true | false | false | Paper profile requirements that are not yet satisfied; historical field name retained for compatibility. |
| validation_missing_requirement_count | metric | none | true | false | false | Count of missing probe-paper requirements. |
| probe_paper_hard_required_config_missing | governance | none | true | false | false | probe_paper 阶段不可通过配置关闭的公平比较硬前置缺口列表。 |
| probe_paper_hard_required_config_missing_count | metric | none | true | false | false | probe_paper 公平比较硬前置配置缺口数量。 |
| validation_profile_names | protocol | none | true | false | false | Runtime profile names accepted as probe-paper generation records. |
| validation_generation_record_count | metric | none | true | false | false | Number of successful generation records produced by probe-paper profiles. |
| validation_prompt_count | metric | none | true | false | false | Number of prompts covered by probe-paper generation records. |
| validation_seed_per_prompt_min | metric | none | true | false | false | Minimum seed count per prompt in probe-paper generation records. |
| minimum_prompt_count | protocol | none | true | false | false | Minimum prompt count required by a gate. |
| minimum_seed_per_prompt | protocol | none | true | false | false | Minimum seed count per prompt required by a gate. |
| internal_ablation_record_count | metric | none | true | false | false | Number of internal ablation records available to a validation or full-paper gate. |
| internal_ablation_status | governance | none | true | false | false | Internal ablation readiness or claim boundary status. |
| validation_internal_ablation_decision | governance | none | true | false | false | Decision for probe-paper formal internal ablation runner. |
| validation_internal_ablation_variant_count | metric | none | true | false | false | Number of method variants covered by probe-paper internal ablation records. |
| validation_internal_ablation_attack_count | metric | none | true | false | false | Number of attacks covered by probe-paper internal ablation records. |
| validation_internal_ablation_score_margin | metric | none | true | false | false | Mean score margin between full method and formal component-removal variants in probe-paper. |
| validation_internal_ablation_evidence_level | governance | none | true | false | false | Evidence level for probe-paper internal ablation records. |
| ablation_runtime_profile | protocol | none | true | false | false | Internal ablation record 对应的 runtime profile, 用于区分 probe_paper、pilot_paper 与 full_paper 覆盖。 |
| validation_internal_ablation_profile_counts | metric | none | true | false | false | Internal ablation records 按 runtime profile 汇总的数量映射。 |
| validation_internal_ablation_missing_variants | governance | none | true | true | false | probe_paper / pilot_paper / full_paper 同构内部消融矩阵缺失的 method variant 列表。 |
| validation_internal_ablation_trace_counts | metric | none | true | false | false | 正式内部消融记录按 method variant 汇总的 trajectory trace 数量映射。 |
| pilot_paper_internal_ablation_record_count | metric | none | true | true | false | Internal ablation records 中属于 pilot_paper profile 的数量。 |
| validation_ablation_evidence_level | governance | none | true | false | false | Historical compatibility field for per-record probe-paper ablation evidence level; formal records use formal_internal_ablation_evidence_level. |
| validation_ablation_source_score | metric | none | true | false | false | Historical compatibility field for the source score used by a probe-paper ablation record; formal records use formal_internal_ablation_score. |
| validation_ablation_proxy_score | metric | none | true | false | false | Historical compatibility field that must not support formal claims; formal records use formal_internal_ablation_score. |
| adaptive_attack_record_count | metric | none | true | false | false | Number of Flow-specific adaptive attack records available to a validation or full-paper gate. |
| formal_adaptive_attack_record_count | metric | none | true | true | false | Number of adaptive attack records backed by measured_formal formal_adaptive_attack_execution evidence. |
| adaptive_attack_non_formal_record_count | metric | none | true | true | false | Number of adaptive attack records that are missing measured_formal formal_adaptive_attack_execution evidence and therefore cannot support paper claims. |
| adaptive_attack_status | governance | none | true | false | false | Adaptive attack readiness or claim boundary status. |
| adaptive_attack_decision | governance | none | true | false | false | Adaptive attack validation runner 决策。 |
| adaptive_attack_name | protocol | none | true | false | false | Adaptive attack 名称。 |
| adaptive_attack_family | protocol | none | true | false | false | Adaptive attack 家族。 |
| adaptive_attack_strength | metric | none | true | false | false | Adaptive attack 强度。 |
| adaptive_attack_score | metric | none | true | true | false | Formal adaptive attack measured score, usually TPR or robustness score under the registered non-runtime/adaptive protocol. |
| adaptive_attack_failure_reason | governance | none | true | false | false | Formal adaptive attack 记录缺失或不可用时的阻断原因。 |
| adaptive_attack_budget | protocol | none | true | false | false | Adaptive attack 预算。 |
| attack_knowledge_level | protocol | none | true | false | false | 攻击者知识层级。 |
| targeted_evidence_layer | protocol | none | true | false | false | Adaptive attack 目标证据层。 |
| endpoint_preservation_status | governance | none | true | false | false | Endpoint 保持状态。 |
| path_response_suppression_score | metric | none | true | false | false | Path response suppression proxy 分数。 |
| velocity_projection_suppression_score | metric | none | true | false | false | Velocity projection suppression proxy 分数。 |
| adaptive_residual_proxy_score | metric | none | true | false | false | Historical compatibility field for old adaptive residual diagnostics; formal adaptive evidence uses adaptive_attack_score with measured_formal inputs。 |
| replay_signature_mismatch_status | governance | none | true | false | false | Replay signature mismatch 状态。 |
| trajectory_sketch_tamper_status | governance | none | true | false | false | Trajectory sketch tamper 状态。 |
| quality_guard_status | governance | none | true | false | false | Quality guard 状态。 |
| semantic_projection_status | governance | none | true | false | false | Semantic projection 状态。 |
| adaptive_negative_fpr | metric | none | true | false | false | Adaptive negative FPR。 |
| adaptive_negative_fpr_status | governance | none | true | false | false | Adaptive negative FPR 可用状态。 |
| adaptive_attack_success_status | governance | none | true | false | false | Adaptive attack success 状态; 正式论文证据必须同时具备 measured_formal 和 formal_adaptive_attack_execution evidence。 |
| adaptive_attack_claim_support_status | claim | none | true | false | false | Adaptive attack claim 支撑状态。 |
| adaptive_attack_name_count | metric | none | true | false | false | Adaptive attack 名称覆盖数量。 |
| adaptive_attack_family_count | metric | none | true | false | false | Adaptive attack family 覆盖数量。 |
| adaptive_attack_knowledge_level_count | metric | none | true | false | false | Adaptive attack 攻击者知识层级覆盖数量。 |
| adaptive_attack_targeted_layer_count | metric | none | true | false | false | Adaptive attack 目标证据层覆盖数量。 |
| adaptive_attack_missing_names | governance | none | true | false | false | 缺失的 adaptive attack 名称列表。 |
| adaptive_attack_score_mean | metric | none | true | false | false | Formal adaptive attack measured score 的平均值, 由 formal_adaptive_attack_execution records 聚合。 |
| adaptive_robustness_claim_allowed | governance | none | true | false | false | 是否允许 adaptive robustness 强 claim。 |
| adaptive_attack_evidence_level | governance | none | true | false | false | Adaptive attack evidence 等级。 |
| replay_or_sketch_status | governance | none | true | false | false | Replay/sketch 完整证据就绪状态; 正式 profile 不接受 Claim-3 降级状态。 |
| claim3_original_scope | claim | none | true | false | false | Claim-3 原始强主张范围。 |
| claim3_allowed_scope | claim | none | true | false | false | 当前证据允许的 Claim-3 范围。 |
| claim3_full_support_allowed | governance | none | true | false | false | 是否允许把 Claim-3 写成强 supported claim。 |
| claim3_missing_replay_requirement_count | metric | none | true | false | false | Claim-3 replay/sketch 缺失要求数量。 |
| claim3_missing_replay_requirements | governance | none | true | false | false | Claim-3 replay/sketch 缺失要求列表。 |
| authenticated_trajectory_sketch_status | governance | none | true | false | false | Authenticated trajectory sketch 就绪状态。 |
| trajectory_sketch_verification_status | governance | none | true | false | false | Trajectory sketch 验证状态。 |
| replay_uncertainty_records_ready | governance | none | true | false | false | Replay uncertainty records 是否就绪。 |
| wrong_sampler_replay_records_ready | governance | none | true | false | false | Wrong sampler replay records 是否就绪。 |
| wrong_prompt_replay_records_ready | governance | none | true | false | false | Wrong prompt replay records 是否就绪。 |
| confidence_interval_status | governance | none | true | false | false | Statistical confidence interval report readiness status. |
| statistical_confidence_interval_decision | governance | none | true | false | false | Decision for probe-paper statistical confidence interval reporter. |
| statistical_confidence_interval_family | governance | none | true | false | false | Metric family covered by a statistical confidence interval record. |
| ci_record_count | metric | none | true | false | false | Number of confidence interval records. |
| ci_success_count | metric | none | true | false | false | Count of successful events used in a confidence interval. |
| ci_total_count | metric | none | true | false | false | Total event count used in a confidence interval. |
| ci_point_estimate | metric | none | true | false | false | Point estimate for a confidence interval. |
| ci_wilson_lower | metric | none | true | false | false | Wilson lower bound for a binomial confidence interval. |
| ci_wilson_upper | metric | none | true | false | false | Wilson upper bound for a binomial confidence interval. |
| ci_confidence_level | protocol | none | true | false | false | Confidence level used by a confidence interval record. |
| ci_evidence_level | governance | none | true | false | false | Evidence level for a probe-paper confidence interval record. |
| cluster_by_video_interval_status | governance | none | true | false | false | Status of cluster-by-video confidence interval availability. |
| paper_low_fpr_ci_status | governance | none | true | false | false | Status of paper-level low-FPR confidence interval availability. |
| artifact_rebuild_status | governance | none | true | false | false | Artifact rebuild dry-run readiness status. |
| validation_artifact_rebuild_dry_run_decision | governance | none | true | false | false | Decision for probe-paper artifact rebuild dry-run. |
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
| selection_policy | governance | none | true | false | false | generative_video_model_probe external baseline selection policy block. |
| primary_selection_rule | governance | none | true | false | false | generative_video_model_probe external baseline primary selection rule. |
| claim_rule | claim | none | true | true | false | generative_video_model_probe external baseline claim usage rule. |
| fallback_rule | governance | none | true | false | false | generative_video_model_probe external baseline fallback rule. |
| internal_mechanism_baselines | protocol | none | true | false | false | generative_video_model_probe internal mechanism baseline list paired with external explicit synchronization baselines. |
| baseline_score_margin | metric | none | true | false | false | Score margin between key-conditioned trajectory score and the compared baseline score. |
| control_name | protocol | none | true | false | false | Controlled negative trajectory control name used by generative_video_model_probe postprocess. |
| runtime_mechanism_decision | governance | none | true | false | false | Raw mechanism decision reported by the generation runtime stage; final paper claims must use paper profile or pilot/full paper gate decisions. |
| effective_mechanism_decision | governance | none | true | false | false | Package-level mirror of the generation runtime mechanism decision; it no longer merges deleted proxy postprocess artifacts. |
| mechanism_decision_source | governance | none | true | false | false | Artifact source used for the package-level mechanism decision; current formal mainline uses runtime_mechanism_artifact and separate paper gates. |
| video_decode_status | governance | none | true | false | false | Decode status for generated mp4 files used by generative_video_model_probe formal metrics. |
| video_metric_failure_reason | governance | none | true | false | false | Failure reason for generated video file metric extraction. |
| decoded_frame_count | metric | none | true | false | false | Number of decoded frames sampled from a generated video. |
| sampled_frame_count | metric | none | true | false | false | Number of frames sampled for generative_video_model_probe video file metrics. |
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
| sampling_time_constraint_preflight_decision | governance | none | true | false | false | sampling_time_constraint_probe sampling-time constraint preflight decision. |
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
| callback_latent_displacement_available | governance | none | true | false | false | Whether adjacent callback latent state displacement is available in paper-profile trajectory traces. |
| callback_latent_displacement_source | protocol | none | true | false | false | Source used to derive adjacent callback latent displacement records. |
| callback_latent_displacement_evidence_level | protocol | none | true | false | false | Evidence level for callback latent displacement records in formal paper profiles. |
| callback_latent_displacement_norm_before_constraint | metric | none | true | false | false | Adjacent callback latent displacement norm before sampling-time callback update. |
| callback_latent_displacement_norm_after_constraint | metric | none | true | false | false | Adjacent callback latent displacement norm after sampling-time callback update. |
| callback_latent_displacement_alignment_before_constraint | metric | none | true | false | false | Alignment between adjacent callback latent displacement and keyed constraint direction before callback update. |
| callback_latent_displacement_alignment_after_constraint | metric | none | true | false | false | Alignment between adjacent callback latent displacement and keyed constraint direction after callback update. |
| callback_latent_displacement_alignment_gain | metric | none | true | false | false | Alignment gain showing whether callback latent state displacement participates in watermark synchronization. |
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
| constraint_variant_summary_records | artifact | none | true | false | false | Aggregated sampling_time_constraint_probe Colab constraint variant summary records. |
| keyed_constraint_alignment_gain_mean | metric | none | true | false | false | Mean alignment gain for keyed sampling constraint variant. |
| baseline_alignment_gain_mean | metric | none | true | false | false | Mean alignment gain for unconstrained trajectory baseline. |
| sampling_time_constraint_colab_probe | governance | none | true | false | false | Stage identifier for sampling_time_constraint_probe real sampling callback probe. |
| sampling_time_constraint_colab_postprocess | governance | none | true | false | false | Stage identifier for sampling_time_constraint_probe Colab postprocess artifacts. |

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
| pilot_paper_hard_required_config_missing | governance | none | true | false | false | pilot_paper 不可通过配置关闭的 probe_paper 与公平比较硬前置缺口列表。 |
| pilot_paper_hard_required_config_missing_count | metric | none | true | false | false | pilot_paper 硬前置配置缺口数量。 |
| paper_result_level | governance | none | true | true | false | 结果包的论文级别, 例如 pilot_paper 或 full_paper。 |
| paper_claim_id | governance | none | true | true | false | 三层正式论文结果包的 claim 标识, 只能使用 probe_claim、pilot_claim 或 full_claim。 |
| paper_claim_level | governance | none | true | true | false | 当前正式 claim 所属的结果层级, 与 paper_result_level 保持一致。 |
| paper_claim_support_status | governance | none | true | true | false | 当前三层正式 claim 的支持状态, 由 paper_result_formality_guard 与对应 gate/checker 联合给出。 |
| paper_result_formality_guard_decision | governance | none | true | false | false | 三层正式结果包是否拒绝 proxy、placeholder 和 fallback 证据的 PASS / FAIL 判定。 |
| paper_result_formality_guard_status | governance | none | true | false | false | 三层正式结果包正式性门禁的可读状态。 |
| paper_result_formality_guard_violation_count | metric | none | true | false | false | 三层正式结果包中被正式性门禁发现的弱证据标记数量。 |
| paper_result_formality_guard_scanned_file_count | metric | none | true | false | false | 三层正式结果包正式性门禁扫描的结构化文件数量。 |
| paper_result_formality_guard_blocking_terms | governance | none | true | false | false | 三层正式结果包正式性门禁发现的阻断标记类别。 |
| paper_result_formality_guard_violations | governance | none | true | false | false | 三层正式结果包正式性门禁发现的具体阻断位置列表。 |
| clean_negative_pair_role | protocol | none | true | false | false | clean negative 视频与同 prompt / seed 水印视频之间的配对方式。 |
| clean_negative_event_source | protocol | none | true | false | false | clean negative 分数事件来源, 例如同 prompt / seed clean video 的正式 key trial。 |
| clean_negative_key_trial_index | metric | none | true | false | false | 单个 clean video 上的正式 detector key trial 序号。 |
| clean_negative_key_trial_count_for_video | metric | none | true | false | false | 单个 clean video 为满足 fixed-FPR 校准所展开的 key trial 数量。 |
| clean_negative_source_video_index | metric | none | true | false | false | clean negative 视频在当前 run_root clean 视频集合中的序号。 |
| clean_negative_video_sha256 | artifact | none | true | false | false | clean negative 视频文件的 SHA-256 摘要。 |
| sstw_clean_negative_record_count | metric | none | true | false | false | SSTW clean negative detector records 总数。 |
| sstw_clean_negative_ready_count | metric | none | true | false | false | 具备 measured_formal 状态的 SSTW clean negative detector records 数量。 |
| sstw_clean_negative_missing_count | metric | none | true | false | false | 未达到正式 clean negative 检测条件的 SSTW clean negative records 数量。 |
| sstw_clean_negative_required | governance | none | true | false | false | 当前 runtime detection 阶段是否按 paper profile 配置要求 clean negative records。 |
| sstw_clean_negative_requirement_met | governance | none | true | false | false | 当前 runtime detection 阶段的 clean negative 正式检测要求是否满足。 |
| formal_adaptive_attack_execution_decision | governance | none | true | false | false | 11 个 non-runtime / adaptive 协议是否完成正式视频文件重检测执行。 |
| formal_adaptive_attack_execution_record_count | metric | none | true | false | false | formal adaptive attack execution records 总数。 |
| formal_adaptive_attack_execution_ready_count | metric | none | true | false | false | 具备 measured_formal 状态的 formal adaptive attack execution records 数量。 |
| formal_adaptive_attack_execution_record_id | artifact | none | true | false | false | 单条 formal adaptive attack execution record 的稳定 ID。 |
| adaptive_attack_input_video_path | artifact | none | true | false | false | adaptive attack 正式执行所读取的视频文件路径。 |
| adaptive_attack_input_video_sha256 | artifact | none | true | false | false | adaptive attack 正式执行所读取视频文件的 SHA-256 摘要。 |
| adaptive_attack_video_source_kind | protocol | none | true | false | false | adaptive attack 使用 source generation video 或 runtime transformed video 的来源类别。 |
| adaptive_attack_source_runtime_attack_name | protocol | none | true | false | false | adaptive attack 复用 runtime transformed video 时对应的 runtime attack 名称。 |
| adaptive_attack_detector_key_transformation | protocol | none | true | false | false | adaptive attack 对检测 key 或检测上下文施加的协议级变换。 |
| adaptive_attack_score_semantics | metric | none | true | false | false | adaptive attack score 的分数语义。 |
| adaptive_attack_score_orientation | metric | none | true | false | false | adaptive attack score 的方向, 正式比较要求 higher_is_more_watermarked。 |
| adaptive_attack_detected_by_sstw | metric | none | true | false | false | adaptive attack 输入视频在 SSTW 正式检测器默认判定阈值下是否为 positive。 |
| field_path | governance | none | true | false | false | 结构化 artifact 内部的字段路径, 用于定位正式性门禁发现的问题字段。 |
| violation_kind | governance | none | true | false | false | 正式性门禁发现的违规类型。 |
| observed_value_preview | governance | none | true | false | false | 正式性门禁记录的短值预览, 用于定位问题但不作为论文指标。 |
| paper_result_target_fpr_matches_profile | governance | none | true | false | false | 当前 target_fpr 是否匹配 probe_paper、pilot_paper 或 full_paper 的正式 FPR 配置。 |
| expected_formal_target_fpr | protocol | none | true | false | false | 当前 paper_result_level 对应的正式 target_fpr。 |
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
| full_paper_prompt_count | metric | none | true | false | false | full_paper result checker 读取到的 prompt 覆盖数量。 |
| full_paper_seed_per_prompt_min | metric | none | true | false | false | full_paper 每个 prompt 的最小 seed 覆盖数量。 |
| full_paper_calibration_seed_per_prompt_min | metric | none | true | false | false | full_paper calibration split 中每个 prompt 的最小 seed 覆盖数量。 |
| full_paper_test_seed_per_prompt_min | metric | none | true | false | false | full_paper held-out test split 中每个 prompt 的最小 seed 覆盖数量。 |
| full_paper_unique_video_count | metric | none | true | false | false | full_paper 成功生成且可进入 checker 的 unique video 数量。 |
| full_paper_calibration_unique_video_count | metric | none | true | false | false | full_paper calibration split 的 unique video 数量。 |
| full_paper_test_unique_video_count | metric | none | true | false | false | full_paper held-out test split 的 unique video 数量。 |
| full_paper_runtime_attack_event_count_per_attack_min | metric | none | true | false | false | full_paper runtime attack records 中每个必需 attack 的最小 ready 事件数。 |
| full_paper_runtime_detection_event_count_per_attack_min | metric | none | true | false | false | full_paper runtime detection records 中每个必需 attack 的最小 ready 事件数。 |
| full_paper_runtime_attack_event_counts | metric | none | true | false | false | full_paper runtime attack records 中各必需 attack 的 ready 事件数映射。 |
| full_paper_runtime_detection_event_counts | metric | none | true | false | false | full_paper runtime detection records 中各必需 attack 的 ready 事件数映射。 |
| minimum_unique_video_count | protocol | none | true | false | false | gate 要求的最小 unique video 数量。 |
| minimum_calibration_seed_per_prompt | protocol | none | true | false | false | gate 要求的每个 prompt 在 calibration split 中的最小 seed 数量。 |
| minimum_test_seed_per_prompt | protocol | none | true | false | false | gate 要求的每个 prompt 在 held-out test split 中的最小 seed 数量。 |
| minimum_calibration_unique_video_count | protocol | none | true | false | false | gate 要求的 calibration split 最小 unique video 数量。 |
| minimum_test_unique_video_count | protocol | none | true | false | false | gate 要求的 held-out test split 最小 unique video 数量。 |
| minimum_calibration_negative_event_count | protocol | none | true | false | false | gate 要求的最小 calibration negative event 数量。 |
| minimum_heldout_test_negative_event_count | protocol | none | true | false | false | gate 要求的最小 held-out test negative event 数量。 |
| minimum_heldout_attacked_positive_event_count | protocol | none | true | false | false | gate 要求的最小 held-out attacked positive event 数量。 |
| minimum_calibration_negative_event_count_per_family | protocol | none | true | false | false | gate 要求的 calibration split 每个 negative family 最小事件数。 |
| minimum_heldout_negative_event_count_per_family | protocol | none | true | false | false | gate 要求的 held-out split 每个 negative family 最小事件数。 |
| minimum_negative_event_count_per_family | protocol | none | true | false | false | gate 要求的每个 negative family 最小事件数, 用于检查 clean negative event 是否按 family 均衡覆盖。 |
| minimum_attack_event_count_per_attack | protocol | none | true | false | false | gate 要求的每个 attack 最小 held-out positive event 数量。 |
| minimum_external_baseline_trace_count | protocol | none | true | false | false | 当前 paper profile 对每个正式 baseline 的最小 held-out trace 覆盖数量, 该字段只允许随统计规模变化。 |
| minimum_internal_ablation_trace_count | protocol | none | true | false | false | 当前 paper profile 对每个必需消融变体的最小 held-out trace 覆盖数量, 该字段只允许随统计规模变化。 |
| paper_profile_names | protocol | none | true | false | false | 当前正式 profile 允许进入公共参数化 gate 的运行 profile 名称列表。 |
| profile_gate_entrypoint | protocol | none | true | false | false | 外层 workflow 为当前 profile 选择的公共 gate Python 模块入口。 |
| profile_gate_contract | protocol | none | true | false | false | 外层 workflow 声明三档 profile 共用参数化 gate 且仅允许 FPR 与样本统计规模变化。 |
| profile_stage_transition_control | governance | none | true | false | false | 公共 gate 通过后仍须由外层 workflow 的 profile 专属 transition stage 决定后续阶段。 |
| paper_profile_unique_video_count | metric | none | true | true | false | 公共 gate 按主模型 full-method 独立生成身份统计的当前 profile 视频总数。 |
| paper_profile_calibration_unique_video_count | metric | none | true | true | false | 公共 gate 在 calibration split 中统计的独立主方法视频数量。 |
| paper_profile_heldout_test_unique_video_count | metric | none | true | true | false | 公共 gate 在 held-out test split 中统计的独立主方法视频数量。 |
| paper_profile_calibration_seed_per_prompt_min | metric | none | true | true | false | calibration split 中每个 prompt 覆盖 seed 数量的最小值。 |
| paper_profile_test_seed_per_prompt_min | metric | none | true | true | false | held-out test split 中每个 prompt 覆盖 seed 数量的最小值。 |
| calibration_negative_video_cluster_count | metric | none | true | true | false | calibration negative 按 source-video cluster 去重后的独立统计单元数量。 |
| heldout_negative_video_cluster_count | metric | none | true | true | false | held-out negative 按 source-video cluster 去重后的独立统计单元数量。 |
| calibration_negative_family_cluster_counts | metric | none | true | false | false | calibration split 中各真实负假设族按 source-video cluster 去重后的数量映射。 |
| heldout_negative_family_cluster_counts | metric | none | true | false | false | held-out split 中各真实负假设族按 source-video cluster 去重后的数量映射。 |
| heldout_attack_event_counts | metric | none | true | false | false | held-out attacked-positive 按正式 attack 名统计的事件数量映射。 |
| external_baseline_trace_count_min | metric | none | true | true | false | 所有必需正式 baseline 在 held-out 主方法 trace 上覆盖数量的最小值。 |
| external_baseline_trace_counts | metric | none | true | false | false | 各必需正式 baseline 在 held-out 主方法 trace 上的覆盖数量映射。 |
| full_paper_gate_decision | governance | none | true | true | false | 公共参数化 gate 为 full_paper 写出的 profile 专属兼容决策字段。 |
| require_external_baseline_comparison_ready | protocol | none | true | false | false | pilot_paper gate 是否要求 external_baseline adapter comparison 已完成。 |
| require_modern_external_baseline_formal_results | protocol | none | true | false | false | pilot_paper gate 是否要求现代视频水印 baseline 使用正式 adapter measured_formal 结果。 |
| require_internal_ablation_matrix_ready | protocol | none | true | false | false | pilot_paper gate 是否要求内部消融矩阵已完成。 |
| require_data_split_and_leakage_guard | protocol | none | true | false | false | gate 是否要求 calibration / held-out test split 与阈值来源通过泄漏防护审计。 |
| require_external_baseline_self_containment_decision | protocol | none | true | false | false | gate 是否要求 external baseline 自包含产出判定已通过。 |
| require_complete_result_artifact_skeleton | protocol | none | true | false | false | gate 是否要求论文结果表格、图和报告骨架已由 records 自动构建。 |
| require_paper_result_artifact_skeleton | protocol | none | true | false | false | gate 是否要求当前 paper profile 的正式结果表格、图和报告骨架存在。 |
| require_video_quality_metric_records | protocol | none | true | false | false | gate 是否要求正式视频质量指标 records 已落盘。 |
| require_efficiency_metric_records | protocol | none | true | false | false | gate 是否要求效率指标 records 已落盘。 |
| require_low_fpr_curve_records | protocol | none | true | false | false | gate 是否要求低 FPR 曲线图表 records 已落盘。 |
| require_real_adaptive_attack_records | protocol | none | true | false | false | gate 是否要求真实 adaptive attack records 已落盘。 |
| require_real_world_attack_records | protocol | none | true | false | false | gate 是否要求真实平台、screen recording 或真实转码类 attack records 已落盘。 |
| require_adaptive_attack_records | protocol | none | true | false | false | gate 是否要求 adaptive attack 相关 records 已落盘。 |
| require_confidence_interval_report | protocol | none | true | false | false | gate 是否要求置信区间报告已生成。 |
| require_claim_audit_report | protocol | none | true | false | false | gate 是否要求 claim audit 报告已生成。 |
| require_artifact_rebuild_report | protocol | none | true | false | false | gate 是否要求 artifact rebuild 报告已生成。 |
| require_statistical_confidence_interval_decision | protocol | none | true | false | false | gate 是否要求统计置信区间轻量判定已通过。 |
| require_artifact_rebuild_dry_run | protocol | none | true | false | false | gate 是否要求 artifact rebuild dry-run 判定已通过。 |
| required_external_baseline_adapter_names | protocol | none | true | false | false | pilot_paper gate 要求出现的 external_baseline adapter 名称列表。 |
| required_modern_external_baseline_adapter_names | protocol | none | true | false | false | pilot_paper gate 要求产出 measured_formal records 的现代视频水印 baseline adapter 名称列表。 |
| required_internal_ablation_variants | protocol | none | true | false | false | pilot_paper gate 要求出现的内部消融 method variant 列表。 |
| minimum_internal_ablation_variant_count | protocol | none | true | false | false | pilot_paper gate 要求的内部消融变体最小数量。 |
| minimum_modern_external_baseline_formal_adapter_count | protocol | none | true | false | false | pilot_paper gate 要求的现代视频水印 measured_formal adapter 最小数量。 |
| pilot_paper_external_baseline_comparison_ready | governance | none | true | true | false | pilot_paper gate 中 external_baseline comparison 是否满足完整论文协议要求。 |
| pilot_paper_internal_ablation_matrix_ready | governance | none | true | true | false | pilot_paper gate 中 internal ablation matrix 是否满足完整论文协议要求。 |
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
| probe_paper_claim_support_status | governance | none | true | true | false | package manifest 或下游 gate 中记录的 probe-paper handoff 状态摘要; probe_paper 通过后支持 target_fpr=0.1 下的完整三层论文结论, 但不支持向 pilot_paper 或 full_paper 的更低 FPR 外推。 |
| paper_gate_preflight_layer | protocol | none | true | false | false | 当前 workflow profile 是否处于 paper gate 预检层; probe_paper 在 target_fpr=0.1 小样本论文闭合语义下必须为 true。 |
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
| runtime_detection_detectable_count | metric | none | true | false | false | Number of runtime detection records whose attacked videos are positive under the formal SSTW video content detector. |
| runtime_detection_attack_count | metric | none | true | false | false | Number of distinct attacks covered by runtime detection records. |
| runtime_detection_score_mean | metric | none | true | false | false | Mean runtime attacked video formal detector score. |
| runtime_detection_evidence_level | governance | none | true | false | false | Evidence level for runtime attacked video detection records. |
| sstw_detector_evidence_level | governance | none | true | true | false | SSTW detector score 的正式证据层级; paper profile 只能接受 attacked_video_content_detector。 |
| sstw_detector_input_contract | protocol | none | true | false | false | SSTW 正式 detector 输入契约, 当前为 video_file_plus_project_watermark_key。 |
| sstw_raw_detector_score | metric | none | true | true | false | SSTW 正式视频内容检测主分数。 |
| raw_detector_score | metric | none | true | true | false | 跨方法公平比较可复用的原始 detector score 字段。 |
| sstw_detector_score_semantics | protocol | none | true | false | false | SSTW detector score 的语义说明。 |
| sstw_detector_key_digest | provenance | none | true | false | false | SSTW detector key 的短 digest, 用于复现审计但不暴露原始 key。 |
| sstw_content_feature_count | metric | none | true | false | false | SSTW 视频内容检测器实际使用的低频时空特征数量。 |
| sstw_detector_sampled_frame_count | metric | none | true | false | false | SSTW 视频内容检测器实际抽取的视频帧数量。 |
| trajectory_trace_used_for_score | governance | none | true | true | false | 检测分数是否读取 generation trajectory trace; paper profile 要求为 false。 |
| runtime_detection_claim_level | governance | none | true | true | false | Runtime detection 可支持的论文 claim 层级。 |
| runtime_detection_formal_detector_ready_count | metric | none | true | true | false | runtime detection records 中正式 SSTW 视频内容检测 ready 数量。 |
| runtime_detection_formal_detector_missing_count | metric | none | true | true | false | runtime detection ready records 中缺少正式视频内容检测证据的数量。 |
| sstw_clean_negative_score | metric | none | true | true | false | SSTW clean negative 视频内容检测分数, 用于 fixed-FPR 校准。 |
| sstw_clean_negative_score_semantics | protocol | none | true | false | false | SSTW clean negative 分数语义。 |
| clean_negative_video_path | artifact | none | true | false | false | SSTW clean negative 视频文件路径。 |
| clean_negative_evidence_level | governance | none | true | true | false | clean negative 分数的证据层级; paper profile 要求 project_owned_clean_video_content_detector。 |
| clean_negative_status | governance | none | true | false | false | SSTW clean negative detector record 状态。 |
| clean_negative_failure_reason | governance | none | true | false | false | SSTW clean negative detector record 失败原因。 |
| sstw_formal_video_detector_positive_count | metric | none | true | true | false | SSTW measured_formal records 中具备正式视频 detector evidence 的 positive 数量。 |
| sstw_formal_video_detector_clean_negative_count | metric | none | true | true | false | SSTW measured_formal records 中具备正式视频 detector evidence 的 clean negative 数量。 |
| runtime_detection_status | governance | none | true | false | false | Per-record runtime detection status. |
| runtime_detection_failure_reason | governance | none | true | false | false | Per-record runtime detection failure reason. |
| attacked_video_decode_status | governance | none | true | false | false | Whether attacked video file can be decoded during runtime detection. |
| attacked_video_decode_failure_reason | governance | none | true | false | false | Decode failure reason for attacked video runtime detection. |
| attacked_video_detectable | metric | none | true | false | false | Whether attacked video produced a positive result under the formal SSTW video content detector. |
| attacked_video_decoded_frame_count | metric | none | true | false | false | Number of decoded frames observed by runtime detection runner. |
| clean_negative_video_decoded_frame_count | metric | none | true | false | false | official clean negative 视频写出后重新读取得到的帧数, 用于确认 clean negative 校准也经过文件级路径。 |
| source_to_attack_frame_ratio | metric | none | true | false | false | Ratio between attacked frame count and source frame count. |
| decoded_to_source_frame_ratio | metric | none | true | false | false | Ratio between decoded attacked frame count and source frame count. |
| attack_score_delta | metric | none | true | false | false | Runtime attacked video score delta against the source or baseline formal detector score when available; not required for fair TPR@FPR calibration. |
| S_runtime_attack_detection | metric | none | true | false | false | Runtime attacked video 的正式 SSTW 视频内容检测分数, 兼容旧表格字段名。 |

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
| replay_uncertainty_weight | metric | none | true | false | false | Weight derived from owner-side replay uncertainty diagnostics for replay boundary records. |
| replay_scheduler_id | governance | none | true | false | false | Scheduler identifier used by replay or sketch validation records. |
| replay_time_grid_id | governance | none | true | false | false | Time-grid identifier used by replay or sketch validation records. |
| wrong_prompt_replay_control | governance | none | true | false | false | Control label for wrong-prompt replay validation records. |
| replay_and_sketch_gate_decision | governance | none | true | true | false | Decision status for replay and authenticated sketch gate. |
| replay_and_sketch_evidence_level | governance | none | true | true | false | Evidence level for replay and authenticated sketch gate outputs; owner-side diagnostic values cannot support a strong Claim-3 unless claim3_full_support_allowed is true. |
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
| external_baseline_self_containment_ready_count | metric | none | true | true | false | probe_paper gate 重新审计到的 self-contained modern baseline 数量。 |
| external_baseline_self_containment_required_count | metric | none | true | true | false | probe_paper gate 要求 self-contained 的 modern baseline 数量。 |
| external_baseline_self_containment_gate_missing_requirements | governance | none | true | true | false | probe_paper gate 对 self-containment artifact 复查得到的缺口列表。 |
| self_contained_modern_external_baseline_count | metric | none | true | true | false | 已完成项目内 official bundle 执行闭环的现代 external baseline 数量。 |
| missing_self_contained_modern_external_baseline_names | governance | none | true | false | false | 尚未完成项目内 official bundle 执行闭环的现代 external baseline 名称列表。 |
| missing_self_containment_requirements | governance | none | true | false | false | External baseline self-containment 阶段仍缺失的要求列表。 |
| source_clone_ready | governance | none | true | false | false | External baseline self-containment 行中表示当前 checkout 或 clone manifest 是否已经提供源码克隆证据。 |
| repository_generated_official_bundle_ready | governance | none | true | false | false | External baseline self-containment 行中表示项目内生成的 official bundle 与 execution manifest 是否足以证明 clone/build/run 证据。 |
| missing_repository_generated_official_bundle_modern_external_baseline_names | governance | none | true | false | false | External baseline self-containment 中缺少项目内 official bundle 执行闭环的现代 baseline 名称列表。 |
| official_bundle_anchor_ready | governance | none | true | false | false | 单个 baseline self-containment 行中表示 official bundle payload 是否与 measured_formal record 的 prompt_id、seed_id 和 attack_name 完全一致。 |
| official_bundle_anchor_ready_count | metric | none | true | false | false | 单个 baseline self-containment 行中通过 official bundle prompt / seed / attack anchor 一致性校验的 record 数量。 |
| missing_official_bundle_anchor_modern_external_baseline_names | governance | none | true | false | false | External baseline self-containment 中 official bundle anchor 与 measured_formal record anchor 不一致或缺失的现代 baseline 名称列表。 |
| official_bundle_record_count | artifact | none | true | false | false | 单个 baseline self-containment 行中绑定的 official bundle record 数量。 |
| official_bundle_record_ok_count | artifact | none | true | false | false | 单个 baseline self-containment 行中通过 provenance 与 execution manifest 校验的 official bundle record 数量。 |
| official_execution_manifest_count | artifact | none | true | false | false | 单个 baseline self-containment 行中绑定的 official execution manifest 数量。 |
| official_execution_manifest_ok_count | artifact | none | true | false | false | 单个 baseline self-containment 行中通过项目内执行闭环校验的 official execution manifest 数量。 |
| materialized_official_bundle_path_count | artifact | none | true | false | false | 单个 baseline self-containment 行中实际落盘可读取的 official bundle 路径数量。 |
| materialized_official_execution_manifest_path_count | artifact | none | true | false | false | 单个 baseline self-containment 行中实际落盘可读取的 official execution manifest 路径数量。 |
| runtime_attack_coverage_ready | governance | none | true | false | false | 单个 baseline self-containment 行中表示该 baseline 的 measured_formal records 是否覆盖当前 profile 要求的全部 runtime attack。 |
| measured_runtime_attack_names | protocol | none | true | false | false | 单个 baseline self-containment 行中 measured_formal records 实际覆盖的 runtime attack 名称集合。 |
| missing_runtime_attack_names | governance | none | true | false | false | 单个 baseline self-containment 行中相对 required_runtime_attack_names 缺失的 runtime attack 名称集合。 |
| missing_runtime_attack_count | metric | none | true | false | false | 单个 baseline self-containment 行中相对 required_runtime_attack_names 缺失的 runtime attack 数量。 |
| missing_runtime_attack_coverage_modern_external_baseline_names | governance | none | true | false | false | External baseline self-containment 中未覆盖当前 profile 必需 runtime attack 的现代 baseline 名称列表。 |

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
| require_sstw_measured_formal_records | protocol | none | true | false | false | probe_paper gate 是否要求 SSTW 本方法 measured_formal records 已落盘。 |
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
| require_formal_method_baseline_comparison | protocol | none | true | false | false | probe_paper gate 是否要求 SSTW 与 5 个主实验 baseline 的同协议 measured_formal 比较表。 |
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
| require_formal_baseline_difference_interval | protocol | none | true | false | false | probe_paper gate 是否要求 SSTW 相对 5 个 baseline 的差值置信区间报告。 |
| paper_profile_gate_missing_validation_requirements | governance | none | true | false | false | pilot_paper gate 读取到的 probe_paper gate 原始缺失要求列表。 |
| paper_profile_gate_missing_requirement_count | metric | none | true | false | false | pilot_paper gate 读取到的 probe_paper gate 原始缺失要求数量。 |
| paper_profile_gate_fairness_missing_requirements | governance | none | true | false | false | pilot_paper gate 复核 probe_paper 公平比较闭环时发现的缺失要求。 |
| probe_paper_fair_detection_calibration_ready_count | metric | none | true | true | false | probe_paper gate 中已通过 clean negative 公平校准的方法数量。 |
| probe_paper_formal_method_baseline_comparison_ready_count | metric | none | true | true | false | probe_paper gate 中同协议 method-baseline 比较已 ready 的方法数量。 |
| probe_paper_formal_baseline_difference_interval_ready_count | metric | none | true | true | false | probe_paper gate 中 SSTW 相对 baseline 差值区间已 ready 的 baseline 数量。 |
| require_probe_paper_sstw_advantage_claim_ready | protocol | none | true | false | false | 共享 gate 是否要求 SSTW 相对 5 个现代 baseline 的 target_fpr=0.1 优势证据 ready; probe_paper 正式配置必须为 true。 |
| probe_paper_sstw_advantage_claim_ready | governance | none | true | true | false | 共享 paper profile/probe gate 中 SSTW target_fpr=0.1 优势证据是否满足完整论文主张标准; 当前正式消费层为 probe_paper。 |
| probe_paper_sstw_advantage_ready_baseline_count | metric | none | true | true | false | 共享 paper profile/probe gate 中 SSTW 优势差值和置信区间已 ready 的现代 baseline 数量。 |
| probe_paper_sstw_advantage_missing_baseline_names | governance | none | true | false | false | 共享 paper profile/probe gate 中尚未满足 SSTW 优势证据标准的现代 baseline 名称集合。 |
| probe_paper_sstw_advantage_blocking_reasons | governance | none | true | false | false | 共享 paper profile/probe gate 阻断 SSTW target_fpr=0.1 完整优势主张的原因列表。 |
| probe_paper_sstw_advantage_claim_status | claim | none | true | true | false | 共享 paper profile/probe gate 对 SSTW target_fpr=0.1 完整优势主张的 claim 支撑状态; 当前只有 probe_paper 可以把它升级为论文闭合证据。 |
| minimum_sstw_advantage_baseline_count | protocol | none | true | false | false | 三个正式 profile 共同要求 SSTW 优势证据覆盖的现代 baseline 最小数量。 |
| minimum_sstw_tpr_at_target_fpr_difference | protocol | none | true | false | false | 三个正式 profile 共同要求 SSTW 相对 baseline 的 TPR@target FPR 差值下限。 |
| require_sstw_advantage_ci_lower_above_zero | protocol | none | true | false | false | 三个正式 profile 是否要求 SSTW 相对 baseline 的差值置信区间下界大于0。 |
| probe_paper_transition_claim_support_status | governance | none | true | false | false | 历史兼容字段: pilot_paper gate 读取到的 probe_paper -> pilot_paper 旧跳转 claim_support_status; 当前主链改用 probe_paper_transition_* 字段。 |
| probe_paper_transition_source_gate_passed | governance | none | true | false | false | 历史兼容字段: probe_paper -> pilot_paper 旧跳转记录中的 source gate 是否已通过。 |
| probe_paper_transition_missing_requirements | governance | none | true | false | false | 历史兼容字段: probe_paper -> pilot_paper 旧跳转记录中的原始缺失要求列表。 |
| probe_paper_transition_missing_requirement_count | metric | none | true | false | false | 历史兼容字段: probe_paper -> pilot_paper 旧跳转记录中的原始缺失要求数量。 |
| probe_paper_transition_allowed_next_result_profiles | governance | none | true | false | false | probe_paper 通过后允许进入的下一结果 profile 列表。 |
| probe_paper_transition_blocked_next_result_profiles | governance | none | true | false | false | probe_paper 通过后仍禁止跳转的后续结果 profile 列表。 |
| probe_paper_transition_fairness_missing_requirements | governance | none | true | false | false | pilot_paper gate 复核 probe_paper 跳转判定完整性时发现的缺失要求。 |
| probe_paper_formal_internal_ablation_decision | governance | none | true | true | false | probe_paper 级 formal-compatible 内部消融汇总判定。 |
| formal_internal_ablation_evidence_level | governance | none | true | true | false | 内部消融汇总中某个变体的证据层级。 |
| formal_internal_ablation_score | metric | none | true | true | false | 单条正式内部消融记录的 SSTW 视频内容检测分数。 |
| formal_internal_ablation_score_semantics | protocol | none | true | false | false | 单条正式内部消融分数的语义, 当前为 sstw_key_conditioned_video_content_detector_score。 |
| formal_internal_ablation_source_record_family | artifact | none | true | false | false | 内部消融汇总行所使用的源 records 家族。 |
| formal_internal_ablation_record_count | metric | none | true | false | false | 内部消融汇总行聚合的源分数数量。 |
| formal_internal_ablation_score_mean | metric | none | true | true | false | 内部消融汇总行的平均分数。 |
| formal_internal_ablation_full_method_score_mean | metric | none | true | true | false | 内部消融汇总中 full-method measured_formal 平均分数。 |
| formal_internal_ablation_delta_vs_full_method | metric | none | true | true | false | 内部消融变体相对 full-method 平均分数的差值。 |
| formal_internal_ablation_variant_count | metric | none | true | true | false | probe_paper 级内部消融已覆盖变体数量。 |
| formal_internal_ablation_expected_variant_count | metric | none | true | false | false | probe_paper 级内部消融应覆盖变体数量。 |
| formal_internal_ablation_full_method_formal_ready | governance | none | true | true | false | full-method 行是否来自 SSTW measured_formal 结果。 |
| formal_internal_ablation_missing_variants | governance | none | true | true | false | probe_paper 级内部消融缺失的变体列表。 |
| formal_internal_ablation_missing_variant_count | metric | none | true | true | false | probe_paper 级内部消融缺失的变体数量。 |
| require_probe_paper_formal_internal_ablation | protocol | none | true | false | false | probe_paper gate 是否要求 formal-compatible 内部消融汇总。 |
| low_fpr_formal_statistics_decision | governance | none | true | true | false | 低 FPR 正式统计阻断记录阶段判定。 |
| current_target_fpr | protocol | none | true | false | false | 当前 probe_paper profile 使用的 target_fpr。 |
| blocked_result_profile | protocol | none | true | false | false | 当前阻断记录所对应的更高层级结果 profile。 |
| low_fpr_formal_statistics_status | governance | none | true | true | false | 低 FPR 正式统计状态或阻断状态。 |
| formal_low_fpr_claim_allowed | governance | none | true | true | false | 当前阶段是否允许低 FPR 正式 claim。 |
| observed_negative_event_count | metric | none | true | false | false | 当前 run_root 中可用于低 FPR 的 negative event 数量估计。 |
| threshold_protocol_required | protocol | none | true | false | false | 低 FPR 正式统计所需阈值协议。 |
| low_fpr_blocking_reason | governance | none | true | false | false | 低 FPR 正式统计被阻断的原因说明。 |
| low_fpr_formal_statistics_record_count | metric | none | true | true | false | 低 FPR 正式统计阻断 records 数量。 |
| low_fpr_blocked_target_fprs | governance | none | true | true | false | 当前阻断记录覆盖的低 FPR 目标列表。 |
| require_low_fpr_formal_statistics_blocking_record | protocol | none | true | false | false | probe_paper gate 是否要求低 FPR 正式统计阻断记录。 |
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
| require_motion_consistency_exclusion_report | protocol | none | true | false | false | probe_paper gate 是否要求 motion consistency 阻断样本处理报告。 |
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
| external_baseline_clean_negative_payload_bit_accuracy | metric | none | true | false | false | baseline clean negative 视频的 payload bit accuracy 辅助分数, 用于审计与主分数口径是否一致。 |
| external_baseline_clean_negative_video_path | artifact | none | true | false | false | baseline 自身 clean negative 视频路径, 用于审计阈值校准来源。 |
| clean_negative_ready | governance | none | true | true | false | self-containment 行中 measured_formal baseline 是否已经携带 clean negative 分数证据。 |
| clean_negative_ready_count | metric | none | true | true | false | self-containment 行中具备 clean negative 分数证据的 measured_formal record 数量。 |
| missing_clean_negative_modern_external_baseline_names | governance | none | true | false | false | self-containment 判定中缺少 clean negative 分数证据的现代 baseline 名称。 |
| official_clean_negative_source_video_path | artifact | none | true | false | false | official runtime 中未施加 runtime attack 的 baseline clean negative 视频路径。 |
| official_clean_negative_frame_array_path | artifact | none | true | false | false | VidSig official attack.py 读取的 clean negative 帧数组路径。 |
| official_clean_negative_attack_log_path | artifact | none | true | false | false | official detector 在 clean negative 视频上的日志路径。 |
| official_clean_negative_attack_stdout_path | artifact | none | true | false | false | official clean negative 检测命令 stdout 证据路径。 |
| official_clean_negative_attack_stderr_path | artifact | none | true | false | false | official clean negative 检测命令 stderr 证据路径。 |
| official_vidsig_tpr_at_fpr_1e_2 | metric | none | true | false | false | VidSig 官方固定 FPR=1e-2 日志中的 TPR 辅助值, 不作为公平比较主分数。 |
| official_clean_negative_vidsig_tpr_at_fpr_1e_2 | metric | none | true | false | false | VidSig clean negative 官方固定 FPR=1e-2 日志辅助值, 不替代 clean negative bit accuracy 校准。 |
| official_result_key | protocol | none | true | false | false | 官方输出文件中与当前 prompt / seed comparison unit 对应的 result key。 |
| runtime_attack_names | protocol | none | true | false | false | 运行器或补丁 manifest 中声明的 runtime attack 名称集合, 用于审计 baseline 是否覆盖同一攻击协议。 |
| runtime_attack_patch_statuses | governance | none | true | false | false | runtime attack 适配补丁各子步骤的状态列表, 用于定位 helper、attack list 和调用替换是否完成。 |
| official_clean_negative_result_key | protocol | none | true | false | false | clean negative 官方输出文件中与当前 prompt / seed calibration unit 对应的 result key。 |
| official_clean_negative_output_path | artifact | none | true | false | false | 自动生成 clean negative 官方产物的输出目录。 |
| official_clean_negative_score_assignment_policy | protocol | none | true | false | false | 聚合型 official clean negative 分数如何映射到 comparison unit。 |
| official_score_assignment_policy | protocol | none | true | false | false | 正样本 official score 如何映射到当前 prompt / seed / attack comparison unit。 |
| official_score_granularity | protocol | none | true | false | false | official 输出中正样本分数的样本粒度, 用于区分 per_prompt_seed_attack、per_prompt_seed、aggregate 或 binary-only 口径。 |
| official_score_value_type | protocol | none | true | false | false | official 输出中正样本分数的值类型, 如 continuous_detector_score、payload_bit_accuracy_score、fixed_fpr_detection_score 或 binary_decision。 |
| official_score_formal_comparison_eligibility | governance | none | true | false | false | official 正样本分数是否具备进入同 FPR 公平比较的资格。 |
| official_score_formal_comparison_block_reason | governance | none | true | false | false | official 正样本分数不能进入正式公平比较时的阻断原因。 |
| official_clean_negative_score_granularity | protocol | none | true | false | false | official clean negative 分数的样本粒度, aggregate clean negative 不得用于 target FPR 校准。 |
| official_clean_negative_score_value_type | protocol | none | true | false | false | official clean negative 分数的值类型, 必须与可阈值化分数口径兼容。 |
| official_clean_negative_score_formal_comparison_eligibility | governance | none | true | false | false | official clean negative 分数是否具备进入 target FPR 校准的资格。 |
| official_clean_negative_score_formal_comparison_block_reason | governance | none | true | false | false | official clean negative 分数不能进入 target FPR 校准时的阻断原因。 |
| external_baseline_official_score_granularity | protocol | none | true | false | false | 写入 measured_formal record 的 external baseline 正样本官方分数粒度。 |
| external_baseline_official_score_value_type | protocol | none | true | false | false | 写入 measured_formal record 的 external baseline 正样本官方分数值类型。 |
| external_baseline_official_score_formal_comparison_eligibility | governance | none | true | true | false | external baseline 正样本分数是否可进入 probe_paper 同协议公平比较。 |
| external_baseline_official_score_formal_comparison_block_reason | governance | none | true | true | false | external baseline 正样本分数被公平比较门禁阻断的原因。 |
| external_baseline_official_clean_negative_score_granularity | protocol | none | true | false | false | 写入 measured_formal record 的 external baseline clean negative 官方分数粒度。 |
| external_baseline_official_clean_negative_score_value_type | protocol | none | true | false | false | 写入 measured_formal record 的 external baseline clean negative 官方分数值类型。 |
| external_baseline_official_clean_negative_score_formal_comparison_eligibility | governance | none | true | true | false | external baseline clean negative 分数是否可进入 target FPR 阈值校准。 |
| external_baseline_official_clean_negative_score_formal_comparison_block_reason | governance | none | true | true | false | external baseline clean negative 分数被校准门禁阻断的原因。 |
| official_detection_logic | protocol | none | true | false | false | 官方 wrapper 或 official runtime 记录的 detector / extractor 分数计算逻辑。 |
| official_attacked_video_io_backend | protocol | none | true | false | false | official attacked 视频写出后重新读取并用于检测的 I/O 后端。 |
| official_clean_negative_video_io_backend | protocol | none | true | false | false | official clean negative 视频读取所使用的 I/O 后端。 |
| official_checkpoint_path | artifact | none | true | false | false | official runtime 实际使用的公开 checkpoint 路径, 用于审计资源来源, 不单独支撑 measured_formal。 |
| official_video_frame_count | metric | none | true | false | false | official runtime 实际纳入嵌入或检测的视频帧数量。 |
| official_video_frame_size | protocol | none | true | false | false | official runtime 为适配官方模型而使用的帧空间尺寸。 |
| official_frame_adapter_policy | protocol | none | true | false | false | 图像水印逐帧适配视频 baseline 的帧级嵌入和视频级聚合策略。 |
| official_payload_message_digest | provenance | none | true | false | false | official runtime 使用的 deterministic payload message digest, 用于复现实验而不暴露原始随机消息。 |
| external_baseline_clean_negative_confidence | metric | none | true | false | false | baseline clean negative 检测时的辅助 confidence 均值, 不替代 clean negative 主分数。 |
| source_sstw_video_path | artifact | none | true | false | false | official bundle 记录的原始 SSTW source video 路径, 用于审计同一 prompt / seed 输入来源。 |
| sstw_attacked_video_path | artifact | none | true | false | false | official bundle 记录的 SSTW runtime attacked video 路径, 用于审计同一 attack anchor。 |
| generate_clean_negative_reference | protocol | none | true | false | false | baseline 官方运行器是否自动生成 clean negative reference 并抽取官方分数。 |
| clean_negative_output_path | protocol | none | true | false | false | baseline 官方运行器 clean negative 输出目录配置。 |
| clean_negative_attack_transform | protocol | none | true | false | false | clean negative 视频施加的 runtime attack 变换描述。 |
| clean_negative_attack_strength | protocol | none | true | false | false | clean negative 视频施加的 runtime attack 强度描述。 |
| fair_comparison_protocol | protocol | none | true | false | false | 公平比较协议名称, 当前为 method-specific clean negative calibration 到统一 target FPR。 |
| require_fair_detection_calibration | protocol | none | true | false | false | probe_paper gate 是否要求 clean negative 公平校准通过。 |
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
| calibration_clean_negative_score_count | metric | none | true | true | false | calibration split 中用于冻结方法阈值的 clean negative 分数数量。 |
| heldout_clean_negative_score_count | metric | none | true | true | false | held-out test split 中用于报告冻结阈值 FPR 的 clean negative 分数数量。 |
| heldout_attacked_positive_score_count | metric | none | true | true | false | held-out test split 中用于报告 TPR@target FPR 的 attacked positive 分数数量。 |
| calibration_fpr_at_calibrated_threshold | metric | none | true | false | false | 方法特定冻结阈值在 calibration split 上的经验 FPR。 |
| attacked_positive_score_count | metric | none | true | true | false | attacked positive 检测分数数量。 |
| positive_anchor_count | metric | none | true | true | false | fair calibration 中 attacked positive 的 prompt / seed / attack anchor 数量。 |
| positive_anchor_missing_count | metric | none | true | true | false | fair calibration 中带分数但缺少 prompt_id、seed_id 或 attack_name 的 attacked positive 记录数量。 |
| positive_formal_evidence_missing_count | metric | none | true | true | false | fair calibration 中带分数但缺少 official evidence、clean negative 或完整 anchor 的 formal attacked positive 记录数量。 |
| negative_formal_evidence_missing_count | metric | none | true | true | false | fair calibration 中带 clean negative 分数但缺少 official evidence 或官方分数抽取证据的记录数量。 |
| positive_anchor_keys | protocol | none | true | false | false | fair calibration 中 attacked positive 的规范 prompt / seed / attack anchor 键集合。 |
| positive_attack_names | protocol | none | true | false | false | fair calibration 中 attacked positive records 实际覆盖的 runtime attack 名称集合。 |
| shared_attack_protocol_config_path | protocol | none | true | false | false | probe_paper、pilot_paper 和 full_paper 共同引用的 runtime / non-runtime attack 协议配置路径。 |
| shared_attack_protocol_id | protocol | none | true | false | false | 共享 attack 协议配置的稳定语义标识。 |
| shared_attack_protocol_resolved_path | provenance | none | true | false | false | 运行时解析得到的共享 attack 协议配置实际路径。 |
| shared_attack_protocol_resolution_status | governance | none | true | false | false | 共享 attack 协议配置是否已合并到当前 profile config 的状态。 |
| required_runtime_attack_names | protocol | none | true | false | false | 当前 workflow profile 要求必须覆盖的 runtime attack 名称集合。 |
| required_runtime_attack_count | metric | none | true | false | false | 当前 workflow profile 要求必须覆盖的 runtime attack 数量。 |
| runtime_attack_family_minimums | protocol | none | true | false | false | 当前 workflow profile 对 compression、temporal、spatial、visual 和 combined 等 attack family 的最低覆盖要求。 |
| runtime_attack_family_counts | metric | none | true | false | false | 当前 protocol config 或已落盘 records 覆盖到的 runtime attack family 计数。 |
| runtime_attack_missing_family_minimums | governance | none | true | false | false | 当前 runtime attack manifest 未满足的 family 最低覆盖要求列表。 |
| runtime_attack_protocol_decision | governance | none | true | true | false | 当前 runtime attack manifest 是否满足 profile 分层协议要求。 |
| required_non_runtime_attack_protocols | protocol | none | true | false | false | full_paper 还必须覆盖的非 runtime 自适应攻击或生成式重压缩协议名称集合。 |
| minimum_non_runtime_attack_protocol_count | protocol | none | true | false | false | probe_paper、pilot_paper 或 full_paper gate 要求覆盖的非 runtime / adaptive 协议最小数量。 |
| non_runtime_attack_protocol | protocol | none | true | false | false | 单条 adaptive attack record 映射到论文协议中的非 runtime / adaptive 攻击名称。 |
| non_runtime_attack_protocol_count | metric | none | true | true | false | 当前 run_root 中 adaptive attack records 已覆盖的非 runtime / adaptive 协议数量。 |
| observed_non_runtime_attack_protocols | governance | none | true | false | false | 当前 run_root 中实际观测到的非 runtime / adaptive 协议名称集合。 |
| missing_non_runtime_attack_protocols | governance | none | true | false | false | full_paper protocol config 尚未登记的非 runtime 攻击协议名称集合。 |
| adaptive_attack_missing_non_runtime_protocols | governance | none | true | false | false | probe_paper gate 从 adaptive attack records 中发现的缺失非 runtime / adaptive 协议集合。 |
| adaptive_attack_missing_non_runtime_protocol_count | metric | none | true | true | false | probe_paper gate 从 adaptive attack records 中发现的缺失非 runtime / adaptive 协议数量。 |
| top_tier_attack_protocol_status | governance | none | true | false | false | protocol config 对顶会顶刊级 attack 覆盖的摘要状态。 |
| required_runtime_attack_protocol_note | protocol | none | true | false | false | 解释 probe_paper、pilot_paper 和 full_paper 分层 attack 协议差异的配置说明。 |
| target_fpr_levels | protocol | none | true | false | false | 同构论文协议族登记的 FPR 等级集合, 用于 probe_paper、pilot_paper 和 full_paper 切换。 |
| missing_required_runtime_attack_names | governance | none | true | false | false | 当前 record 相对 required_runtime_attack_names 缺失的 runtime attack 名称集合。 |
| missing_required_runtime_attack_count | metric | none | true | true | false | 当前 record 相对 required_runtime_attack_names 缺失的 runtime attack 数量。 |
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
| comparison_anchor_keys | protocol | none | true | false | false | formal method baseline comparison 行使用的 prompt / seed / attack anchor 键集合。 |
| comparison_attack_names | protocol | none | true | false | false | formal method baseline comparison 行实际覆盖的 runtime attack 名称集合。 |
| reference_anchor_count | metric | none | true | true | false | 参考方法 SSTW 的 prompt / seed / attack anchor 数量。 |
| baseline_anchor_count | metric | none | true | true | false | baseline 方法的 prompt / seed / attack anchor 数量。 |
| missing_reference_anchor_count | metric | none | true | true | false | baseline 缺少的 SSTW 参考 anchor 数量。 |
| extra_anchor_count | metric | none | true | true | false | baseline 相比 SSTW 参考 anchor 多出的 anchor 数量。 |
| unpaired_reference_anchor_count | metric | none | true | true | false | 差值区间中无法与 baseline 配对的 SSTW anchor 数量。 |
| unpaired_baseline_anchor_count | metric | none | true | true | false | 差值区间中无法与 SSTW 配对的 baseline anchor 数量。 |
| paired_comparison_anchor_keys | protocol | none | true | false | false | 差值区间中 SSTW 与 baseline 完全配对的 prompt / seed / attack anchor 键集合。 |
| paired_attack_names | protocol | none | true | false | false | 差值区间中完全配对比较单元覆盖的 runtime attack 名称集合。 |
| comparison_anchor_alignment_status | governance | none | true | true | false | 当前方法是否与 SSTW 使用同一 prompt / seed / attack anchor 集合。 |
| reference_source_fair_detection_target_fpr | protocol | none | true | true | false | 差值区间中参考方法上游 fair_detection_calibration record 的 target_fpr。 |
| baseline_source_fair_detection_target_fpr | protocol | none | true | true | false | 差值区间中 baseline 上游 fair_detection_calibration record 的 target_fpr。 |
| reference_tpr_at_target_fpr | metric | none | true | true | false | 差值区间中参考方法的 TPR@target FPR。 |
| baseline_tpr_at_target_fpr | metric | none | true | true | false | 差值区间中 baseline 的 TPR@target FPR。 |
| tpr_at_target_fpr_difference | metric | none | true | true | false | SSTW TPR@target FPR 减 baseline TPR@target FPR 的差值。 |
| baseline_id | protocol | none | true | false | false | modern external baseline 的稳定身份标识。 |
| git_short_commit | artifact | none | true | false | false | Notebook 计时 manifest 记录的当前 Git 短 commit。 |
| notebook_run_id | protocol | none | true | false | false | 单次 Notebook repository stage plan 运行的稳定标识。 |
| notebook_started_at_utc | artifact | none | true | false | false | Notebook repository stage plan 计时开始 UTC 时间。 |
| notebook_finished_at_utc | artifact | none | true | false | false | Notebook repository stage plan 计时结束 UTC 时间。 |
| notebook_elapsed_sec | metric | none | true | false | false | Notebook repository stage plan 总耗时秒数, 用于运行成本估算。 |
| notebook_elapsed_min | metric | none | true | false | false | Notebook repository stage plan 总耗时分钟数, 用于运行成本估算。 |
| notebook_timing_status | governance | none | true | false | false | Notebook 计时状态, 例如 running、completed 或 completed_before_stage_package_publish。 |
| notebook_timing_scope | protocol | none | true | false | false | Notebook 计时覆盖范围, 例如 repository_stage_plan 或 external_baseline_formal_reference_plan。 |
| notebook_timing_coverage_status | governance | none | true | false | false | Notebook 计时覆盖说明, 明确是否排除人工 Colab 设置或阶段包发布耗时。 |
| notebook_run_timing_manifest_path | artifact | none | true | false | false | Notebook 总耗时 manifest 路径。 |
| notebook_run_timing_manifest | artifact | none | true | false | false | Notebook 总耗时 manifest 的内联摘要。 |
| notebook_stage_timing_record_count | metric | none | true | false | false | Notebook 阶段耗时记录条数。 |
| notebook_stage_timing_records_path | artifact | none | true | false | false | Notebook 阶段耗时 JSONL 路径。 |
| stage_name | protocol | none | true | false | false | Notebook stage plan 中单个阶段的语义名称。 |
| stage_execution_kind | protocol | none | true | false | false | 阶段执行方式, 例如 command 或 python_helper。 |
| stage_execution_status | governance | none | true | false | false | 阶段计时层归一化后的执行状态。 |
| stage_started_at_utc | artifact | none | true | false | false | 单个 Notebook 阶段开始 UTC 时间。 |
| stage_finished_at_utc | artifact | none | true | false | false | 单个 Notebook 阶段结束 UTC 时间。 |
| stage_elapsed_sec | metric | none | true | false | false | 单个 Notebook 阶段耗时秒数。 |
| stage_elapsed_min | metric | none | true | false | false | 单个 Notebook 阶段耗时分钟数。 |
| stage_failure_type | governance | none | true | false | false | 阶段失败时记录的异常类型。 |
| stage_failure_message | governance | none | true | false | false | 阶段失败时记录的异常摘要。 |
| stage_package_publish_timing_policy | governance | none | true | false | false | 阶段包发布耗时与 Notebook 总耗时之间的记录策略说明。 |
| stage_package_publish_included_in_notebook_elapsed | governance | none | true | false | false | 阶段包发布耗时是否计入 notebook_elapsed_sec。 |
| completed_stage_count | metric | none | true | false | false | Notebook 计时 manifest 中已完成阶段数量。 |
| failed_stage_count | metric | none | true | false | false | Notebook 计时 manifest 中失败阶段数量。 |
| notebook_timing_start_source | governance | none | true | false | false | Notebook 总耗时计时起点来源, 例如第一格环境变量或 helper 初始化。 |
| stage_package_publish_elapsed_sec | metric | none | true | false | false | publish_colab_stage_package 从开始到写出远端 zip 和 manifest 的耗时秒数。 |
| notebook_runtime_report_path | artifact | none | true | false | false | 每个阶段结果包必须包含的 notebook_runtime_report.json 路径。 |
| notebook_runtime_started_at_utc | artifact | none | true | false | false | 共享 Colab layout 入口层记录的 Notebook 运行时间起点。 |
| notebook_runtime_start_perf_counter | metric | none | true | false | false | 共享 Colab layout 入口层记录的 Python 单调时钟起点。 |
| notebook_runtime_start_source | governance | none | true | false | false | Notebook runtime 起点来源, 当前为 shared_colab_stage_layout。 |
| notebook_runtime_timing_scope | protocol | none | true | false | false | 共享入口层 runtime 计时覆盖范围。 |
| notebook_runtime_workflow_profile | protocol | none | true | false | false | runtime session 初始化时绑定的 workflow profile。 |
| notebook_runtime_notebook_role | protocol | none | true | false | false | runtime session 初始化时绑定的 Notebook role。 |
| notebook_runtime_baseline_id | protocol | none | true | false | false | runtime session 初始化时绑定的 external baseline 身份。 |
| notebook_runtime_repo_root | artifact | none | true | false | false | runtime session 初始化时记录的仓库根目录。 |
| notebook_runtime_report | artifact | none | true | false | false | Notebook 运行时间报告的内联摘要对象。 |
| formal_comparison_external_baseline_environment_decision | governance | none | true | false | false | formal comparison scoring 阶段 external baseline 环境预检判定。 |
| formal_comparison_external_baseline_environment_status | governance | none | true | false | false | formal comparison scoring 阶段 external baseline 环境预检状态说明。 |

| probe_paper_gate_decision | governance | none | true | true | false | probe_paper 小样本 fpr=0.1 论文闭合门禁判定; PASS 后必须再通过 probe_paper_to_pilot_paper_transition_decision 才能进入 pilot_paper。 |
| probe_paper_to_pilot_paper_transition_decision | governance | none | true | true | false | probe_paper -> pilot_paper 的轻量跳转判定, 防止未通过显式 transition 就进入 pilot_paper。 |
| probe_paper_claim_support_status | claim | none | true | false | false | pilot_paper gate 读取到的 probe_paper claim 支撑状态。 |
| probe_paper_gate_fairness_missing_requirements | governance | none | true | false | false | pilot_paper gate 复核 probe_paper gate 时发现的缺失或不合规项。 |
| probe_paper_transition_fairness_missing_requirements | governance | none | true | false | false | pilot_paper gate 复核 probe_paper -> pilot_paper 跳转时发现的缺失或不合规项。 |

## 完整论文机制新增字段

下列字段由真实 Flow velocity、endpoint、path、replay、HMAC sketch 与固定 FPR 检测链路写出。

| field_name | category | required_suffix | allowed_in_records | allowed_in_claims | replacement_required | description |
| --- | --- | --- | --- | --- | --- | --- |
| admissibility_thresholds | metric | none | true | false | false | 完整论文机制中 `admissibility_thresholds` 的受治理记录字段。 |
| audited_profile_count | metric | none | true | false | false | 完整论文机制中 `audited_profile_count` 的受治理记录字段。 |
| calibration_negative_count | metric | none | true | false | false | 完整论文机制中 `calibration_negative_count` 的受治理记录字段。 |
| claim3_real_replay_record_count | metric | none | true | false | false | 完整论文机制中 `claim3_real_replay_record_count` 的受治理记录字段。 |
| claim_1_empirical_fpr | metric | none | true | true | false | 完整论文机制中 `claim_1_empirical_fpr` 的受治理记录字段。 |
| claim_1_heldout_negative_count | metric | none | true | true | false | 完整论文机制中 `claim_1_heldout_negative_count` 的受治理记录字段。 |
| claim_1_heldout_positive_count | metric | none | true | true | false | 完整论文机制中 `claim_1_heldout_positive_count` 的受治理记录字段。 |
| claim_1_tpr_at_target_fpr | metric | none | true | true | false | 完整论文机制中 `claim_1_tpr_at_target_fpr` 的受治理记录字段。 |
| claim_1_velocity_constraint_detectable_watermark_decision | governance | none | true | true | false | 完整论文机制中 `claim_1_velocity_constraint_detectable_watermark_decision` 的受治理记录字段。 |
| claim_2_paired_comparison_count | metric | none | true | true | false | 完整论文机制中 `claim_2_paired_comparison_count` 的受治理记录字段。 |
| claim_2_paired_detection_gain_mean | metric | none | true | true | false | 完整论文机制中 `claim_2_paired_detection_gain_mean` 的受治理记录字段。 |
| claim_2_paired_score_gain_ci_95_lower | metric | none | true | true | false | 完整论文机制中 `claim_2_paired_score_gain_ci_95_lower` 的受治理记录字段。 |
| claim_2_paired_score_gain_ci_95_upper | metric | none | true | true | false | 完整论文机制中 `claim_2_paired_score_gain_ci_95_upper` 的受治理记录字段。 |
| claim_2_paired_score_gain_mean | metric | none | true | true | false | 完整论文机制中 `claim_2_paired_score_gain_mean` 的受治理记录字段。 |
| claim_2_path_evidence_independent_gain_decision | governance | none | true | true | false | 完整论文机制中 `claim_2_path_evidence_independent_gain_decision` 的受治理记录字段。 |
| claim_3_attacked_video_replay_posterior_decision | governance | none | true | true | false | 完整论文机制中 `claim_3_attacked_video_replay_posterior_decision` 的受治理记录字段。 |
| clean_negative_score | metric | none | true | false | false | 完整论文机制中 `clean_negative_score` 的受治理记录字段。 |
| clean_negative_trial_index | protocol | none | true | false | false | 完整论文机制中 `clean_negative_trial_index` 的受治理记录字段。 |
| coverage | metric | none | true | false | false | 完整论文机制中 `coverage` 的受治理记录字段。 |
| detector_key_digest | protocol | none | true | false | false | 检测 key 的单向摘要, 不暴露原始 key。 |
| endpoint | protocol | none | true | false | false | 完整论文机制中 `endpoint` 的受治理记录字段。 |
| endpoint_control_enabled | protocol | none | true | false | false | 完整论文机制中 `endpoint_control_enabled` 的受治理记录字段。 |
| endpoint_control_multiplier | protocol | none | true | false | false | 完整论文机制中 `endpoint_control_multiplier` 的受治理记录字段。 |
| endpoint_response_before_step | protocol | none | true | false | false | 完整论文机制中 `endpoint_response_before_step` 的受治理记录字段。 |
| evidence_means | metric | none | true | false | false | 完整论文机制中 `evidence_means` 的受治理记录字段。 |
| evidence_standard_deviations | protocol | none | true | false | false | 完整论文机制中 `evidence_standard_deviations` 的受治理记录字段。 |
| flow_detector_score_source | metric | none | true | false | false | 完整论文机制中 `flow_detector_score_source` 的受治理记录字段。 |
| flow_endpoint_state | protocol | none | true | false | false | 完整论文机制中 `flow_endpoint_state` 的受治理记录字段。 |
| flow_key_direction_digest | protocol | none | true | false | false | 由 key、latent shape 与 tubelet 布局确定的稳定方向摘要。 |
| flow_key_direction_norm | metric | none | true | false | false | 完整论文机制中 `flow_key_direction_norm` 的受治理记录字段。 |
| flow_observation_variance | metric | none | true | false | false | 完整论文机制中 `flow_observation_variance` 的受治理记录字段。 |
| flow_path_consistency_state | protocol | none | true | false | false | 完整论文机制中 `flow_path_consistency_state` 的受治理记录字段。 |
| flow_payload_negative_count | metric | none | true | false | false | 完整论文机制中 `flow_payload_negative_count` 的受治理记录字段。 |
| flow_payload_positive_count | metric | none | true | false | false | 完整论文机制中 `flow_payload_positive_count` 的受治理记录字段。 |
| flow_phase | metric | none | true | false | false | 完整论文机制中 `flow_phase` 的受治理记录字段。 |
| flow_phase_state | metric | none | true | false | false | 完整论文机制中 `flow_phase_state` 的受治理记录字段。 |
| flow_phase_weight | metric | none | true | false | false | 完整论文机制中 `flow_phase_weight` 的受治理记录字段。 |
| flow_posterior_confidence | metric | none | true | false | false | 完整论文机制中 `flow_posterior_confidence` 的受治理记录字段。 |
| flow_process_variance | metric | none | true | false | false | 完整论文机制中 `flow_process_variance` 的受治理记录字段。 |
| flow_replay_reliability_state | metric | none | true | false | false | 完整论文机制中 `flow_replay_reliability_state` 的受治理记录字段。 |
| flow_state_admissibility_failures | protocol | none | true | false | false | 完整论文机制中 `flow_state_admissibility_failures` 的受治理记录字段。 |
| flow_state_posterior_entropy | metric | none | true | false | false | 完整论文机制中 `flow_state_posterior_entropy` 的受治理记录字段。 |
| flow_temporal_disturbance_state | protocol | none | true | false | false | 完整论文机制中 `flow_temporal_disturbance_state` 的受治理记录字段。 |
| flow_time_grid_reliability_state | metric | none | true | false | false | 完整论文机制中 `flow_time_grid_reliability_state` 的受治理记录字段。 |
| flow_tubelet_count | metric | none | true | false | false | 完整论文机制中 `flow_tubelet_count` 的受治理记录字段。 |
| flow_tubelet_key_code_status | governance | none | true | false | false | 完整论文机制中 `flow_tubelet_key_code_status` 的受治理记录字段。 |
| flow_tubelet_spatial_height | protocol | none | true | false | false | 完整论文机制中 `flow_tubelet_spatial_height` 的受治理记录字段。 |
| flow_tubelet_spatial_width | protocol | none | true | false | false | 完整论文机制中 `flow_tubelet_spatial_width` 的受治理记录字段。 |
| flow_tubelet_temporal_size | protocol | none | true | false | false | 完整论文机制中 `flow_tubelet_temporal_size` 的受治理记录字段。 |
| flow_velocity_consistency_state | protocol | none | true | false | false | 完整论文机制中 `flow_velocity_consistency_state` 的受治理记录字段。 |
| formal_flow_clean_negative_record_count | metric | none | true | false | false | 完整论文机制中 `formal_flow_clean_negative_record_count` 的受治理记录字段。 |
| formal_flow_detector_input_contract | protocol | none | true | false | false | 完整论文机制中 `formal_flow_detector_input_contract` 的受治理记录字段。 |
| formal_flow_evidence_decision | governance | none | true | false | false | 完整论文机制中 `formal_flow_evidence_decision` 的受治理记录字段。 |
| formal_flow_evidence_failure_reason | protocol | none | true | false | false | 完整论文机制中 `formal_flow_evidence_failure_reason` 的受治理记录字段。 |
| formal_flow_evidence_level | protocol | none | true | false | false | 完整论文机制中 `formal_flow_evidence_level` 的受治理记录字段。 |
| formal_flow_evidence_record_count | metric | none | true | false | false | 完整论文机制中 `formal_flow_evidence_record_count` 的受治理记录字段。 |
| formal_flow_evidence_status | governance | none | true | false | false | 完整论文机制中 `formal_flow_evidence_status` 的受治理记录字段。 |
| formal_flow_evidence_unit_id | protocol | none | true | false | false | 完整论文机制中 `formal_flow_evidence_unit_id` 的受治理记录字段。 |
| formal_flow_failure_record_count | metric | none | true | false | false | 完整论文机制中 `formal_flow_failure_record_count` 的受治理记录字段。 |
| formal_flow_missing_method_variants | protocol | none | true | false | false | 完整论文机制中 `formal_flow_missing_method_variants` 的受治理记录字段。 |
| formal_flow_observed_method_variants | protocol | none | true | false | false | 完整论文机制中 `formal_flow_observed_method_variants` 的受治理记录字段。 |
| formal_flow_positive_record_count | metric | none | true | false | false | 完整论文机制中 `formal_flow_positive_record_count` 的受治理记录字段。 |
| formal_flow_threshold_record_count | metric | none | true | false | false | 完整论文机制中 `formal_flow_threshold_record_count` 的受治理记录字段。 |
| formal_mechanism_contract_id | protocol | none | true | false | false | 三个 paper profile 共同绑定的完整论文机制契约标识。 |
| frozen_final_score_threshold | metric | none | true | false | false | 完整论文机制中 `frozen_final_score_threshold` 的受治理记录字段。 |
| generation_nonce_random | metric | _random | true | false | false | 生成时为防止 trajectory sketch 重放而产生的随机 nonce。 |
| model_signature | protocol | none | true | false | false | 完整论文机制中 `model_signature` 的受治理记录字段。 |
| paired_endpoint_only_detector_decision | governance | none | true | false | false | 历史 Claim-2 原型字段, 正式机制不再使用; endpoint-only 仍仅作为生成级说明性对照。 |
| paired_endpoint_only_detector_score | metric | none | true | false | false | 历史 Claim-2 原型字段, 不得支持路径独立增益主张。 |
| paired_full_detector_decision | governance | none | true | false | false | 完整论文机制中 `paired_full_detector_decision` 的受治理记录字段。 |
| paired_full_detector_score | metric | none | true | false | false | 完整论文机制中 `paired_full_detector_score` 的受治理记录字段。 |
| paired_path_evidence_detection_gain | metric | none | true | true | false | 完整论文机制中 `paired_path_evidence_detection_gain` 的受治理记录字段。 |
| paired_path_evidence_score_gain | metric | none | true | true | false | 完整论文机制中 `paired_path_evidence_score_gain` 的受治理记录字段。 |
| paired_source_method_variant | protocol | none | true | false | false | 完整论文机制中 `paired_source_method_variant` 的受治理记录字段。 |
| paper_mechanism_contract_decision | governance | none | true | false | false | paper profile 机制一致性审计判定。 |
| paper_mechanism_contract_violations | governance | none | true | false | false | 完整论文机制中 `paper_mechanism_contract_violations` 的受治理记录字段。 |
| path | protocol | none | true | false | false | 完整论文机制中 `path` 的受治理记录字段。 |
| path_endpoint_consistency | protocol | none | true | false | false | 同一 key 方向上 replay 路径证据与 endpoint 证据的一致性。 |
| path_observation_step_count | metric | none | true | false | false | 完整论文机制中 `path_observation_step_count` 的受治理记录字段。 |
| path_projection | metric | none | true | false | false | 完整论文机制中 `path_projection` 的受治理记录字段。 |
| path_projection_normalized | metric | none | true | false | false | 完整论文机制中 `path_projection_normalized` 的受治理记录字段。 |
| path_step_norm | metric | none | true | false | false | 完整论文机制中 `path_step_norm` 的受治理记录字段。 |
| path_velocity_consistency | protocol | none | true | false | false | 完整论文机制中 `path_velocity_consistency` 的受治理记录字段。 |
| path_velocity_consistency_mean | metric | none | true | false | false | 完整论文机制中 `path_velocity_consistency_mean` 的受治理记录字段。 |
| posterior_confidence | metric | none | true | false | false | 完整论文机制中 `posterior_confidence` 的受治理记录字段。 |
| prompt_digest | protocol | none | true | false | false | 完整论文机制中 `prompt_digest` 的受治理记录字段。 |
| protocol_split | protocol | none | true | false | false | 完整论文机制中 `protocol_split` 的受治理记录字段。 |
| replay_control_execution_status | governance | none | true | false | false | 完整论文机制中 `replay_control_execution_status` 的受治理记录字段。 |
| replay_cycle_error_maximum | metric | none | true | false | false | 完整论文机制中 `replay_cycle_error_maximum` 的受治理记录字段。 |
| replay_cycle_error_mean | metric | none | true | false | false | 完整论文机制中 `replay_cycle_error_mean` 的受治理记录字段。 |
| replay_endpoint_ensemble_variance | metric | none | true | false | false | 完整论文机制中 `replay_endpoint_ensemble_variance` 的受治理记录字段。 |
| replay_ensemble_count | metric | none | true | false | false | 完整论文机制中 `replay_ensemble_count` 的受治理记录字段。 |
| replay_inversion_status | governance | none | true | false | false | 完整论文机制中 `replay_inversion_status` 的受治理记录字段。 |
| replay_primary_step_count | metric | none | true | false | false | 完整论文机制中 `replay_primary_step_count` 的受治理记录字段。 |
| replay_reliability | metric | none | true | false | false | 完整论文机制中 `replay_reliability` 的受治理记录字段。 |
| replay_reliability_weight | metric | none | true | false | false | 完整论文机制中 `replay_reliability_weight` 的受治理记录字段。 |
| replay_step_counts | metric | none | true | false | false | 完整论文机制中 `replay_step_counts` 的受治理记录字段。 |
| replay_trajectory_source | protocol | none | true | false | false | 完整论文机制中 `replay_trajectory_source` 的受治理记录字段。 |
| require_claim3_full_support | governance | none | true | false | false | 当前 paper profile 是否强制要求攻击后视频 replay 后验的完整证据。 |
| required_claim_ids | governance | none | true | false | false | 完整论文机制中 `required_claim_ids` 的受治理记录字段。 |
| required_result_capabilities | governance | none | true | false | false | 完整论文机制中 `required_result_capabilities` 的受治理记录字段。 |
| sampler_signature | protocol | none | true | false | false | 完整论文机制中 `sampler_signature` 的受治理记录字段。 |
| scheduler_velocity_sign | protocol | none | true | false | false | 完整论文机制中 `scheduler_velocity_sign` 的受治理记录字段。 |
| stage_id | protocol | none | true | false | false | 完整论文机制中 `stage_id` 的受治理记录字段。 |
| three_layer_mechanism_pre_replay_decision | governance | none | true | false | false | Claim-1 与 Claim-2 在 Claim-3 认证门禁前的合并判定。 |
| time_grid_id | protocol | none | true | false | false | 完整论文机制中 `time_grid_id` 的受治理记录字段。 |
| time_grid_reliability | metric | none | true | false | false | 完整论文机制中 `time_grid_reliability` 的受治理记录字段。 |
| trajectory_sketch_authentication_algorithm | protocol | none | true | false | false | 完整论文机制中 `trajectory_sketch_authentication_algorithm` 的受治理记录字段。 |
| trajectory_sketch_format | protocol | none | true | false | false | 完整论文机制中 `trajectory_sketch_format` 的受治理记录字段。 |
| trajectory_sketch_payload | protocol | none | true | false | false | 完整论文机制中 `trajectory_sketch_payload` 的受治理记录字段。 |
| trajectory_sketch_signature | protocol | none | true | false | false | trajectory sketch 的 HMAC-SHA256 认证标签。 |
| trajectory_step_count | metric | none | true | false | false | 完整论文机制中 `trajectory_step_count` 的受治理记录字段。 |
| trajectory_steps | protocol | none | true | false | false | 完整论文机制中 `trajectory_steps` 的受治理记录字段。 |
| velocity | protocol | none | true | false | false | 完整论文机制中 `velocity` 的受治理记录字段。 |
| velocity_alignment_after_constraint | protocol | none | true | false | false | 完整论文机制中 `velocity_alignment_after_constraint` 的受治理记录字段。 |
| velocity_alignment_before_constraint | protocol | none | true | false | false | 完整论文机制中 `velocity_alignment_before_constraint` 的受治理记录字段。 |
| velocity_alignment_gain | metric | none | true | false | false | 完整论文机制中 `velocity_alignment_gain` 的受治理记录字段。 |
| velocity_constraint_delta_norm | metric | none | true | false | false | 完整论文机制中 `velocity_constraint_delta_norm` 的受治理记录字段。 |
| velocity_constraint_delta_ratio | metric | none | true | false | false | 完整论文机制中 `velocity_constraint_delta_ratio` 的受治理记录字段。 |
| velocity_constraint_lambda | protocol | none | true | false | false | 完整论文机制中 `velocity_constraint_lambda` 的受治理记录字段。 |
| velocity_field_constraint_status | governance | none | true | false | false | 完整论文机制中 `velocity_field_constraint_status` 的受治理记录字段。 |
| velocity_field_source | protocol | none | true | false | false | 完整论文机制中 `velocity_field_source` 的受治理记录字段。 |
| velocity_norm_after_constraint | metric | none | true | false | false | 完整论文机制中 `velocity_norm_after_constraint` 的受治理记录字段。 |
| velocity_norm_before_constraint | metric | none | true | false | false | 完整论文机制中 `velocity_norm_before_constraint` 的受治理记录字段。 |
| velocity_norm_ratio_budget | metric | none | true | false | false | 完整论文机制中 `velocity_norm_ratio_budget` 的受治理记录字段。 |
| velocity_projection_normalized | metric | none | true | false | false | 完整论文机制中 `velocity_projection_normalized` 的受治理记录字段。 |
| wrong_key_S_path_inv | protocol | none | true | false | false | 完整论文机制中 `wrong_key_S_path_inv` 的受治理记录字段。 |
| wrong_key_control_margin | metric | none | true | false | false | 完整论文机制中 `wrong_key_control_margin` 的受治理记录字段。 |
| wrong_key_endpoint_score | metric | none | true | false | false | 完整论文机制中 `wrong_key_endpoint_score` 的受治理记录字段。 |
| wrong_prompt_S_path_inv | protocol | none | true | false | false | 完整论文机制中 `wrong_prompt_S_path_inv` 的受治理记录字段。 |
| wrong_prompt_control_margin | metric | none | true | false | false | 完整论文机制中 `wrong_prompt_control_margin` 的受治理记录字段。 |
| wrong_prompt_control_prompt_digest | protocol | none | true | false | false | 完整论文机制中 `wrong_prompt_control_prompt_digest` 的受治理记录字段。 |
| wrong_prompt_replay_cycle_error | metric | none | true | false | false | 完整论文机制中 `wrong_prompt_replay_cycle_error` 的受治理记录字段。 |
| wrong_sampler_S_path_inv | protocol | none | true | false | false | 完整论文机制中 `wrong_sampler_S_path_inv` 的受治理记录字段。 |
| wrong_sampler_control_margin | metric | none | true | false | false | 完整论文机制中 `wrong_sampler_control_margin` 的受治理记录字段。 |
| wrong_sampler_control_shift | metric | none | true | false | false | 完整论文机制中 `wrong_sampler_control_shift` 的受治理记录字段。 |
| wrong_sampler_replay_cycle_error | metric | none | true | false | false | 完整论文机制中 `wrong_sampler_replay_cycle_error` 的受治理记录字段。 |

| S_final_unconstrained | metric | none | true | false | false | admissibility 置零前的多证据保守分数, 用于正式 admissibility 消融。 |
| endpoint_bit_accuracy | metric | none | true | false | false | endpoint latent 中按 tubelet 恢复的 payload bit 准确率。 |
| endpoint_coverage_ratio | metric | none | true | false | false | 具有有效范数并参与 endpoint 检测的 tubelet 覆盖率。 |
| endpoint_evidence_source | protocol | none | true | false | false | endpoint 证据的数据来源。 |
| endpoint_evidence_status | protocol | none | true | false | false | endpoint VAE 重编码与 key 检测状态。 |
| endpoint_latent_norm | metric | none | true | false | false | Wan VAE 归一化 endpoint latent 的范数。 |
| endpoint_latent_shape | protocol | none | true | false | false | Wan VAE 归一化 endpoint latent 的五维 shape。 |
| endpoint_projection | metric | none | true | false | false | endpoint latent 对同源 key direction 的余弦投影。 |
| endpoint_score | metric | none | true | false | false | 映射到 [0, 1] 的 endpoint key 检测分数。 |
| endpoint_tubelet_count | metric | none | true | false | false | endpoint 检测覆盖的时空 tubelet 数。 |
| endpoint_vae_encode_status | protocol | none | true | false | false | 攻击后视频的 Wan VAE 编码状态。 |
| endpoint_vae_model_class | protocol | none | true | false | false | 执行 endpoint 重建的 Wan VAE 类名。 |
| endpoint_video_frame_count | metric | none | true | false | false | 进入 Wan VAE endpoint 重建的视频帧数。 |
| flow_scheduler_runtime_verified | protocol | none | true | false | false | 当前 step 是否由 Flow scheduler 运行时包装器真实记录。 |
| attacked_video_replay_uncertainty_records_ready | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `attacked_video_replay_uncertainty_records_ready` 的受治理字段。 |
| authenticated_generation_step_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `authenticated_generation_step_count` 的受治理字段。 |
| authenticated_generation_time_grid_id | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `authenticated_generation_time_grid_id` 的受治理字段。 |
| authenticated_trajectory_sketch_records_ready | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `authenticated_trajectory_sketch_records_ready` 的受治理字段。 |
| claim3_full_support_blocking_reason | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `claim3_full_support_blocking_reason` 的受治理字段。 |
| complete_result_flags | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `complete_result_flags` 的受治理字段。 |
| matched_sketch_digest | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `matched_sketch_digest` 的受治理字段。 |
| minimum_replay_control_pass_rate | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `minimum_replay_control_pass_rate` 的受治理字段。 |
| replay_and_sketch_missing_requirement_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_and_sketch_missing_requirement_count` 的受治理字段。 |
| replay_control_pass_rates | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_control_pass_rates` 的受治理字段。 |
| replay_prompt_digest | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_prompt_digest` 的受治理字段。 |
| replay_record_type | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_record_type` 的受治理字段。 |
| replay_sampler_signature | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_sampler_signature` 的受治理字段。 |
| replay_uncertainty_ready_count | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_uncertainty_ready_count` 的受治理字段。 |
| replay_uncertainty_record_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_uncertainty_record_count` 的受治理字段。 |
| replay_uncertainty_records | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_uncertainty_records` 的受治理字段。 |
| replay_uncertainty_weight_mean | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `replay_uncertainty_weight_mean` 的受治理字段。 |
| trace_steps | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `trace_steps` 的受治理字段。 |
| trajectory_sketch_verification_record_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `trajectory_sketch_verification_record_count` 的受治理字段。 |
| trajectory_sketch_verification_records | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `trajectory_sketch_verification_records` 的受治理字段。 |
| trajectory_sketch_verified_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `trajectory_sketch_verified_count` 的受治理字段。 |
| wrong_key | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_key` 的受治理字段。 |
| wrong_key_replay_control_reliable | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_key_replay_control_reliable` 的受治理字段。 |
| wrong_key_replay_cycle_error | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_key_replay_cycle_error` 的受治理字段。 |
| wrong_key_replay_record_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_key_replay_record_count` 的受治理字段。 |
| wrong_key_replay_records | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_key_replay_records` 的受治理字段。 |
| wrong_key_replay_rejected_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_key_replay_rejected_count` 的受治理字段。 |
| wrong_prompt | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt` 的受治理字段。 |
| wrong_prompt_id | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt_id` 的受治理字段。 |
| wrong_prompt_replay_control_reliable | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt_replay_control_reliable` 的受治理字段。 |
| wrong_prompt_replay_record_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt_replay_record_count` 的受治理字段。 |
| wrong_prompt_replay_records | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt_replay_records` 的受治理字段。 |
| wrong_prompt_replay_rejected_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt_replay_rejected_count` 的受治理字段。 |
| wrong_prompt_sketch_digest | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_prompt_sketch_digest` 的受治理字段。 |
| wrong_sampler | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_sampler` 的受治理字段。 |
| wrong_sampler_replay_control | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_sampler_replay_control` 的受治理字段。 |
| wrong_sampler_replay_control_reliable | governance | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_sampler_replay_control_reliable` 的受治理字段。 |
| wrong_sampler_replay_record_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_sampler_replay_record_count` 的受治理字段。 |
| wrong_sampler_replay_records | protocol | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_sampler_replay_records` 的受治理字段。 |
| wrong_sampler_replay_rejected_count | metric | none | true | false | false | 完整 Claim-3 replay/sketch 门禁中 `wrong_sampler_replay_rejected_count` 的受治理字段。 |
| complete_paper_mechanism_claim_decision | governance | none | true | true | false | 三层论文主张同时闭合后的统一机制判定。 |
| adaptive_attack_execution_backend | protocol | none | true | false | false | adaptive protocol 的真实执行后端或查询策略。 |
| adaptive_attack_query_count | protocol | none | true | false | false | 黑盒 adaptive attack 实际比较的候选查询数量。 |
| adaptive_video_attack_generation_status | protocol | none | true | false | false | 跨样本 copy/collusion 视频是否成功生成。 |
| adaptive_video_attack_input_paths | protocol | none | true | false | false | 跨样本 adaptive video 使用的全部源视频路径。 |
| attack_runtime_failure_reason | protocol | none | true | false | false | 视频攻击执行失败原因。 |
| attack_runtime_status | protocol | none | true | false | false | 视频攻击的实际执行状态。 |
| formal_method_variant_execution | protocol | none | true | false | false | 当前 generation 是否属于正式8变体消融运行。 |
| generation_sample_role | protocol | none | true | false | false | 生成单元在 positive 或 clean negative 协议中的角色。 |
| require_complete_paper_mechanism_contract | protocol | none | true | false | false | profile 是否强制启用三层完整论文机制契约。 |
| watermark_embedding_status | protocol | none | true | false | false | 生成单元实际采用的水印嵌入或 clean 控制状态。 |
| claim_2_paired_detection_gain_ci_95_lower | metric | none | true | true | false | 固定 FPR 下同视频配对检测增益的 95% 区间下界。 |
| claim_2_paired_detection_gain_ci_95_upper | metric | none | true | true | false | 固定 FPR 下同视频配对检测增益的 95% 区间上界。 |
| claim_1_empirical_fpr_ci_95_lower | metric | none | true | true | false | held-out empirical FPR 的 Wilson 95% 区间下界。 |
| claim_1_empirical_fpr_ci_95_upper | metric | none | true | true | false | held-out empirical FPR 的 Wilson 95% 区间上界。 |
| claim_1_tpr_ci_95_lower | metric | none | true | true | false | 目标 FPR 下 attacked-positive TPR 的 Wilson 95% 区间下界。 |
| claim_1_tpr_ci_95_upper | metric | none | true | true | false | 目标 FPR 下 attacked-positive TPR 的 Wilson 95% 区间上界。 |
| flow_replay_posterior_ready_count | metric | none | true | true | false | 具备 posterior confidence、entropy 和保守分数的真实 replay 记录数。 |
| minimum_replay_reliability_mean | metric | none | true | true | false | Claim-3 门禁要求的 replay reliability 均值下界。 |
| flow_replay_posterior_records_ready | metric | none | true | true | false | 真实 replay posterior 字段覆盖是否完整。 |
| replay_reliability_mean_ready | metric | none | true | true | false | replay reliability 均值是否达到 Claim-3 契约。 |

## 审稿证据索引字段

| 字段名 | 类别 | 后缀要求 | 必需 | 可空 | 可支持主张 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| reviewer_evidence_index_decision | governance | none | true | false | true | 当前 paper profile 的三层主张是否均映射到存在且可校验的 governed artifacts。 |
| reviewer_evidence_index_required | protocol | none | true | false | true | 正式 paper profile 是否必须在最终 gate 后生成审稿证据索引。 |
| indexed_claim_count | metric | none | true | false | false | 审稿证据索引覆盖的主张数量。 |
| indexed_artifact_count | metric | none | true | false | false | 审稿证据索引覆盖的 artifact 数量。 |
| failed_claim_ids | governance | none | true | false | false | 未通过完整机制门禁的 claim ID 列表。 |
| missing_evidence_paths | governance | none | true | false | false | 声明为必需但没有实际文件的 artifact 相对路径。 |
| evidence_rows | artifact | none | true | false | true | claim、artifact 路径、存在状态和内容摘要组成的索引行。 |
| claim_decision_field | governance | none | true | false | true | 审稿证据行对应的完整机制 claim decision 字段名。 |
| claim_decision | governance | none | true | false | true | 审稿证据行对应的 Claim-1、Claim-2 或 Claim-3 判定值。 |
| evidence_exists | governance | none | true | false | true | 对应 evidence_path 是否为实际文件。 |
| evidence_sha256 | artifact | none | true | true | true | governed artifact 的 SHA-256 内容摘要。 |
| evidence_source | governance | none | true | false | true | 证据来源类型，正式索引固定为 governed_artifact。 |
| profile_gate_path | artifact | none | true | true | true | 当前 profile 最终 gate decision 的相对路径。 |
| profile_gate_passed | governance | none | true | false | true | 当前 profile 最终 gate 是否已经通过。 |

## 完整论文机制新增字段

| 字段名 | 类别 | 后缀要求 | 必需 | 可空 | 可支持主张 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| paper_profile_common_contract_path | protocol | none | true | false | true | 三个正式 profile 共用机制契约的路径。 |
| paper_profile_common_contract_id | protocol | none | true | false | true | 三个正式 profile 共用机制契约标识。 |
| paper_profile_common_contract_status | governance | none | true | false | true | profile 公共机制字段逐项一致性状态。 |
| require_replay_and_sketch_full_support | governance | none | true | false | true | 正式 profile 是否强制 Claim-3 replay 与认证 sketch 完整支持。 |
| require_cluster_aware_statistics | governance | none | true | false | true | 是否强制以 source video 为统计簇。 |
| require_claim1_velocity_causal_gain | governance | none | true | false | true | 是否强制 Claim-1 无速度约束因果对照。 |
| require_calibrated_probability_posterior | governance | none | true | false | true | 是否强制使用校准概率后验。 |
| require_per_video_adaptive_attack_optimization | governance | none | true | false | true | 是否强制逐视频执行 adaptive attack 搜索。 |
| require_heldout_fpr_confidence_upper_bound | governance | none | true | false | true | 是否要求报告 held-out FPR 单侧置信上界。 |
| statistical_cluster_id | protocol | none | true | false | true | source video prompt-seed 独立统计簇标识。 |
| statistical_independent_unit | protocol | none | true | false | true | 明确统计独立单元的语义。 |
| statistical_within_cluster_trial_index | protocol | none | true | false | true | 同一视频内 key trial 的重复测量序号。 |
| replay_log_likelihood_ratio | metric | none | true | false | true | 候选 key forward 相对 null forward 的循环误差对数似然比。 |
| replay_log_likelihood_ratio_mean | metric | none | true | false | true | 多时间网格 replay 对数似然比均值。 |
| replay_log_likelihood_ratio_standard_deviation | metric | none | true | false | true | 多时间网格 replay 对数似然比标准差。 |
| replay_null_cycle_error_mean | metric | none | true | false | true | null forward hypothesis 循环误差均值。 |
| flow_watermark_posterior_probability | metric | none | true | false | true | class-balanced calibration reference prior 下的水印概率后验。 |
| flow_watermark_posterior_log_odds | metric | none | true | false | true | 水印概率后验的 log odds。 |
| posterior_model_type | method | none | true | false | true | 冻结后验模型类型。 |
| posterior_reference_prior | method | none | true | false | true | 概率后验校准使用的 reference prior。 |
| posterior_probability_semantics | method | none | true | false | true | 后验概率的统计语义。 |
| posterior_feature_names | method | none | true | false | true | 冻结概率后验的有序特征名。 |
| posterior_feature_means | method | none | true | false | true | calibration split 拟合的特征均值。 |
| posterior_feature_scales | method | none | true | false | true | calibration split 拟合的特征标准差。 |
| posterior_coefficients | method | none | true | false | true | 冻结 logistic 后验系数。 |
| posterior_platt_slope | method | none | true | false | true | 冻结 Platt calibration 斜率。 |
| posterior_platt_intercept | method | none | true | false | true | 冻结 Platt calibration 截距。 |
| posterior_calibration_brier_score | metric | none | true | false | true | 分组交叉拟合后验的 class-balanced Brier score。 |
| posterior_calibration_expected_calibration_error | metric | none | true | false | true | 分组交叉拟合后验的 ECE。 |
| posterior_calibration_group_count | metric | none | true | false | true | 后验 calibration 使用的独立视频簇数量。 |
| paired_velocity_causal_score_gain | metric | none | true | false | true | 同 prompt-seed-attack 下完整方法相对无速度约束的分数差。 |
| paired_velocity_causal_detection_gain | metric | none | true | false | true | 同 prompt-seed-attack 下完整方法相对无速度约束的检测判定差。 |
| claim_1_velocity_causal_score_gain_ci_95_lower | metric | none | true | false | true | Claim-1 速度约束因果分数增益簇 bootstrap 下界。 |
| ci_one_sided_exact_upper | metric | none | true | false | true | 按独立 source video 计算的 FPR exact 单侧上界。 |
| ci_statistical_cluster_count | metric | none | true | false | true | 置信区间使用的独立视频簇数量。 |
| adaptive_attack_candidate_records | artifact | none | true | false | true | 单视频 adaptive 搜索的候选视频与查询日志。 |
| adaptive_attack_output_video_sha256 | artifact | none | true | false | true | adaptive attack 输出视频文件 SHA-256。 |
| adaptive_attack_output_quality_psnr | metric | none | true | false | true | adaptive 输出相对输入视频的 PSNR 质量约束。 |
| adaptive_attack_endpoint_tolerance | protocol | none | true | false | true | endpoint-preserving 搜索允许的 endpoint 分数偏差。 |
| per_video_adaptive_attack_optimization | governance | none | true | false | true | adaptive 记录是否由逐视频候选生成与查询形成。 |

## 正式 profile 公共证据闭合字段

| 字段名 | 类别 | 后缀要求 | 必需 | 可空 | 可支持主张 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| paper_profile_evidence_closure_decision | governance | none | true | false | true | 当前正式 profile 是否通过与另外两个层级相同的公共论文证据门禁。 |
| paper_profile_evidence_closure_checks | governance | none | true | false | true | 已启用公共条件及其布尔判定的映射。 |
| paper_profile_evidence_closure_required_check_count | metric | none | true | false | false | 当前 profile 实际启用的公共证据条件数量。 |
| paper_profile_evidence_closure_missing_requirements | governance | none | true | false | true | 未通过的公共证据条件名称列表。 |
| paper_profile_evidence_closure_missing_requirement_count | metric | none | true | false | false | 未通过公共证据条件的数量。 |
| post_gate_requirements | governance | none | true | false | false | 因依赖 profile gate 本身而必须在 gate 后验证的条件列表。 |
| common_profile_contract_matched | governance | none | true | false | true | 正式 profile 配置是否与公共机制契约逐字段一致。 |
| complete_paper_mechanism_claim_passed | governance | none | true | false | true | 三层主张的统一完整机制 decision 是否通过。 |
| claim_1_velocity_constraint_full_support_passed | governance | none | true | false | true | Claim-1 是否同时通过因果配对、Flow 证据与完整机制判定。 |
| claim_2_path_evidence_full_support_passed | governance | none | true | false | true | Claim-2 是否同时通过路径独立增益、Flow 证据与完整机制判定。 |
| claim_3_replay_posterior_full_support_passed | governance | none | true | false | true | Claim-3 是否通过 replay/sketch 门禁且未使用降级语义。 |
| calibrated_probability_posterior_passed | governance | none | true | false | true | replay 后验是否通过正式 Flow evidence 与概率校准审计。 |
| per_video_adaptive_attack_optimization_passed | governance | none | true | false | true | adaptive attack 是否逐视频执行并允许正式鲁棒性结论。 |
| statistical_confidence_interval_passed | governance | none | true | false | true | 当前目标 FPR 的正式置信区间 artifact 是否通过。 |
| heldout_fpr_confidence_upper_within_target | governance | none | true | false | true | held-out FPR 的95%单侧置信上界是否不高于目标 FPR。 |
| low_fpr_formal_statistics_passed | governance | none | true | false | true | 当前 profile 的低 FPR 阻断统计是否允许当前层级结论。 |
| low_fpr_curve_passed | governance | none | true | false | true | 当前目标 FPR 的 governed 曲线点是否就绪。 |
| formal_internal_ablation_summary_passed | governance | none | true | false | true | 完整内部消融汇总是否通过。 |
| internal_ablation_matrix_passed | governance | none | true | false | true | 组件移除与匹配对照矩阵是否通过。 |
| video_quality_metrics_passed | governance | none | true | false | true | 当前 profile 的视频质量指标 artifact 是否通过。 |
| efficiency_metrics_passed | governance | none | true | false | false | 当前 profile 的运行效率指标 artifact 是否通过。 |
| real_adaptive_attack_records_passed | governance | none | true | false | true | 真实 adaptive attack 汇总是否通过。 |
| real_world_attack_records_passed | governance | none | true | false | true | 真实世界攻击汇总是否通过。 |
| paper_result_artifact_skeleton_passed | governance | none | true | false | true | 论文结果 records、tables、figures 与 reports 骨架是否完整。 |
| artifact_rebuild_dry_run_passed | governance | none | true | false | true | governed artifacts 的可重建性检查是否通过。 |
| data_split_and_leakage_guard_passed | governance | none | true | false | true | calibration、held-out test 与身份键隔离是否通过。 |
| fair_detection_calibration_passed | governance | none | true | false | true | 所有方法是否按各自 clean negative 在同一 FPR 下冻结阈值。 |
| formal_method_baseline_comparison_passed | governance | none | true | false | true | SSTW 与外部 baseline 的正式比较是否通过。 |
| formal_baseline_difference_interval_passed | governance | none | true | false | true | 相对 baseline 的差值区间是否通过。 |
| external_baseline_self_containment_passed | governance | none | true | false | true | baseline 分数是否由仓库内官方运行路径生成。 |
| external_baseline_comparison_passed | governance | none | true | false | true | 必需的现代外部 baseline 是否具有正式结果。 |
| sstw_measured_formal_records_passed | governance | none | true | false | true | SSTW 正式检测记录是否由真实攻击视频产生。 |
| motion_threshold_calibration_passed | governance | none | true | false | true | 运动一致性阈值是否仅由 calibration split 冻结。 |
| adaptive_attack_protocol_passed | governance | none | true | false | true | adaptive 与 non-runtime 协议聚合门禁是否通过。 |

## VideoMark 官方 baseline 字段

| 字段名 | 类别 | 后缀要求 | 必需 | 可空 | 可支持主张 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| detected_frame_rate | metric | none | true | false | true | VideoMark 官方 PRC detector 判定为有效水印帧的比例。 |
| official_inversion_frame_count | metric | none | true | false | true | VideoMark DDIM inversion 实际返回的 latent 帧数。 |
| official_temporal_matching_evaluated_frame_count | metric | none | true | false | true | Temporal Matching Module 实际参与视频级聚合的帧数。 |
| official_method_primitives | method | none | true | false | true | official bundle 实际调用的 VideoMark 算法原语列表。 |
| videomark_message_shift | protocol | none | true | false | true | 当前 prompt / seed 视频在官方 PRC 消息序列中的窗口起点。 |
| videomark_message_bit_count | protocol | none | true | false | true | 单帧 VideoMark PRC 消息的 bit 数量。 |
| videomark_internal_detector_fpr | protocol | none | true | false | true | VideoMark 官方逐帧 PRC Detect 使用的内部 FPR 参数。 |
| generated_prompt_seed_pair_count | metric | none | true | false | false | official bundle 为避免按 attack 重复生成而缓存的独立 prompt / seed 数量。 |
| supplemental_external_baseline_count | metric | none | true | false | false | table plan 中只进入附录、不参与主表必跑门禁的外部 baseline 数量。 |

## 跨模型 Flow latent 与泛化字段

| 字段名 | 类别 | 后缀要求 | 必需 | 可空 | 可支持主张 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| flow_latent_layout_id | method | none | true | false | true | 模型原生 latent 到 SSTW 五维 tubelet 坐标的可逆布局标识。 |
| flow_latent_native_rank | method | none | true | false | true | 第三方生成模型 scheduler 实际消费的 latent 张量阶数。 |
| flow_latent_canonical_rank | method | none | true | false | true | SSTW tubelet 原语固定使用的规范 latent 张量阶数。 |
| flow_latent_layout_roundtrip_exact | method | none | true | false | true | layout pack/unpack 是否为逐元素精确可逆变换。 |
| flow_latent_num_frames | method | none | false | true | true | 五维规范 latent 的时间长度。 |
| flow_latent_height | method | none | false | true | true | 五维规范 latent 的空间高度。 |
| flow_latent_width | method | none | false | true | true | 五维规范 latent 的空间宽度。 |
| flow_latent_spatial_patch_size | method | none | false | true | true | token 模型使用的空间 patch 大小。 |
| flow_latent_temporal_patch_size | method | none | false | true | true | token 模型使用的时间 patch 大小。 |
| flow_latent_token_count | method | none | false | true | true | packed token latent 的序列长度。 |
| endpoint_native_latent_shape | method | none | false | true | true | VAE endpoint 转换为 scheduler 原生布局后的形状。 |
| model_specific_calibration | protocol | none | true | false | true | frozen detector 是否按 generation model 隔离校准。 |
| cross_model_validation_record_count | generalization | none | true | false | false | 跨模型生成计划实际产生的记录数。 |
| cross_model_validation_success_count | generalization | none | true | false | false | 跨模型生成成功记录数。 |
| cross_model_generalization_decision | generalization | none | true | false | true | 资源受限跨模型子集的三层机制方向审计结论。 |
| cross_model_generalization_passed | governance | none | true | false | true | 三个正式 profile 是否都通过同一跨模型支持性泛化门禁。 |
| cross_model_generalization_claim_scope | generalization | none | true | false | true | 跨模型证据的论文主张边界, 不得冒充主固定 FPR 闭合结论。 |
| cross_model_generalization_model_ids | generalization | none | true | false | true | 实际参加泛化审计的生成模型 ID 列表。 |
| cross_model_generalization_record_count | generalization | none | true | false | true | 跨模型正式 Flow evidence 记录数。 |
| cross_model_generalization_per_model | generalization | none | true | false | true | 每个跨模型的检测、因果、路径与 replay 审计摘要。 |
| cross_model_generalization_model_decision | generalization | none | true | false | true | 单个跨模型是否复现 SSTW 三层机制的方向性证据。 |
| cross_model_test_positive_cluster_count | generalization | none | true | false | true | 跨模型 test positive 独立视频簇数量。 |
| cross_model_test_negative_cluster_count | generalization | none | true | false | true | 跨模型 test negative 独立视频簇数量。 |
| cross_model_test_tpr | generalization | none | true | false | true | 跨模型模型专属冻结阈值下的 test TPR。 |
| cross_model_test_fpr | generalization | none | true | false | true | 跨模型模型专属冻结阈值下的 test FPR 点估计。 |
| cross_model_test_fpr_ci_95_upper | generalization | none | true | false | true | 跨模型 test FPR 的95%置信上界, 用于披露小样本不确定性。 |
| cross_model_path_pair_count | generalization | none | true | false | true | 跨模型 Claim-2 同视频路径增益配对数量。 |
| cross_model_path_score_gain_mean | generalization | none | true | false | true | 跨模型完整检测器相对仅移除路径证据嵌套消融的平均分数增益。 |
| cross_model_velocity_pair_count | generalization | none | true | false | true | 跨模型 Claim-1 速度约束因果配对数量。 |
| cross_model_velocity_score_gain_mean | generalization | none | true | false | true | 跨模型完整方法相对无速度约束的平均分数增益。 |
| cross_model_replay_control_record_count | generalization | none | true | false | true | 跨模型真实 replay control 记录数。 |
| claim_decision_source_path | governance | none | true | false | true | 审稿证据索引读取当前主张判定的 governed artifact 路径。 |
| replay_likelihood_model_id | method | none | true | true | false | replay 残差概率模型的固定标识, 用于阻断误差比冒充似然比。 |
| replay_minimum_observation_noise_variance | method | none | true | false | false | replay 高斯观测模型预注册的最小方差。 |
| replay_relative_observation_noise_standard_deviation | method | none | true | true | false | 仅由模型特定 calibration clean-video null residual 拟合并冻结的相对 endpoint 噪声标准差。 |
| replay_likelihood_calibration_protocol | method | none | true | true | false | replay 噪声模型的拟合协议, 正式值必须为 clean-video 簇等权最大似然拟合。 |
| replay_likelihood_calibration_cluster_count | metric | none | true | true | false | replay 噪声模型拟合使用的独立 clean-video 簇数量, 不把多个时间网格重复计数。 |
| replay_likelihood_calibration_record_id | identity | none | true | false | false | 模型特定 replay 噪声冻结记录的稳定摘要标识。 |
| replay_likelihood_calibration_source_split | protocol | none | true | true | false | replay 噪声模型的数据来源, 正式路径只能是 calibration split。 |
| replay_likelihood_calibration_clean_video_cluster_count | metric | none | true | true | false | 单个生成模型参与 replay 噪声拟合的独立 clean-video 簇数量。 |
| replay_likelihood_calibration_null_residual_observation_count | metric | none | true | false | false | calibration clean videos 在预注册噪声拟合网格上产生的 null residual 观测总数。 |
| replay_likelihood_calibration_step_count | protocol | none | true | true | false | 噪声拟合 bootstrap replay 使用的预注册单网格步数。 |
| replay_likelihood_calibration_step_counts | protocol | none | true | true | false | 模型特定噪声冻结记录实际使用的 replay 步数列表。 |
| replay_likelihood_calibration_grid_policy | protocol | none | true | true | false | 噪声拟合只使用单个预注册主网格以限制 GPU 计算与内存占用的协议。 |
| replay_likelihood_calibration_status | governance | none | true | true | false | 模型特定 replay 噪声是否已由 calibration clean videos 真实拟合并冻结。 |
| test_time_likelihood_update_blocked | governance | none | true | true | false | held-out test 与 adaptive 查询期间是否禁止更新 replay 噪声模型。 |
| replay_likelihood_calibration_decision | governance | none | true | true | true | 全部正式生成模型的 replay 噪声拟合、覆盖与冻结审计结论。 |
| replay_likelihood_calibration_record_count | metric | none | true | false | true | 正式 replay 噪声冻结记录数量。 |
| replay_likelihood_calibration_failures | governance | none | true | false | true | replay 噪声拟合协议、样本规模或模型覆盖失败明细。 |
| minimum_replay_likelihood_calibration_clean_video_cluster_count | protocol | none | true | true | false | 每个生成模型拟合 replay 噪声所需的最少独立 calibration clean-video 簇数量。 |
| flow_state_observation_step_index | method | none | true | false | false | 固定 replay 路径上状态空间观测的步骤索引。 |
| velocity_score | metric | none | true | true | false | 单个 Flow phase 的 key-conditioned 速度投影观测。 |
| path_score | metric | none | true | true | false | 单个 Flow phase 的 key-conditioned 路径投影观测。 |
| key_agnostic_endpoint_energy | metric | none | true | false | false | generic SSM 对照可使用的 endpoint 能量, 不包含密钥方向。 |
| key_agnostic_velocity_energy | metric | none | true | false | false | generic SSM 对照可使用的速度能量, 不包含密钥方向。 |
| key_agnostic_path_energy | metric | none | true | false | false | generic SSM 对照可使用的路径能量, 不包含密钥方向。 |
| replay_observation_noise_variance | metric | none | true | true | false | 单个 Flow phase 的 replay 高斯观测噪声方差。 |
| flow_state_observation_sequence | method | none | true | true | false | 从固定反演路径逐 phase 测得的状态空间观测序列。 |
| flow_state_observation_sequence_status | governance | none | true | true | false | 状态空间观测序列是否来自真实固定 replay 路径。 |
| flow_state_observation_step_count | metric | none | true | true | false | 状态空间观测序列实际包含的 phase 数量。 |
| flow_state_transition_source | method | none | true | true | false | 状态转移参数的拟合来源, 正式结果只能来自 calibration split。 |
| replay_observation_noise_variance_mean | metric | none | true | true | false | 多网格 replay 高斯观测方差的平均值。 |
| replay_candidate_log_likelihood_per_dimension_mean | metric | none | true | true | false | 候选密钥 replay 的逐 latent 维平均对数似然。 |
| replay_null_log_likelihood_per_dimension_mean | metric | none | true | true | false | null replay 的逐 latent 维平均对数似然。 |
| transition_matrix | method | none | true | false | false | calibration split 拟合的线性高斯状态转移矩阵。 |
| transition_bias | method | none | true | false | false | calibration split 拟合的线性高斯状态转移偏置。 |
| process_covariance | method | none | true | false | false | calibration split 拟合的状态过程噪声协方差。 |
| observation_covariance | method | none | true | false | false | calibration split 拟合的状态观测噪声协方差。 |
| initial_mean | method | none | true | false | false | 双假设状态空间模型的初始状态均值。 |
| initial_covariance | method | none | true | false | false | 双假设状态空间模型的初始状态协方差。 |
| training_sequence_count | metric | none | true | false | false | 单个假设状态模型使用的 calibration 序列数。 |
| training_group_count | metric | none | true | true | false | 单个假设状态模型使用的独立 source-video group 数量。 |
| training_transition_count | metric | none | true | true | false | 单个假设状态模型实际拟合的跨 phase 转移数。 |
| training_transition_group_count | metric | none | true | true | false | 对状态转移拟合有贡献的独立 source-video group 数量。 |
| posterior_negative_state_space_model | method | none | true | true | false | 可从 threshold artifact 重建的 H0 线性高斯状态空间参数。 |
| posterior_positive_state_space_model | method | none | true | true | false | 可从 threshold artifact 重建的 H1 线性高斯状态空间参数。 |
| flow_state_positive_log_likelihood_per_step | metric | none | true | true | false | H1 状态空间模型产生的逐 phase 边际对数似然。 |
| flow_state_negative_log_likelihood_per_step | metric | none | true | true | false | H0 状态空间模型产生的逐 phase 边际对数似然。 |
| flow_state_log_likelihood_ratio | metric | none | true | true | false | H1 与 H0 状态空间边际对数似然之差。 |
| flow_state_filter_step_count | metric | none | true | true | false | Kalman filter 实际消费的状态观测 phase 数。 |
| flow_state_filtering_status | governance | none | true | true | false | 正式后验是否完成 Kalman filtering。 |
| flow_state_smoothing_status | governance | none | true | true | false | 正式后验是否完成 RTS smoothing。 |
| state_space_posterior_mechanism_decision | governance | none | true | true | false | 正式 Flow 门禁对真实动态后验机制的审计结论。 |
| state_space_posterior_mechanism_failures | governance | none | true | false | false | 状态空间后验机制未闭合时的明确缺失条件列表。 |
| flow_replay_state_space_filtering_smoothing_ready | governance | none | true | true | false | Claim-3 replay 门禁是否逐记录具备多步 filtering 和 smoothing 证据。 |
| generation_seed_random | random | _random | true | false | false | 生成视频使用的受控随机种子, 仅用于配对复现和因果审计。 |
| generation_generator_state_digest_random | random | _digest_random | true | true | false | pipeline 调用前 GPU generator 状态的摘要, 用于证明配对生成共享随机起点。 |
| velocity_causal_pair_id | method | none | true | true | false | 同模型、prompt、seed 和 split 的速度约束因果配对标识。 |
| velocity_causal_intervention_status | method | none | true | true | false | 配对生成中速度约束干预是启用、禁用或不适用。 |
| generation_source_video_sha256 | provenance | none | true | true | false | 进入攻击与检测的生成源视频内容摘要。 |
| paired_detector_method_variant | method | none | true | true | false | Claim-1 因果配对两侧共同使用的冻结检测器变体。 |
| paired_detector_threshold_source_split | governance | none | true | true | false | Claim-1 配对检测阈值必须来自 calibration split。 |
| paired_test_time_threshold_update_blocked | governance | none | true | true | false | Claim-1 配对检测是否禁止 test-time 阈值更新。 |
| full_generation_generator_state_digest_random | random | _digest_random | true | true | false | 完整方法生成侧的 GPU generator 状态摘要。 |
| control_generation_generator_state_digest_random | random | _digest_random | true | true | false | 无速度约束对照生成侧的 GPU generator 状态摘要。 |
| full_generation_source_video_sha256 | provenance | none | true | true | false | 完整方法配对视频的内容摘要。 |
| control_generation_source_video_sha256 | provenance | none | true | true | false | 无速度约束配对视频的内容摘要。 |
| velocity_causal_pairing_status | governance | none | true | true | false | Claim-1 配对是否满足单一干预、共享随机状态和共享采样配置。 |
| claim_1_velocity_causal_expected_pair_count | metric | none | true | true | false | 每个 held-out full-method 视频应具有的速度因果配对总数。 |
| claim_1_velocity_causal_pairing_failure_count | metric | none | true | true | false | 未满足严格受控生成设计的 Claim-1 配对数量。 |
| claim_1_velocity_causal_pair_coverage | metric | none | true | true | false | 有效速度因果配对数占预期配对数的比例。 |
| claim_1_velocity_causal_detector_protocol | governance | none | true | true | false | Claim-1 两侧统一应用完整方法冻结检测器的协议标识。 |
| detector_score_source | provenance | none | true | true | false | adaptive 候选查询所返回检测分数的明确机制来源。 |
| adaptive_attack_query_budget | governance | none | true | true | false | 单视频单协议实际允许的冻结检测器查询上限。 |
| adaptive_attack_replay_likelihood_model_id | method | none | true | true | false | adaptive 最优候选使用的冻结 replay 概率模型标识。 |
| adaptive_attack_replay_likelihood_calibration_protocol | method | none | true | true | false | adaptive 最优候选使用的 replay 噪声 calibration 协议。 |
| adaptive_attack_replay_likelihood_calibration_cluster_count | metric | none | true | true | false | adaptive 最优候选使用的冻结 replay 噪声模型独立拟合簇数量。 |
| adaptive_attack_replay_relative_observation_noise_standard_deviation | method | none | true | true | false | adaptive 最优候选查询复用的模型特定冻结 replay 相对噪声标准差。 |
| adaptive_query_role | governance | none | true | false | false | adaptive 查询属于 held-out 视频还是 calibration public negative。 |
| adaptive_attack_candidate_query_count | metric | none | true | true | false | held-out 视频 adaptive 候选的真实冻结检测器查询数。 |
| adaptive_attack_public_negative_query_count | metric | none | true | true | false | 仅用于预注册探测顺序的 calibration public negative 查询数。 |
| adaptive_attack_query_provenance_decision | governance | none | true | true | false | 所有 adaptive 候选是否具备解码视频、内容摘要、状态空间分数和冻结阈值来源。 |
| calibration_negative_cluster_count | metric | none | true | true | false | fixed-FPR calibration 使用的独立 negative source-video group 数。 |
| calibration_positive_cluster_count | metric | none | true | true | false | 概率后验 calibration 使用的独立 positive source-video group 数。 |
| paired_detector_score_source | provenance | none | true | true | false | Claim-1 两侧共享检测器的状态空间分数来源。 |
| paired_frozen_final_score_threshold | metric | none | true | true | false | Claim-1 两侧共享的完整方法冻结阈值。 |
| paired_detector_target_fpr | metric | none | true | true | false | Claim-1 共享检测器阈值对应的目标 FPR。 |
| adaptive_attack_control_margin | metric | none | true | true | false | wrong key、prompt、sampler 或 time-grid replay 对照相对正确假设的实测间隔。 |
| adaptive_attack_control_rejected | metric | none | true | true | false | replay 错误条件是否由先前真实控制运行拒绝。 |
| adaptive_watermark_retention_minimum_rate | governance | none | true | true | false | 预注册的 adaptive 攻击后最低水印检出率。 |
| adaptive_watermark_retention_statistics | metric | none | true | true | false | 各 adaptive 协议按 source-video cluster 计算的检出率与区间。 |
| adaptive_watermark_retention_decision | governance | none | true | true | false | 所有水印保留型 adaptive 协议是否达到预注册检出率。 |
| adaptive_watermark_retention_rate_estimate | metric | none | true | true | false | 单个 adaptive 协议的 cluster-equal 检出率点估计。 |
| adaptive_watermark_retention_rate_ci_95_lower | metric | none | true | true | false | 单个 adaptive 协议检出率的95%区间下界。 |
| adaptive_watermark_retention_rate_ci_95_upper | metric | none | true | true | false | 单个 adaptive 协议检出率的95%区间上界。 |
| adaptive_watermark_retention_rate_cluster_count | metric | none | true | true | false | 单个 adaptive 协议的独立 source-video cluster 数。 |
| adaptive_watermark_retention_rate_observation_count | metric | none | true | false | false | 单个 adaptive 协议的原始记录数, 不作为独立样本量。 |
| adaptive_watermark_retention_rate_bootstrap_resample_count | metric | none | true | false | false | adaptive 检出率 cluster bootstrap 重采样次数。 |
| adaptive_spoof_false_accept_count | metric | none | true | true | false | copy/spoof 协议在独立视频簇上的误接受数量。 |
| adaptive_spoof_cluster_count | metric | none | true | true | false | copy/spoof 协议的独立 source-video cluster 数。 |
| adaptive_spoof_fpr_ci_95_upper | metric | none | true | true | false | copy/spoof 误接受率的单侧95%二项上界。 |
| adaptive_spoof_rejection_decision | governance | none | true | true | false | copy/spoof 误接受上界是否不超过当前 profile 目标 FPR。 |
| adaptive_replay_control_rejection_decision | governance | none | true | true | false | 所有真实 wrong-condition replay 对照是否被拒绝。 |
| minimum_adaptive_watermark_retention_rate | governance | none | true | false | false | 三档正式 profile 共享的 adaptive 水印最低保留率配置。 |
| wrong_key_flow_state_observation_sequence | method | none | true | true | false | 在正确固定 reverse path 上执行 wrong-key forward hypothesis 得到的逐 phase 观测。 |
| wrong_prompt_flow_state_observation_sequence | method | none | true | true | false | 在正确固定 reverse path 上执行 wrong-prompt forward hypothesis 得到的逐 phase 观测。 |
| wrong_sampler_flow_state_observation_sequence | method | none | true | true | false | 在正确固定 reverse path 上执行 wrong-sampler/time-grid forward hypothesis 得到的逐 phase 观测。 |
| replay_control_fixed_reverse_path_reused | governance | none | true | true | false | 三类错误 replay control 是否逐元素复用正确条件的固定 reverse states。 |
| controlled_negative_hypothesis_type | method | none | true | true | false | controlled negative 对应 wrong key、prompt 或 sampler/time-grid 的物理假设类型。 |
| formal_negative_hypothesis_family_decision | governance | none | true | true | false | calibration 与 held-out 是否都覆盖四类真实负假设且达到独立视频数量要求。 |
| formal_required_negative_hypothesis_families | governance | none | true | true | false | 正式 SSTW 检测预注册的四类负假设名称。 |
| formal_calibration_negative_family_cluster_counts | metric | none | true | true | false | calibration split 各负假设 family 的独立 source-video 数量。 |
| formal_heldout_negative_family_cluster_counts | metric | none | true | true | false | held-out split 各负假设 family 的独立 source-video 数量。 |
| calibration_negative_raw_hypothesis_count | metric | none | true | false | false | calibration split 的原始负假设记录数, 不作为独立样本量。 |
| heldout_negative_raw_hypothesis_count | metric | none | true | false | false | held-out split 的原始负假设记录数, 不作为独立样本量。 |
| coverage_ratio | metric | none | true | true | false | 单个 Flow phase 或 endpoint 的有效 tubelet 覆盖比例。 |
| failure_scope | governance | none | true | false | false | 正式门禁失败发生在 frozen calibration 还是 scored sequence。 |
| failed_requirements | governance | none | true | false | false | 单条机制失败记录未满足的明确要求列表。 |
| paired_full_method_decision | metric | none | true | true | false | Claim-1 共享冻结检测器对完整方法配对视频的判定。 |
| paired_without_velocity_constraint_decision | metric | none | true | true | false | Claim-1 共享冻结检测器对无速度约束配对视频的判定。 |
| posterior_entropy_maximum | method | none | true | false | false | calibration positive group 分布确定的后验熵 admissibility 上限。 |
| wrong_key_replay_log_likelihood_ratio | metric | none | true | true | false | wrong-key forward hypothesis 相对固定 null replay 的高斯 LLR。 |
| wrong_prompt_replay_log_likelihood_ratio | metric | none | true | true | false | wrong-prompt forward hypothesis 相对固定 null replay 的高斯 LLR。 |
| wrong_sampler_replay_log_likelihood_ratio | metric | none | true | true | false | wrong-sampler/time-grid forward hypothesis 相对固定 null replay 的高斯 LLR。 |
| adaptive_attack_output_decoded_frame_count | metric | none | true | true | false | 跨视频攻击视频落盘后重新解码得到的实际帧数。 |
| adaptive_attack_secondary_input_video_path | provenance | none | true | false | false | copy/spoof 或 collusion 查询使用的第二个真实输入视频路径。 |
| adaptive_attack_secondary_input_video_sha256 | provenance | none | true | true | false | 跨视频查询第二个输入视频的内容摘要。 |
| formal_adaptive_attack_expected_query_record_count | metric | none | true | true | false | 按各协议实际查询数与 public negative 探测数计算的预期查询日志总数。 |
| adaptive_attack_donor_statistical_cluster_id | provenance | none | true | true | false | copy/spoof 攻击中提供水印内容的 donor 视频独立簇标识。 |
| adaptive_attack_member_statistical_cluster_ids | provenance | none | true | true | false | collusion 不重叠视频对包含的两个 source-video cluster 标识。 |
| attack_parameters | method | none | true | true | false | 单次 adaptive detector 查询实际使用的连续攻击参数。 |
| adaptive_attack_selected_parameters | method | none | true | true | false | 冻结检测器目标下最终选中候选的攻击参数。 |
| adaptive_attack_optimizer_type | method | none | true | true | false | 逐视频 adaptive attack 使用的序贯黑盒优化算法。 |
| adaptive_search_protocol | method | none | true | true | false | 参数化 adaptive attack 的二维 detector-feedback pattern search 协议标识。 |
| adaptive_search_coordinate_names | method | none | true | true | false | 当前攻击族两个独立归一化搜索坐标的稳定名称。 |
| adaptive_search_query_phase | method | none | true | true | false | 单次查询属于基点、坐标探针或 detector-feedback 细化阶段。 |
| adaptive_search_coordinate_1_name | method | none | true | true | false | 第一个独立原生攻击参数对应的归一化坐标名称。 |
| adaptive_search_coordinate_1_value | method | none | true | true | false | 第一个原生攻击参数的实际归一化查询坐标。 |
| adaptive_search_coordinate_2_name | method | none | true | true | false | 第二个独立原生攻击参数对应的归一化坐标名称。 |
| adaptive_search_coordinate_2_value | method | none | true | true | false | 第二个原生攻击参数的实际归一化查询坐标。 |
| adaptive_search_feedback_parent_candidate_index | provenance | none | true | true | false | detector-feedback 细化查询所依据的历史最优可接受候选索引; 初始探针为空。 |
| adaptive_attack_public_negative_informed_strength | metric | none | true | true | false | calibration public negative 查询确定的 held-out 初始攻击强度。 |
| model_vae_regeneration_status | governance | none | true | true | false | 候选是否完成模型 VAE encode-perturb-decode 重生成。 |
| model_vae_class | provenance | none | true | true | false | 执行生成式重压缩的官方 VAE 类名。 |
| model_vae_latent_noise_ratio | method | none | true | true | false | 相对 latent 标准差的重生成噪声强度。 |
| model_vae_random_seed_random | random | _random | true | false | false | 模型 VAE 重生成噪声使用的可复现随机种子。 |
| model_vae_source_frame_count | metric | none | true | true | false | VAE encode 输入的真实视频帧数。 |
| model_vae_output_frame_count | metric | none | true | true | false | VAE decode 后写入候选视频的帧数。 |
| formal_metric_complete_value_count | metric | none | true | true | false | 当前质量 scope 中同时具备全部必需实测指标的记录数。 |
| paired_video_quality_status | governance | none | true | true | false | 同模型、prompt、seed 的 clean-reference 配对质量计算状态。 |
| paired_video_quality_failure_reason | governance | none | true | false | false | 配对质量指标不可用时的明确原因。 |
| paired_quality_frame_count | metric | none | true | true | false | 配对 PSNR、SSIM 与时间差分实际使用的公共帧数。 |
| paired_watermark_psnr | metric | none | true | true | false | watermarked 视频相对同源 clean reference 的配对 PSNR。 |
| paired_watermark_ssim | metric | none | true | true | false | watermarked 视频相对同源 clean reference 的配对 SSIM。 |
| paired_temporal_delta_error | metric | none | true | true | false | watermarked 与 clean 视频相邻帧差分之间的归一化平均绝对误差。 |
| paired_video_quality_required | governance | none | true | true | false | 当前记录是否属于必须计算配对失真的 SSTW 完整方法视频。 |
| paired_reference_video_path | provenance | none | true | false | false | 与当前 watermarked 视频共享模型、prompt 和 seed 的 clean reference 路径。 |
| formal_paired_video_quality_ready | governance | none | true | true | false | 当前记录或整体质量门禁是否具备所需配对质量指标。 |
| formal_paired_video_quality_required_count | metric | none | true | true | false | 正式质量门禁要求配对指标的 SSTW 完整方法视频数量。 |
| formal_paired_video_quality_ready_count | metric | none | true | true | false | 已成功获得配对质量指标的 SSTW 完整方法视频数量。 |
| baseline_clean_reference_video_path | provenance | none | true | true | false | 正式 baseline 生成自身水印时使用的同模型、prompt、seed clean reference 路径。 |
| baseline_clean_reference_trajectory_trace_id | provenance | none | true | false | false | 匹配 clean reference 对应的生成轨迹标识, 用于审计输入身份。 |
| baseline_clean_reference_status | governance | none | true | true | false | baseline 输入是否成功匹配同模型、prompt、seed clean reference。 |
| baseline_input_source_policy | protocol | none | true | true | false | 正式 baseline 必须在 clean reference 上嵌入自己的水印, 不得复用 SSTW watermarked source。 |
| require_baseline_matched_video_quality_metrics | protocol | none | true | true | false | 当前 profile 是否要求 SSTW 与全部正式 baseline 完成同口径配对质量计算。 |
| video_quality_comparison_protocol | protocol | none | true | true | false | 跨方法质量比较使用同源 clean reference、方法自身 watermarked source 以及配对 PSNR、SSIM、时间差分的固定协议。 |
| quality_metric_source_kind | provenance | none | true | false | false | 方法级质量汇总所消费的 governed source record 类型。 |
| paired_quality_unit_count | metric | none | true | true | false | 去除重复 attack 行后实际参与方法级配对质量计算的独立视频单元数。 |
| paired_quality_ready_count | metric | none | true | true | false | 成功得到全部配对质量指标的独立视频单元数。 |
| paired_quality_blocked_count | metric | none | true | false | false | 配对质量解码、路径或协议不满足要求的视频单元数。 |
| mean_paired_watermark_psnr | metric | none | true | true | false | 方法自身 watermarked source 相对匹配 clean reference 的平均配对 PSNR。 |
| mean_paired_watermark_ssim | metric | none | true | true | false | 方法自身 watermarked source 相对匹配 clean reference 的平均配对 SSIM。 |
| mean_paired_temporal_delta_error | metric | none | true | true | false | 方法自身 watermarked source 相对匹配 clean reference 的平均时间差分误差。 |
| robustness_tpr_at_target_fpr | metric | none | true | true | false | 与质量记录同一方法在当前固定 FPR 下的 held-out TPR。 |
| robustness_tpr_ci_lower | metric | none | true | false | false | 当前固定 FPR 下方法鲁棒性 TPR 置信区间下界。 |
| robustness_tpr_ci_upper | metric | none | true | false | false | 当前固定 FPR 下方法鲁棒性 TPR 置信区间上界。 |
| quality_metric_failure_reasons | governance | none | true | false | false | 方法级配对质量被阻断的全部 fail-closed 原因。 |
| video_quality_required_method_ids | protocol | none | true | true | false | 正式质量比较必须覆盖的 SSTW 与 baseline 方法集合。 |
| video_quality_ready_method_ids | governance | none | true | true | false | 已完成同口径配对质量和固定 FPR 鲁棒性绑定的方法集合。 |
| video_quality_missing_method_ids | governance | none | true | true | false | 尚未完成正式质量比较的方法集合。 |
| sstw_paired_video_quality_ready | governance | none | true | true | false | SSTW 完整方法的配对 PSNR、SSIM 与时间差分质量是否全部就绪。 |
| baseline_matched_video_quality_ready | governance | none | true | true | false | 5个正式 baseline 是否均以匹配 clean reference 和自身 watermarked source 完成配对质量计算。 |
| baseline_matched_video_quality_ready_method_ids | governance | none | true | false | false | 已完成同口径配对质量计算的正式 baseline 集合。 |
| baseline_matched_video_quality_missing_method_ids | governance | none | true | true | false | 缺少路径、视频或配对指标的正式 baseline 集合。 |
| baseline_matched_video_quality_passed | governance | none | true | true | true | 当前 profile 的跨方法匹配配对质量公共证据门禁是否通过。 |
| alternate_encodings | protocol | none | true | false | false | 同一 governed figure rows 的补充坐标编码, 用于并列呈现 PSNR 与 SSIM。 |
| detector_configuration_id | provenance | none | true | false | false | 外层实验为参数化核心检测器分配的配置标识, 核心方法不根据该标识切换语义。 |
| flow_state_admissibility_enforced | method | none | true | true | false | 冻结检测器是否执行状态证据可接受域约束。 |
| velocity_constraint_enabled | method | none | true | true | false | 当前 scheduler 运行是否启用 SSTW 速度场弱约束。 |
| terminal_endpoint_perturbation_enabled | method | none | true | false | false | 当前参数化运行是否启用终点扰动原语, 仅供外层受控实验组合。 |
| terminal_endpoint_perturbation_delta_norm | metric | none | true | false | false | 终点扰动原语实际施加到模型原生 latent 的增量范数。 |
| package_execution_mode | governance | none | true | false | false | 当前服务器入口运行于开发仓库还是论文产物重建抽离包。 |
| development_checks_packaged | governance | none | true | false | false | 抽离包是否携带开发期 pytest 与 harness 检查实现。 |
| development_checks_execution_policy | governance | none | true | false | false | 开发检查应在抽离前执行还是由当前运行包执行。 |
| skip_reason | governance | none | true | false | false | 某个编排阶段未执行时的明确、可审计原因。 |
| claim_audit_report_passed | governance | none | true | true | false | 完整三层主张 decision 与自动生成的主张审计报告是否同时存在并通过。 |
| cluster_aware_statistics_passed | governance | none | true | true | false | 当前 profile 是否使用 source-video cluster 统计并生成冻结 FPR 置信证据。 |
| formal_motion_claim_passed | governance | none | true | true | false | formal motion 筛选是否已执行且至少保留一个可支撑主张的视频。 |
| sstw_advantage_claim_passed | governance | none | true | true | false | SSTW 相对全部预注册 baseline 的 TPR 差值与置信区间下界是否为正。 |
| sstw_advantage_claim_failures | governance | none | true | false | false | SSTW 优势门禁未通过时按 baseline 记录的明确失败原因。 |
| formal_frozen_threshold_record_id | provenance | none | true | true | false | pilot gate 消费的公平校准 SSTW 阈值 record 标识。 |
| formal_frozen_threshold_source_path | provenance | none | true | true | false | pilot gate 只读消费的正式冻结阈值 record 路径。 |
| formal_frozen_threshold_value | metric | none | true | true | false | 公平校准阶段已经冻结且由 pilot gate 复用的 SSTW 阈值。 |
| formal_frozen_threshold_source_split | provenance | none | true | true | false | 正式冻结阈值的来源 split, 必须为 calibration。 |
| formal_frozen_threshold_protocol | governance | none | true | true | false | 正式冻结阈值使用的 calibration-to-heldout 协议。 |
| formal_frozen_threshold_test_time_update_blocked | governance | none | true | true | false | 正式冻结阈值是否禁止 test-time 更新。 |
| formal_frozen_threshold_calibration_negative_count | metric | none | true | true | false | 正式 SSTW 阈值使用的 calibration negative 独立视频数。 |
| formal_frozen_threshold_calibration_fpr | metric | none | true | true | false | 正式冻结阈值在 calibration split 上的 FPR。 |
| formal_frozen_threshold_heldout_fpr | metric | none | true | true | false | 正式冻结阈值在 held-out split 上的 FPR。 |
| formal_frozen_threshold_tpr | metric | none | true | true | false | 正式冻结阈值在 held-out attacked positives 上的 TPR。 |
| formal_frozen_threshold_heldout_false_positive_count | metric | none | true | true | false | held-out 独立视频中由正式阈值产生的 false positive 数。 |
| formal_frozen_threshold_true_positive_count | metric | none | true | true | false | held-out attacked positives 中由正式阈值检出的数量。 |
| formal_frozen_threshold_ready | governance | none | true | true | false | SSTW 公平校准 record 是否可作为只读正式冻结阈值 artifact。 |
| formal_frozen_threshold_artifact_ready | governance | none | true | true | false | pilot gate 是否已成功消费正式冻结阈值而未重新校准。 |
| stage_transition_dependency_policy | governance | none | true | true | false | 阶段跳转是否明确依赖完整 package manifest 先通过。 |
| source_package_manifest_path | provenance | none | true | false | false | 阶段跳转消费的 source profile package manifest 路径。 |
| source_package_manifest_field | governance | none | true | false | false | source profile package manifest 的规范 decision 字段。 |
| source_package_manifest_decision | governance | none | true | true | false | 阶段跳转读取到的 source profile package 决策。 |
| source_reviewer_evidence_index_path | provenance | none | true | false | false | 阶段跳转消费的审稿证据索引路径。 |
| source_reviewer_evidence_index_decision | governance | none | true | true | false | 阶段跳转读取到的审稿证据索引决策。 |
| source_gate_figure_path | provenance | none | true | false | false | 阶段跳转消费的 source profile 诊断图 manifest 路径。 |
| source_gate_figure_ready | governance | none | true | true | false | source profile 诊断图 manifest 是否已生成。 |
| paper_profile_source_gate_decision_field | governance | none | true | false | false | package builder 为当前 profile 解析出的规范 source gate 字段。 |
| watermark_key_derivation_id | method | none | true | true | false | 从所有者秘密与公开生成上下文派生水印方向的冻结 HMAC 算法标识。 |
| watermark_key_id | provenance | none | true | true | false | 所有者密钥的公开标识, 不包含密钥材料或派生后的水印方向。 |
| posterior_probability_calibration_protocol | method | none | true | true | false | 概率后验使用的嵌套 source-video group 交叉拟合协议。 |
| posterior_probability_calibration_outer_fold_count | metric | none | true | true | false | 生成未见视频簇概率和无泄漏校准指标的外层 fold 数量。 |
| posterior_probability_calibration_inner_fold_minimum | metric | none | true | true | false | 各外层训练分区内部拟合 Platt 映射所用的最小 fold 数量。 |
| fixed_fpr_threshold_score_source | governance | none | true | true | false | 冻结固定 FPR 阈值使用的 calibration 分数来源, 必须为视频簇外推分数。 |
| calibration_source_method_variant | provenance | none | true | true | false | 当前冻结检测器实际复用的 calibration record 方法变体。 |
| detector_only_nested_ablation | governance | none | true | true | false | 当前阈值是否属于不重新生成视频的检测器级嵌套消融。 |
| paired_path_ablation_method_variant | provenance | none | true | true | false | Claim-2 同视频配对使用的仅移除路径证据检测器标识。 |
| paired_path_nested_ablation_status | governance | none | true | true | false | Claim-2 配对是否保持同视频、同 replay 且只移除路径特征。 |
| paired_without_path_evidence_detector_score | metric | none | true | true | false | 同视频 `without_path_evidence` 冻结检测器的保守分数。 |
| paired_without_path_evidence_detector_decision | governance | none | true | true | false | 同视频 `without_path_evidence` 冻结检测器在目标 FPR 下的判定。 |
| paired_full_detector_target_fpr | metric | none | true | true | false | Claim-2 完整检测器使用的预注册目标 FPR。 |
| paired_without_path_evidence_detector_target_fpr | metric | none | true | true | false | Claim-2 仅移除路径检测器使用的预注册目标 FPR。 |
| paired_fpr_alignment_status | governance | none | true | true | false | Claim-2 两个冻结检测器是否使用相同预注册目标 FPR。 |
| claim_2_expected_paired_comparison_count | metric | none | true | true | false | Claim-2 按全部 held-out full-method positive 计算的预期配对数。 |
| claim_2_pairing_failure_count | metric | none | true | true | false | Claim-2 中不满足仅路径单一干预或冻结阈值协议的配对数。 |
| claim_2_paired_comparison_coverage | metric | none | true | true | false | Claim-2 实际配对数占预期 held-out full-method positive 数的比例。 |
| claim_2_nested_ablation_method_variant | provenance | none | true | true | false | Claim-2 因果归因绑定的检测器级嵌套消融标识。 |
| claim_2_causal_comparison_protocol | governance | none | true | true | false | Claim-2 保持视频、replay 和非路径机制不变的冻结检测器比较协议。 |
| S_path_inv_unweighted | metric | none | true | false | false | 应用 replay step 可靠性前的时间重参数化路径积分分数。 |
| path_replay_reliability_weight_mean | metric | none | true | false | false | 路径积分实际消费的逐 step replay 可靠性均值。 |
| path_replay_weighted_aggregation_applied | governance | none | true | false | false | 路径积分是否直接消费逐 step replay 可靠性权重。 |
| path_score_unweighted | metric | none | true | false | false | 单个 Flow phase 在 replay 不确定性衰减前的路径投影。 |
| path_endpoint_consistency_unweighted | metric | none | true | false | false | 单个 Flow phase 在 replay 不确定性衰减前的路径与 endpoint 一致性。 |
| replay_step_reliability_weight | metric | none | true | true | false | 全局多网格可靠性与单步高斯拟合可靠性的乘积, 直接用于路径观测。 |
| replay_step_likelihood_reliability | metric | none | true | false | false | 单个 replay step 根据候选残差和预注册观测方差计算的高斯拟合可靠性。 |
| replay_global_reliability | metric | none | true | false | false | 多时间网格 replay uncertainty 形成的记录级全局可靠性。 |
| path_replay_uncertainty_weighting_status | governance | none | true | true | false | 当前 phase 路径观测采用的 replay 不确定性直接加权协议。 |
| adaptive_parameter_search_policy | method | none | true | true | false | 连续攻击参数如何由先前冻结检测器查询反馈选择下一候选。 |
| attack_strength_semantics | method | none | true | false | false | 兼容字段 attack_strength 的汇总含义, 二维搜索中该值仅作唯一性摘要而不控制原生攻击参数。 |
| adaptive_detector_feedback_search_decision | governance | none | true | true | false | 所有逐视频连续攻击是否按查询预算完成不重复的 detector-feedback 搜索。 |
| adaptive_model_vae_regeneration_decision | governance | none | true | true | false | 生成式攻击的每个候选是否真实执行模型 VAE encode-perturb-decode。 |
| adaptive_public_negative_probe_decision | governance | none | true | true | false | public-negative 探测是否独立完成并只向 held-out 搜索传递攻击参数。 |
| watermark_key_derivation_decision | governance | none | true | true | false | 全部正式 positive 是否声明冻结的所有者秘密 HMAC 派生算法和非空 key ID。 |
| watermark_key_derivation_failures | governance | none | true | false | false | 水印 key 派生算法或公开 key ID 不符合正式协议的记录列表。 |
| paired_watermark_ssim_protocol | method | none | true | true | false | 配对 SSIM 使用的局部高斯窗口、多通道和 data range 配置。 |
| model_vae_noise_direction_policy | method | none | true | true | false | 同一源视频的 latent 攻击是否固定噪声方向并仅优化连续幅度。 |
| detector_only_ablation | governance | none | true | true | false | 当前结果是否复用同一 full-method 视频和 replay, 仅改变检测器观测。 |
| detector_only_source_method_variant | provenance | none | true | true | false | 检测器级消融复用的源视频方法变体, 正式值必须为 `sstw_full_method`。 |
| generation_internal_ablation_variants | governance | none | true | false | false | 三档正式协议中必须独立生成视频的 scheduler 机制变体。 |
| detector_only_internal_ablation_variants | governance | none | true | false | false | 三档正式协议中必须复用 full-method 视频的检测器变体。 |
| internal_ablation_video_reuse_policy | governance | none | true | false | false | 生成机制消融与 detector-only 消融的视频来源隔离策略。 |
| require_internal_ablation_video_reuse_policy | governance | none | true | false | false | 正式 profile 是否强制审计内部消融视频复用策略。 |
| ablation_video_execution_mode | governance | none | true | false | false | 内部消融使用独立生成视频还是复用 full-method 视频。 |
| ablation_source_method_variant | governance | none | true | false | false | 当前消融 record 的真实视频来源方法变体。 |
| ablation_source_trajectory_trace_id | governance | none | true | false | false | 当前消融实际复用或独立生成的视频轨迹标识。 |
| ablation_source_video_sha256 | governance | none | true | false | false | 当前消融实际消费的视频内容摘要。 |
| ablation_independent_video_generation_required | governance | none | true | false | false | 当前变体是否因改变生成机制而必须独立生成视频。 |
| detector_only_video_reuse_decision | governance | none | true | true | false | 全部 detector-only 变体是否逐 trace 复用 full-method 视频。 |
| detector_only_video_reuse_failure_variants | governance | none | true | false | false | 未正确复用 full-method 视频的 detector-only 变体。 |
| generation_variant_independent_video_decision | governance | none | true | true | false | 全部生成机制变体是否使用互不重叠的独立轨迹。 |
| generation_variant_provenance_failure_variants | governance | none | true | false | false | 缺少独立生成来源证据的生成机制变体。 |
| generation_variant_trace_overlap_pairs | governance | none | true | false | false | 错误共享同一 trajectory trace 的生成机制变体对。 |
| internal_ablation_video_reuse_policy_passed | governance | none | true | true | false | 当前 profile 的逐变体样本量与视频复用策略是否通过公共闭合器。 |
