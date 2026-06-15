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
| generation_model_version | protocol | none | true | false | false | B5 generation model version. |
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
