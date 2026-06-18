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
| attack_strength | protocol | none | true | false | false | Runtime attack strength descriptor. |
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
| video_decode_status | governance | none | true | false | false | Decode status for generated mp4 files used by B5 formal metrics. |
| video_metric_failure_reason | governance | none | true | false | false | Failure reason for generated video file metric extraction. |
| decoded_frame_count | metric | none | true | false | false | Number of decoded frames sampled from a generated video. |
| sampled_frame_count | metric | none | true | false | false | Number of frames sampled for B5 video file metrics. |
| mean_brightness | metric | none | true | false | false | Mean brightness over sampled generated video frames. |
| mean_contrast | metric | none | true | false | false | Mean contrast over sampled generated video frames. |
| dark_pixel_ratio | metric | none | true | false | false | Ratio of near-black pixels over sampled generated video frames. |
| bright_pixel_ratio | metric | none | true | false | false | Ratio of near-white pixels over sampled generated video frames. |
| motion_delta_score | metric | none | true | false | false | Mean adjacent-frame absolute difference over sampled generated frames. |
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
