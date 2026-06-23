# 外部 baseline 适配边界

本目录用于保存 SSTW 项目的外部 baseline 适配器。它采用与 `D:\Code\SLM-WM` 相同的核心思想: 第三方方法本体与本项目的受治理输出分离, 由 adapter 把外部方法或显式同步控制转换为统一 records、tables、artifacts 和 reports。

## 接入规则

1. `primary/<baseline_id>/adapter/` 保存本仓库维护的轻量 adapter。
2. 若后续下载第三方官方源码, 应放在 `primary/<baseline_id>/source/` 或 `supplemental/<baseline_id>/source/`, 并由 `.gitignore` 排除。
3. adapter 只能写出受治理 observation 或 score records, 不能手工拼接论文结论。
4. 当前显式 DTW 与 frame matching adapter 属于工程级同步 control, 用于验证 baseline 对比链路闭合, 不能支持正向论文 claim。
5. 现代视频水印 baseline 在官方实现完成适配前, 只能写出 governed non-run record 和 comparison unsupported row。

## 与 `main/` 的职责区别

- `main/` 保存 SSTW 方法、协议字段和可复用算法。
- `external_baseline/` 保存外部方法到 SSTW 统一协议的适配入口。
- `experiments/` 负责在某次 run_root 中调度 adapter 并落盘比较结果。
