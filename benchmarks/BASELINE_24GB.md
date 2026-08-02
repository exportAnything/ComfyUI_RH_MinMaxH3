# MiniMax-H3 24GB 优化基准

优化验收以本文件 + `matrix.json` 为准；85.6GB 多卡报告仅作证据。

## 采集步骤

1. Worker：单卡约 24GB，插件含 telemetry（`OPT_TELEMETRY=True`）。
2. 按 `matrix.json` 的 `cases` 跑工作流；**golden 格 `accel=off`**，固定 `seed=42`。
3. cold：首任务前确保 DiT 冷（`RESIDENCY_POLICY=safe` 或重启）；warm：同模型连续第二任务。
4. Decode 后在 Comfy `output/` 收集 `h3_*.json` sidecar（含 `telemetry.stages_s` / `step_host_s` / `peak_vram` / `residency_mode`）。
5. 汇总：

```bash
python3 benchmarks/run_matrix.py --sidecars /path/to/ComfyUI/output --out benchmarks/results/24gb
python3 benchmarks/run_matrix.py --list
```

6. 数值 golden（可选，需保存 latent `.pt`）：

```bash
python3 benchmarks/compare_golden.py --ref ref.pt --cand cand.pt --max-abs 2e-3 --cosine 0.9999
```

## 媒体闭环

每格有效条件：`ffprobe` 容器 OK + 音轨存在 + 首/中/尾抽帧非黑场。  
不允许仅凭 Comfy `SUCCESS`。

## 验收门槛

| 项 | 门槛 |
|---|---|
| 832x480 INT8 单步 P50（相对本基线） | ≥25% 下降（热路径优化后） |
| 热启动 dit_load（large / 24GB layerwise-warm） | large ≥35% 总时长；24GB 模型阶段 ≥50% |
| residency | 日志与 sidecar 可见；大卡不被锁进 layerwise |
| 峰值显存 | 相对同画布基线不升高 |

## 结果表（跑完后粘贴）

| case id | n | step P50 | step P95 | peak alloc | residency | 备注 |
|---|---:|---:|---:|---:|---|---|
| t2va_832x480_5s_int8_cold | | | | | | |
| t2va_832x480_5s_int8_warm | | | | | | |
