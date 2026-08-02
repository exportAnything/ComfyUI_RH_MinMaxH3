# MiniMax-H3 前端工作流

覆盖 `H3_TASKS` 的全部任务类型及其合法条件/参考形态，均为 ComfyUI 前端格式
（`nodes` + `links`），可直接拖入画布或用 **Load** 打开。

| 任务 | 工作流 | 条件素材 |
|------|--------|----------|
| T2VA | `t2va.json` | 无 |
| FL2VA | `fl2va_first_frame.json` | 首帧图片 |
| FL2VA | `fl2va_last_frame.json` | 尾帧图片 |
| FL2VA | `fl2va_first_last_frame.json` | 首帧 + 尾帧图片 |
| Ref2VA | `ref2va_image.json` | 参考图片 |
| Ref2VA | `ref2va_image_audio.json` | 参考图片 → 参考音频（有序链） |
| Ref2VA | `ref2va_video_audio.json` | 带音轨的参考视频 |

共同默认值：显式 `832×480`、5 秒、50 个 sigma 点、shift `12/3`、`accel=off`、
`denoise_video=true`。Loader 用的是文档里的 INT8 模型名，本地装的是 BF16 分片
或其它逻辑名时按下拉实际选项改。

## 运行前

1. **模型名**：三个 Loader 的 `model_root` 与各组件下拉需与本地一致。权重放置
   规则见根目录 README 的「模型目录」一节。
2. **素材**：`LoadImage` / `LoadAudio` / `LoadVideo` 里的文件名是占位符，替换成
   已上传到 ComfyUI `input/` 的真实文件。
3. **Ref2VA 参考顺序**：参考链严格有序，上一个参考节点的 `references` 输出接进
   下一个参考节点。改变链路顺序会改变多模态提示与条件行的顺序。
4. **Target 尺寸**：`width`/`height` 已显式填写。留空时按 `aspect_ratio` 解析，
   Ref2VA 会落到 1344×768，序列长度与耗时大幅上升。

## 重新生成

工作流由节点定义推导，不手写。改过 `INPUT_TYPES` 后重跑：

```bash
python3 tools/gen_example_workflows.py
```
