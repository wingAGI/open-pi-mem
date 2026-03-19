# open-pi-mem

`open-pi-mem` 目前聚焦在高层策略推理链路的验证，尤其是：

- 单帧高层推理：`goal + image + prev_memory -> next_subtask + next_memory`
- RMBench 视频评测：把短视频片段送进 Gemini，观察高层状态更新
- 本地 viewer：逐 step 查看输入 clip、prompt、prev memory、next subtask、next memory

这版仓库保留了两套最终结果，方便直接对比：

- `Gemini 3.1 Pro`
- `Gemini Robotics-ER 1.5`

结果目录在：

- [data/eval_results/Gemini_3_1_Pro](/Users/hex/workspace/具身智能/open-pi-mem/data/eval_results/Gemini_3_1_Pro)
- [data/eval_results/Gemini_Robotics_ER_1_5](/Users/hex/workspace/具身智能/open-pi-mem/data/eval_results/Gemini_Robotics_ER_1_5)

## 1. 本地模型高层推理

当前本地高层推理入口是：

- [scripts/run_high_level_inference.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_high_level_inference.py)

它支持：

- 从 YAML 加载高层模型配置
- 用 `--model-path` 覆盖成你自己的本地 VLM
- 用 `--checkpoint` 加载微调权重

如果你想直接用本地 `Qwen3-VL-8B` 做推理，可以参考：

```bash
PYTHONPATH=src python scripts/run_high_level_inference.py \
  --config configs/high_level_vlm.yaml \
  --model-path /absolute/path/to/Qwen3-VL-8B-Instruct \
  --local-files-only \
  --image examples/frames/frame_0.png \
  --goal "pick up the target object" \
  --prev-memory "" \
  --planner-hz 1
```

如果你已经有训练好的高层 checkpoint，可以继续加：

```bash
--checkpoint /path/to/high_level_checkpoint.pt
```

仓库里也放了一个本地模型配置示例：

- [configs/high_level_qwen3_vl_8b_local.example.yaml](/Users/hex/workspace/具身智能/open-pi-mem/configs/high_level_qwen3_vl_8b_local.example.yaml)

## 2. Google Gemini 视频推理

当前主入口是：

- [scripts/run_rmbench_high_level_episode_gemini_sdk_video.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_rmbench_high_level_episode_gemini_sdk_video.py)

这个脚本会：

- 读取一个 RMBench `episode0.mp4`
- 按规划频率切成短视频 clip
- 把 clip 送给 Gemini
- 输出 `report.json`
- 抽出每个 step 的预览帧，供 viewer 查看

说明：

- 下面这条命令依赖你本地已经准备好 `data/rmbench_local/...` 里的 RMBench 原始视频与 instruction JSON。
- 仓库默认不提交这部分原始数据，只提交最终可视化结果。

先设置 API key：

```bash
export GEMINI_API_KEY=your_key
```

然后运行一个示例：

```bash
PYTHONPATH=src python scripts/run_rmbench_high_level_episode_gemini_sdk_video.py \
  --video data/rmbench_local/observe_and_pickup_demo_clean/episode0.mp4 \
  --instruction-json data/rmbench_local/observe_and_pickup_demo_clean/episode0.json \
  --hz 1 \
  --model gemini-3.1-pro-preview \
  --max-output-tokens 10000 \
  --update-memory \
  --report-dir data/eval_results/demo_observe_and_pickup
```

也可以把模型换成：

```bash
--model gemini-robotics-er-1.5-preview
```

## 3. Viewer 使用方式

viewer 入口：

- [scripts/run_test_viewer_app.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_test_viewer_app.py)

启动：

```bash
python scripts/run_test_viewer_app.py --host 127.0.0.1 --port 8766
```

打开一个结果：

```text
http://127.0.0.1:8766/?report=data/eval_results/Gemini_3_1_Pro/press_button/report.json
```

或者：

```text
http://127.0.0.1:8766/?report=data/eval_results/Gemini_Robotics_ER_1_5/observe_and_pickup/report.json
```

viewer 里能看到：

- 当前 step 的输入帧动图
- 当前 step 的输入帧缩略图
- `Goal`
- `Previous Memory`
- `Next Subtask`
- `Next Memory`
- 完整 `Prompt`
- 原始 `Raw Output`

## 4. 代码入口

主要保留的文件：

- [scripts/run_high_level_inference.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_high_level_inference.py)
- [scripts/run_rmbench_high_level_episode_gemini_sdk_video.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_rmbench_high_level_episode_gemini_sdk_video.py)
- [scripts/run_rmbench_eval.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_rmbench_eval.py)
- [scripts/run_test_viewer_app.py](/Users/hex/workspace/具身智能/open-pi-mem/scripts/run_test_viewer_app.py)
- [src/open_pi_mem/rmbench/adapter.py](/Users/hex/workspace/具身智能/open-pi-mem/src/open_pi_mem/rmbench/adapter.py)
- [web/viewer](/Users/hex/workspace/具身智能/open-pi-mem/web/viewer)

旧的多帧实验脚本已经移到：

- [scripts/archive](/Users/hex/workspace/具身智能/open-pi-mem/scripts/archive)

## 5. 仓库说明

当前仓库默认不提交原始 RMBench 本地视频数据：

- [data/rmbench_local](/Users/hex/workspace/具身智能/open-pi-mem/data/rmbench_local)

但保留了最终可视化结果：

- [data/eval_results](/Users/hex/workspace/具身智能/open-pi-mem/data/eval_results)

这样更适合直接在 GitHub 上分享最终对比结果，同时避免把原始 episode 数据一起带上去。
