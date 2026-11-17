# affordance-shortcut · 可供性预测中的类别捷径

> **定位**：机械臂项目（SO-ARM101）的垂直领域小模型主线。复现并扩展自己的科研基线：
> **「可供性预测中的类别捷径：诊断与轻量去偏」**（*What Can It Do, or What Is It?*）。
>
> **上游方案**：`D:\Myproject(AI)\downloads\affordance-screening\KAGGLE_RESEARCH_PLAN.md`
> **建立日期**：2026-11-17 ｜ **当前阶段**：Phase 0（数据冒烟，不占 GPU）

## 1. 这个项目要回答什么

现有多模态可供性定位模型在 Seen 物体类别上的高分，有多少来自"认出了物体类别"这条捷径，
而不是真正的"功能区域理解"？

- 主实验：AGD20K 上 Seen / Easy-Unseen / Hard-Unseen 三划分
- 诊断探针：文本换类、同类异功能、Prior Ratio
- 轻量去偏：类别先验减法（E3）、反事实文本增强（E4）、类别对抗头（E5，可选）

## 2. 与机械臂项目的接口

```
阶段一（现在）：图像 → 可供性热图（本仓库，Kaggle 训练）
阶段二（机械臂可用后）：热图 → 手眼标定 → SO-ARM101 抓取点 → 1-B 视觉分拣
```

本仓库**不依赖机械臂进度**；机械臂只是后期的部署载体。模型设计为轻量（冻结 CLIP + 5M 解码器），
后期可量化后部署到 Jetson Orin Nano。

## 3. 目录结构

```
affordance-shortcut/
├── configs/
│   ├── base.yaml              # 主配置（数据/模型/训练/评测）
│   └── debias.yaml            # E3/E4/E5 去偏实验配置
├── data/processed/            # meta.csv 等轻量索引（不入库的原始数据见第 4 节）
├── docs/
│   └── 路线决策_2026-11-17.md  # 三条路线对比与决策记录
├── notebooks/                 # Kaggle Notebook（00–06，Phase 1 后补）
├── results/                   # metrics.csv / figures/ / runs/
├── scripts/
│   ├── fetch_hf_sample.py     # 下载 HF 子集（Phase 0）
│   ├── prepare_meta.py        # 原始数据 → meta.csv
│   └── smoke_check.py         # 环境与数据自检
└── src/
    ├── data.py                # Dataset / prompt 构造 / 划分
    ├── model.py               # 冻结 CLIP + FiLM 解码器（E2）
    ├── baselines.py           # E0 类别平均 / E1 CLIP 零样本
    ├── train.py               # 训练入口
    ├── eval.py                # IoU/NSS/KLD/SIM
    └── probes.py              # 三组诊断探针
```

## 4. 数据布局

原始数据不放在仓库里，默认位于：

```
D:\Myproject(AI)\downloads\affordance-datasets\
├── iffyuan-agd\               # Phase 0 冒烟子集（1,543 条）
│   ├── data\train-00000-of-00001.parquet   # 204.3MB，图片/掩码内嵌
│   ├── extracted\inline_images\            # 1,543 张（从 parquet 解出）
│   └── extracted\inline_masks\             # 1,543 张二值掩码
└── agd20k\                    # Phase 1 主实验（6.7GB，解压后转 256px 紧凑版）
```

**Phase 0 实测（2026-11-17）**：

| 项 | 值 |
|---|---|
| 样本数 | 1,543（parsed 1,543 / paired 1,543 / missing 0） |
| 物体类别 | 49 |
| 可供性（verb） | 36 |
| 冒烟划分 | train 1,246 / val 297（按 id 哈希，非官方协议） |
| 首样本 | `apple / cut`，640×480，掩码二值、正样本 28.8% |

> 注意：HF 子集的随机划分只用于冒烟，**不得写进报告结果**；正式实验使用 AGD20K
> 的 Seen / Easy-Unseen / Hard-Unseen 对象类别协议。

覆盖方式：环境变量 `AGD_DATA_ROOT`，或 `configs/base.yaml` 的 `data.root`。

## 5. 快速开始

```powershell
# 0. 环境（本机 spatial-gesture venv：torch 2.13 CPU + pyarrow 26）
D:\PythonEnv\spatial-gesture\Scripts\python.exe -c "import torch, pyarrow, PIL, numpy"

# 1. 下载子集（httpx 流式，默认走本机代理 127.0.0.1:7892，可用 --proxy 覆盖）
python scripts/fetch_hf_sample.py --proxy http://127.0.0.1:7892

# 2. 生成 meta.csv 并做完整性校验（parquet 内嵌模式，自动跳过 tar）
python scripts/prepare_meta.py --subset iffyuan_agd

# 3. 自检（报告写入 results/smoke_report.txt）
python scripts/smoke_check.py
```

> 本机 venv **没有 open_clip / timm**，模型代码只做了语法检查；首次真实前向与训练
> 在 Kaggle 上执行：`pip install open_clip_torch timm` → `python -m src.train --config configs/base.yaml`。

## 6. 算力预算（已确认）

| 项目 | 额度 | 本方案用量 |
|---|---|---|
| Kaggle GPU | T4 ×2（各 16GB），30 h/周，单会话 12 h | 约 7–10 h/周，4 倍余量 |
| 阿里云学生券 | ¥300 | 储备，暂不动用 |
| 按小时租卡 | 3090/A10 约 ¥1.5–2/h | 仅排不到卡时用 |

## 7. 实验矩阵（与上游方案一致）

| 编号 | 名称 | 输入 | 训练 | 作用 |
|---|---|---|---|---|
| E0 | 类别平均掩码 | 仅类别名 | 无 | 捷径量化基线 |
| E1 | CLIP 零样本 | 图像 + 文本 | 无 | 便宜的对齐基线 |
| E2 | CLIP + 解码器 | 图像 + 文本 | 有 | 主模型 |
| E3 | 类别先验减法 | E2 − E0 | 无 | 零成本去偏 |
| E4 | 反事实文本增强 | 随机替换物体名词 | 有 | 轻量去偏 |
| E5 | 类别对抗头 | 瓶颈 + 梯度反转 | 有 | 可选 |

## 8. 时间线

| 阶段 | 时间 | 内容 |
|---|---|---|
| Phase 0 | 2026-11-17 – 12-19 | 仅数据管道与冒烟，每周 ≤2 h，不占 GPU（四级优先） |
| Phase 1 | 12-20 起 | AGD20K 下载 → 紧凑版 → E0/E1 |
| Phase 2 | 寒假第 1 周 | E2 训练 + 三划分评测 |
| Phase 3 | 寒假第 2 周 | 探针 + E3 |
| Phase 4 | 寒假第 3 周 | E4 + 图表 |
| Phase 5 | 寒假第 4 周 | 4 页报告 + README + 大创段落 |

## 9. 相关项目

- `D:\Myproject(AI)\llm-finetune\`：保留为**支线 A**（机器人领域 LLM），暂不投入。
- `D:\Myproject(AI)\experience-affordance\`：第三层 state-history 原型，不混入本线。
- `D:\Myproject(AI)\机械臂项目\`：SO-ARM101 + ROS2 工程线，后期部署目标。
