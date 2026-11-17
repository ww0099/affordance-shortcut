# Kaggle Notebook 清单

## 01_smoke_e1（Phase 0，已就绪）

- Notebook：`01_smoke_e1.ipynb`
- 数据包：`D:\Myproject(AI)\downloads\affordance-datasets\affordance-smoke5.zip`（0.49 MB，5 样本）
- 使用步骤：
  1. Kaggle → **Datasets → New Dataset** → 上传 zip（名称 `affordance-smoke5`，选 Private）；
  2. Kaggle → **Create → Notebook → Import** → 选择 `01_smoke_e1.ipynb`；
  3. 右侧 **Add Input** → 选中 `affordance-smoke5`；
  4. Settings：**Internet = On**，Accelerator = **GPU T4 ×2**（CPU 也能跑）；
  5. Run All，约 5–10 分钟；结束后 **Save Version** 保存产物。
- 通过标准：打印 patch grid `(1, 768, 14, 14)`；5 张热图无报错；IoU / NSS 不全为 NaN。
- 说明：HF 子集掩码是交互点周围的粗粒度热图块，仅用于验证代码通路。

Phase 1 起在 Kaggle 上建立（T4 ×2，单会话上限 12 h）：

| 编号 | 类型 | 预计耗时 | 产出 |
|---|---|---|---|
| 00_env_check | CPU | 5 min | 版本、GPU、数据挂载检查 |
| 01_prepare_meta | CPU | 20 min | meta.csv + 完整性校验 |
| 02_baselines | GPU | 40 min | E0 类别平均 + E1 零样本 |
| 03_train_main | GPU | 2.5 h | E2 checkpoint + 三划分评测 |
| 04_probes | GPU | 1 h | 换类 / 同类异功能 / Prior Ratio |
| 05_debias | GPU | 2.5 h | E3 / E4（E5 可选） |
| 06_figures | CPU | 20 min | 图表 + metrics.csv |

约定：

- 固定 `seed=42`，`torch.backends.cudnn.deterministic=True`；
- 每次运行把 `metrics.csv` 与配置 hash 写入 `/kaggle/working`，Save Version 后可作为 Dataset 归档；
- 数据走私有 Kaggle Dataset（`agd20k-compact-v1`），不在 Notebook 里解压 6.7GB 原包。
