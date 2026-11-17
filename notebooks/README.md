# Kaggle Notebook 清单

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
