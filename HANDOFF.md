# 项目交接文档 · 2026-11-17

> **主线**：可供性类别捷径（affordance-shortcut，路线 C）
> **当前阶段**：Phase 0（数据冒烟），四级（2026-12）前不动 GPU 大训练
> **状态一句话**：Phase 0 已闭环——本机 CPU 冒烟跑通，代码链路验证通过；Kaggle 提交版本问题不再阻塞主线。

---

## 0. 2026-11-18 更新：Phase 0 冒烟已在本机跑通

| 项 | 结果 |
|---|---|
| 环境 | `D:\PythonEnv\affordance`（Python 3.11 / torch 2.15 CPU / open_clip 4.2.2 / timm 1.0.31） |
| 权重 | `open_clip_model.safetensors` 570.8MB，SHA256 与 HF LFS **完全一致**（`3f25d29d…f227f1`） |
| 前向 | 5 样本 / 4.5s / CPU |
| mean IoU | **0.129** |
| mean NSS | **−0.435** |
| mean KLD / SIM | 2.370 / 0.125 |
| 产物 | `results/smoke_e1_metrics.csv`、`smoke_e1_summary.json`、`smoke_e1_heatmaps.png` |

**结论**：热图落在物体上（自行车/酒瓶/杯子最明显），只是覆盖整个物体而非交互点；
E1 是弱基线，低 IoU 属预期。**代码链路（patch API → visual.proj → 文本相似度 → 指标）已验证。**

**Kaggle**：不再阻塞主线。需要公开复现记录时，用修复版重新 `Save & Run All` 即可，结果应与本机一致（仅速度不同）。

**下一步**：进入等待期，四级优先；**2026-12-20 之后**再启动 AGD20K 下载与 Phase 1。

---

## 1. 今天完成的事

| 类别 | 内容 | 状态 |
|---|---|---|
| 路线决策 | 三路线对比后锁定 C（科研基线 > 复试 demo > 硬件工具） | ✅ |
| 仓库 | 新建 `affordance-shortcut`，GitHub public，main 分支 | ✅ |
| 数据 | IffYuan/AGD 子集 1,543 条，49 物体 / 36 可供性，0 缺失 | ✅ |
| 数据管道 | `fetch_hf_sample.py`、`prepare_meta.py`、`smoke_check.py` | ✅ |
| 代码骨架 | 冻结 CLIP + FiLM 解码器、E0/E1 基线、评测、探针 | ✅ 语法检查 |
| 冒烟包 | Notebook `notebooks/01_smoke_e1.ipynb` + 5 样本 zip（501KB） | ✅ |
| Kaggle 冒烟 | 第一次：einsum 768 vs 512（已修）；第二次：SyntaxError（旧版本） | ❌ 待重跑 |
| 侧线 | GitHub 落地页 batch 1：adrian-portfolio + compound-interest | ✅ 已推送 |

---

## 2. 当前阻塞与下一步（明天开工照做）

### 2.1 Kaggle 冒烟（唯一阻塞）

**现状**：编辑器里的源码已是修复版，但队列里失败的那次用的是旧版本（日志明确报 `heatmapseatmaps.min()`）。

**做法（二选一）**：

1. **重新提交（推荐，产物可归档）**
   - 打开 Kaggle notebook 编辑器；
   - 确认 cell 6 最后一行是：
     ```python
     print('heatmaps:', heatmaps.shape, '| min/max:', float(heatmaps.min()), float(heatmaps.max()))
     ```
   - `Save Version → Save & Run All (Commit)`；
   - 等待队列（T4 ×2 高峰可能十几分钟到半小时）；
   - 结束后在「输出」标签查看 `smoke_e1_metrics.csv`、`smoke_e1_heatmaps.png`。

2. **交互会话直接 Run All（更快，不排队）**
   - 如果编辑器会话还活着，直接 Run All；
   - 结果截图或复制文本即可，事后再补提交。

**通过标准**：

```
patch grid: (1, 768, 14, 14) | side: 14
heatmaps: (5, 224, 224) | min/max: ...
mean IoU: x.xxxx | mean NSS: x.xxxx
```

E1 是弱基线，IoU 偏低正常；主要看热图是否大致落在物体附近。

### 2.2 冒烟通过后

- 把 `smoke_e1_metrics.csv` / `smoke_e1_heatmaps.png` 归档到 `results/`；
- 更新 README 的「项目状态」；
- 提交推送。

### 2.3 冒烟失败时

把**失败 cell 的完整 traceback** 和 open_clip 版本号发出来。已知版本：`open_clip_torch 4.2.2` / `timm 1.0.30`。

---

## 3. 关键路径

| 用途 | 路径 |
|---|---|
| 仓库 | `D:\Myproject(AI)\affordance-shortcut` |
| GitHub | https://github.com/ww0099/affordance-shortcut |
| Kaggle Notebook | https://www.kaggle.com/code/piolwxz/notebook0a88d23945 |
| 原始数据 | `D:\Myproject(AI)\downloads\affordance-datasets\iffyuan-agd`（617MB） |
| meta 索引 | `affordance-shortcut\data\processed\meta.csv`（1,543 行） |
| 冒烟数据包 | `D:\Myproject(AI)\downloads\affordance-datasets\affordance-smoke5.zip`（501KB） |
| 冒烟 Notebook | `affordance-shortcut\notebooks\01_smoke_e1.ipynb` |
| 自检报告 | `affordance-shortcut\results\smoke_report.txt` |
| 路线决策 | `affordance-shortcut\docs\路线决策_2026-11-17.md` |
| 上游方案 | `D:\Myproject(AI)\downloads\affordance-screening\KAGGLE_RESEARCH_PLAN.md` |

Kaggle 数据集：`piolwxz/datelearn`（私有，内含 smoke5 的 images/masks/smoke_meta.csv）。

---

## 4. 已踩的坑（别重复）

1. **open_clip 标准 ViT 没有 `.trunk`**
   标准 ViT-B-16 的视觉塔是自定义 `VisionTransformer`，不是 timm 模型。
   正确入口：`visual.forward_intermediates(..., indices=1, normalize_intermediates=True, intermediates_only=True, output_fmt='NCHW')`。
   `src/model.py` 已实现三条回退路径（timm trunk / forward_intermediates / 旧版手动）。

2. **patch tokens 与文本维度不同**
   patch 特征 768 维，文本 512 维；必须乘 `model.visual.proj`（768→512）后再做余弦相似度。
   已在 `src/baselines.py` 与 Notebook 中修复。

3. **Kaggle 提交的是"保存的版本"，不是编辑器当前状态**
   改完代码必须重新 `Save Version → Save & Run All`；否则队列跑的还是旧代码。

4. **粘贴截断**
   从聊天复制长代码时，最后一行容易被吞字符（本次就是 `heatmaps.shape` 被截断）。粘贴后务必核对最后一行。

5. **HF 子集掩码是粗粒度热图块**
   IffYuan/AGD 的 mask 是围绕交互点的椭圆/点簇，不是 AGD20K 的精细部件掩码。
   **只能用于冒烟**，正式实验必须换 AGD20K + 官方 Seen/Easy/Hard 协议。

6. **下载与推送需要代理**
   代理开着时：HF 下载、`git push` 都要走 `127.0.0.1:7892`。
   `huggingface_hub` 1.x 的 Xet 传输不走代理，已改用 httpx/curl 直连。

7. **数据只放 D 盘**
   原始数据、zip、缓存、Kaggle 配置都不要落到 C 盘。

---

## 5. 后续排期

| 时间 | 任务 | 说明 |
|---|---|---|
| 本周内（≤2h） | Kaggle 冒烟出结果 | 只验证代码通路，不做结论 |
| 2026-12 全月 | **四级优先** | 不占 GPU、不扩实验 |
| 2026-12-20 之后 | Phase 1：AGD20K 下载 → 256px 紧凑版 → 私有 Dataset → E0/E1 | 需先定 Google Drive 还是百度网盘 |
| 寒假第 1 周 | Phase 2：E2 训练 + 三划分评测 | Kaggle T4×2 |
| 寒假第 2 周 | Phase 3：探针 + E3 | |
| 寒假第 3 周 | Phase 4：E4 + 图表 | |
| 寒假第 4 周 | Phase 5：4 页报告 + README + 大创段落 | |

**侧线（不阻塞主线）**：

- Batch 2：pocket-agent 落地页（缺终端演示素材）；
- Batch 3：affordance-shortcut + web-auth 落地页（前者可生成真实掩码示例图，后者需架构图）。

---

## 6. Git 状态

```
main ... origin/main  同步
592ee47 feat: 建立可供性类别捷径主线仓库（Phase 0 数据管道）
cfa479c docs: 补充仓库维护与推送约定
21ea1d6 feat(smoke): 添加 Kaggle E1 冒烟 Notebook 与 5 样本打包脚本
a84c3a0 fix(smoke): E1 前将 patch tokens 投影到文本空间（768→512）
```

维护约定：每完成一批改动 → 中文 commit → `git push origin main`。

---

## 7. 下次会话开场建议

1. 先问："Kaggle 冒烟跑完了吗？"拿到 `mean IoU / NSS` 与热图；
2. 若通过：归档产物、更新 README、结束 Phase 0；
3. 若失败：看 traceback，优先检查版本是否最新、维度是否一致；
4. 不要提前启动 AGD20K 下载（12-20 前）。
