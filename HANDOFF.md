# 项目交接文档 · 更新至 2026-11-18

> **主线**：可供性类别捷径（affordance-shortcut，路线 C）
> **当前阶段**：Phase 0 已完成闭环 → 进入等待期（四级 2026-12 优先）
> **状态一句话**：数据、代码、本机冒烟、结果归档全部完成；主线无阻塞，12-20 前不再扩实验。

---

## 0. 2026-11-18 今日收尾

### 0.1 Phase 0 冒烟已在本机跑通

| 项 | 结果 |
|---|---|
| 环境 | `D:\PythonEnv\affordance`（Python 3.11 / torch 2.15 CPU / open_clip 4.2.2 / timm 1.0.31） |
| 权重 | `open_clip_model.safetensors` 570.8MB，SHA256 与 HF LFS **完全一致**（`3f25d29d…f227f1`） |
| 前向 | 5 样本 / 4.5s / CPU |
| mean IoU | **0.129** |
| mean NSS | **−0.435** |
| mean KLD / SIM | 2.370 / 0.125 |
| 产物 | `results/smoke_e1_metrics.csv`、`smoke_e1_summary.json`、`smoke_e1_heatmaps.png` |

**解读**：热图落在物体上（自行车/酒瓶/杯子最明显），只是覆盖整个物体而非精确交互点；
E1 是弱基线，低 IoU 属预期。**代码链路（patch API → visual.proj → 文本相似度 → 指标）已验证。**

**Kaggle**：不再阻塞主线。需要公开复现记录时，用修复版重新 `Save & Run All` 即可，结果应与本机一致。

### 0.2 本机系统检查（与测试相关）

| 检查项 | 结果 |
|---|---|
| 测试进程 | ✅ 无残留（`D:\PythonEnv\affordance` 的 python、curl、后台 Job 均已退出） |
| CPU / 内存 | Ryzen 5 7430U（6C/12T，负载 40%），29.8GB RAM / 14.7GB 空闲 |
| 磁盘 | Samsung SSD Health OK；C 剩 99.9GB，D 剩 477.7GB |
| Defender | 实时防护开启，Antivirus 正常 |
| AMD 驱动 | `31.0.21924.61`（OEM，2025-12-11），设备状态 OK，无错误码 |
| 显式驱动故障 | 7 天内 **无 4101 驱动重置、无 WHEA、无 BugCheck、无 Kernel-Power 41** |
| 卡顿原因判断 | 无系统级错误；最可能是加载 570MB 权重时 I/O + Defender 实时扫描造成的短暂停顿 |

**发现但未处理的系统项**（均与本次测试无关，未做任何修改）：

- `TurboXNetService.exe` 每天崩溃（缺少 Newtonsoft.Json），自动启动、当前已停止；
- `OrayIddDriver`（向日葵虚拟显示器）每次开机报 `WUDFRd 加载失败`，设备当前显示正常；
- `AiXiaobaoSvr` 服务文件缺失（自动、已停止）；
- AMD 驱动为 2025-12 的 OEM 版本，可按需通过联想官方渠道更新。

详细报告：`D:\Myproject(AI)\downloads\system-check-20261118\events.txt` 与 `events_targeted.txt`。

---

## 1. 项目现状

| 类别 | 内容 | 状态 |
|---|---|---|
| 路线决策 | 三路线对比后锁定 C（科研基线 > 复试 demo > 硬件工具） | ✅ |
| 仓库 | `affordance-shortcut`，GitHub public，main 分支 | ✅ |
| 数据 | IffYuan/AGD 子集 1,543 条，49 物体 / 36 可供性，0 缺失（617MB） | ✅ |
| 数据管道 | `fetch_hf_sample.py`、`prepare_meta.py`、`smoke_check.py`、`build_smoke_pack.py` | ✅ |
| 代码骨架 | 冻结 CLIP + FiLM 解码器、E0/E1 基线、训练/评测/探针 | ✅ 本机已验证 E1 |
| 本机冒烟 | `scripts/local_smoke_e1.py`，5 样本，结果归档 | ✅ |
| Kaggle | Notebook + 5 样本私有 Dataset；本机结果已覆盖其验证目的 | ✅ 可选补跑 |
| 侧线 | GitHub 落地页 batch 1：adrian-portfolio + compound-interest | ✅ 已推送 |

**主线无阻塞。** 今天不剩必须完成的技术动作。

---

## 2. 关键路径

| 用途 | 路径 |
|---|---|
| 仓库 | `D:\Myproject(AI)\affordance-shortcut` |
| GitHub | https://github.com/ww0099/affordance-shortcut |
| Kaggle Notebook | https://www.kaggle.com/code/piolwxz/notebook0a88d23945 |
| 本机专用环境 | `D:\PythonEnv\affordance`（torch 2.15 CPU / open_clip 4.2.2） |
| CLIP 权重缓存 | `downloads\hf-cache\open_clip\ViT-B-16-laion2b-s34b_b88k\open_clip_model.safetensors` |
| 原始数据 | `D:\Myproject(AI)\downloads\affordance-datasets\iffyuan-agd`（617MB） |
| meta 索引 | `affordance-shortcut\data\processed\meta.csv`（1,543 行） |
| 冒烟数据包 | `D:\Myproject(AI)\downloads\affordance-datasets\affordance-smoke5.zip`（501KB） |
| 冒烟结果 | `affordance-shortcut\results\smoke_e1_*` |
| 冒烟 Notebook | `affordance-shortcut\notebooks\01_smoke_e1.ipynb` |
| 自检报告 | `affordance-shortcut\results\smoke_report.txt` |
| 路线决策 | `affordance-shortcut\docs\路线决策_2026-11-17.md` |
| 上游方案 | `D:\Myproject(AI)\downloads\affordance-screening\KAGGLE_RESEARCH_PLAN.md` |
| 系统检查报告 | `D:\Myproject(AI)\downloads\system-check-20261118\` |

Kaggle 数据集：`piolwxz/datelearn`（私有，内含 smoke5 的 images/masks/smoke_meta.csv）。

---

## 3. 已踩的坑（别重复）

1. **open_clip 标准 ViT 没有 `.trunk`**
   标准 ViT-B-16 的视觉塔是自定义 `VisionTransformer`，不是 timm 模型。
   正确入口：`visual.forward_intermediates(..., indices=1, normalize_intermediates=True, intermediates_only=True, output_fmt='NCHW')`。
   `src/model.py` 已实现三条回退路径（timm trunk / forward_intermediates / 旧版手动）。

2. **patch tokens 与文本维度不同**
   patch 特征 768 维，文本 512 维；必须乘 `model.visual.proj`（768→512）后再做余弦相似度。
   已在 `src/baselines.py` 与 Notebook 中修复。

3. **Kaggle 提交的是“保存的版本”，不是编辑器当前状态**
   改完代码必须重新 `Save Version → Save & Run All`；否则队列跑的还是旧代码。

4. **粘贴截断**
   从聊天复制长代码时，最后一行容易被吞字符（本次就是 `heatmaps.shape` 被截断）。粘贴后务必核对最后一行。

5. **HF 子集掩码是粗粒度热图块**
   IffYuan/AGD 的 mask 是围绕交互点的椭圆/点簇，不是 AGD20K 的精细部件掩码。
   **只能用于冒烟**，正式实验必须换 AGD20K + 官方 Seen/Easy/Hard 协议。

6. **HF 官方 CDN 走代理极慢**
   实测 0.03 MB/s（570MB 需约 5 小时）；改用 `hf-mirror.com` 直连 + 8 线程分块，
   约 2–3 分钟完成。权重 SHA256 已与 HF LFS 核对。

7. **下载与推送需要代理**
   代理开着时：`git push` 要设置 `HTTP_PROXY/HTTPS_PROXY=127.0.0.1:7892`；
   `huggingface_hub` 1.x 的 Xet 传输不走代理，已改用 httpx/curl。

8. **数据只放 D 盘**
   原始数据、zip、缓存、Kaggle 配置都不要落到 C 盘。

---

## 4. 下一步

### 4.1 等待期（现在 – 2026-12-19）

- **四级优先**：不动 GPU、不扩实验、不再开新主线；
- 主线最多做维护性改动（文档、小修）；
- 侧线可选（不阻塞）：
  - Batch 2：pocket-agent 落地页（缺终端演示素材）；
  - Batch 3：affordance-shortcut + web-auth 落地页。

### 4.2 Phase 1（2026-12-20 之后）

1. 先决定 AGD20K 下载渠道（Google Drive 还是百度网盘）；
2. 下载 6.7GB → 转 256px 紧凑版 → 私有 Kaggle Dataset；
3. 跑 E0（类别平均）与 E1（CLIP 零样本）基线；
4. 正式实验使用 Seen / Easy-Unseen / Hard-Unseen 协议。

### 4.3 系统清理（可选，需用户确认）

- `TurboXNetService` 每天崩溃：不用就卸载/禁用；
- `AiXiaobaoSvr` 文件缺失：可禁用；
- `OrayIddDriver`：不用向日葵远程控制就卸载，用就更新；
- AMD 驱动：通过联想官方渠道更新，不直接装通用驱动。

---

## 5. 后续排期

| 时间 | 任务 | 说明 |
|---|---|---|
| 2026-11-18 ✅ | Phase 0 冒烟（本机 CPU） | 已完成，结果归档 |
| 2026-12 全月 | **四级优先** | 不占 GPU、不扩实验 |
| 2026-12-20 之后 | Phase 1：AGD20K → 紧凑版 → E0/E1 | 先定下载渠道 |
| 寒假第 1 周 | Phase 2：E2 训练 + 三划分评测 | Kaggle T4×2 |
| 寒假第 2 周 | Phase 3：探针 + E3 | |
| 寒假第 3 周 | Phase 4：E4 + 图表 | |
| 寒假第 4 周 | Phase 5：4 页报告 + README + 大创段落 | |

---

## 6. Git 状态

```
main ... origin/main  同步
592ee47 feat: 建立可供性类别捷径主线仓库（Phase 0 数据管道）
cfa479c docs: 补充仓库维护与推送约定
21ea1d6 feat(smoke): 添加 Kaggle E1 冒烟 Notebook 与 5 样本打包脚本
a84c3a0 fix(smoke): E1 前将 patch tokens 投影到文本空间（768→512）
7d97453 feat(smoke): 本机跑通 E1 零样本冒烟并归档指标与热图
```

维护约定：每完成一批改动 → 中文 commit → `git push origin main`。

---

## 7. 下次会话开场建议

1. **主线在等待期，不需要主动推进**；先确认四级备考进度；
2. 12-20 之后再启动 AGD20K 下载与 Phase 1，不要提前；
3. Kaggle 冒烟不必重跑——本机结果已归档，除非需要公开复现记录；
4. 侧线任务按需做：pocket-agent 落地页（#5）、affordance/web-auth 落地页（#6）；
5. 系统清理项需要用户明确同意后再动服务或驱动。

---

## 8. 备注：本次测试的安全性结论

- 本机测试使用独立 venv，没有修改系统环境或驱动；
- 权重来自 HF，SHA256 与官方 LFS 完全一致；
- 没有管理员级操作，没有残留进程；
- 系统日志无蓝屏、驱动重置或硬件错误记录。
