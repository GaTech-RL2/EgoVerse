# EgoVerse 本地恢复版本 — 2026-11-27

本版本从 GitHub `rby1_aria_shared` 的 `8663e4c`（2026-10-27）出发，
合入 Mac 上保留的 11 月部署源码、训练模板和后续修复。
恢复期间没有连接 Skynet。它是可继续开发、可在本机验证的恢复分支；
不能证明与失去访问权限前的 Skynet 工作区逐字一致。

## 代码入口与来源

| 位置 | 恢复内容 | 来源及边界 |
|---|---|---|
| `egomimic/` | patch/transformer 点云编码器、分段采样、HPT/flow 及训练辅助改动 | Nov20 部署包与 GitHub 逐文件比较；保留 GitHub 中快照未覆盖的文件 |
| `campaign/policy.py`, `hd2_model.py`, `objectives.py`, `kinematics.py` | A/B/C/D 目标、DINO future head、FK、policy transfer | Nov19 PCD 部署包；policy 与 Nov25 模板一致 |
| `campaign/data.py` | 49-D action、22-D proprio、H64、DINO +10、尾部 mask、哈希与 mmap | Nov25 模板保留的完整源码 |
| `campaign/train.py`, `contracts.py` | fixed-target B 微调、60k、原完整状态 checkpoint 和验收约束 | Nov25 FT 模板；补回同日 Drawer 名称支持 |
| `campaign/photometric.py` | 参数化 strong MSC；可选 GeoAug | Nov19 原码 + 本地历史记录中 Nov23/25 的明确补丁 |
| `campaign/geoaug.py` | point dropout、米制 jitter、shared grayscale | Nov25 完整本地源码；仅按配置启用 |
| `campaign/dino_lora.py`, `train_lora.py` | inherited/frozen DINO head + rank8/alpha16 LoRA，其他参数 full FT | Nov25 模块原码和 `setup_lora.py` 的补丁重放 |
| `campaign/train_cloth.py`, `contracts_cloth.py` | 早期 Cloth PCD scratch/FT runner | Nov21 完整版本；不冒充最新 Nov25 scratch runner |
| `campaign/train_rgb.py`, `rgb_policy.py`, `rgb_data.py` | 独立 RGB variant | Nov20 RGB 部署包；与 PCD runner 分开 |
| `configs/hd2-reference.yaml`, `assets/robot.*` | PCD 网络结构、RBY1/XHand FK 模型 | Nov19 已部署包 |
| `scripts/fit_normalizer.py` | source-train-only 分位数统计与常量维保护 | Nov08 保留源码；不是所有最新 target-only/pooled adaptation 配方的替代品 |
| `deployment/robotpc/` | 部署支持、background observation / RTC 客户端 | Nov19/24 本地文件；未连接机器人，仍需按部署包布局配置路径 |
| `recovery/snapshots/` | preprocess、fix1124、BS、Cart/Drawer、Cloth recovery、LoRA、GeoAug、数据迁移、近期配置与验收证据 | 原文件按来源隔离，不作为可直接运行的本地流水线 |

原始来源和 SHA-256 在 `import-manifest.json`；恢复期间明确做过的修改在
`adaptations.json`；最终文件指纹在 `file-manifest.json`。
源文件的重复导入记录保留在 provenance 中，实际文件采用最后一个已说明的来源。
所有原有 Mac 数据、代码快照和部署包均保留。

## 在 Mac 上验证

```bash
cd /Users/chendongxin/Research-RL2/warp/EgoVerse-local-recovery
.venv/bin/python -m recovery.verify
```

已创建独立 Python 3.12 环境。重建环境可用：

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-local.lock.txt
```

`requirements-local.lock.txt` 是本机验证环境的精确包版本快照；不是遗失的
Skynet CUDA 环境锁。原始 `requirements.txt` 和 `environment.yaml` 保留。

验证工具运行单元行为检查、A/B/C/D CPU forward/backward/update、FK 与独立
kinematics 对照、paired initialization/transfer、gaze 检查及各 runner 的 CLI。
验证中的模型访问禁用 Hugging Face 下载，不启动 WebSocket 服务、训练任务或机器人。
结果写入 `recovery/runtime/validation.json`。本次结果另存 `validation.json`。

还可直接读取现存部署包，用恢复后的代码做 checkpoint 严格加载，以及三组
plain/RTC 动作与包内 CPU reference 的比较：

```bash
.venv/bin/python -m recovery.verify --checkpoint-package \
  ../SEW-Geometric-Teleop/docs/ai/artifacts/basket-ft10k-deploy-r2-20261119/basket_B180_MSCaug_FT10k-robotpc-20261119-r2
```

这验证的是 **Nov19 Basket B、10k checkpoint**，不代表 Nov25 最后一次训练的权重。
权重不复制进 Git。模型包含 68,101,864 个参数，输出 action horizon64 × dim49。

## 训练及数据状态

恢复的正式 runner 继续要求 CUDA/BF16、真实 release、normalizer/manifest 校验、
smoke 及对应验收；没有为了通过 Mac 验证而删除这些门槛。
`code_hash()` 已适配本仓库的 `egomimic/` 目录，避免旧 `source/egomimic/`
布局造成源码未计入。新仓库代码身份因此不同，不能伪造旧身份直接续训。

已保存的 Nov25 Cart/Basket scratch180k 配置、strong MSC、GeoAug 及 FT 配方在
`snapshots/` 下。它们保留历史状态和远端路径；此处没有生成新的 READY，
没有重新宣布历史任务处于 RUNNING，也没有把旧报告作为新版本通过全量 QC 的证明。

本机 Aria staging 有 352 个独立 VRS / 70.93 GB，含全部 Cloth new2 170 takes，
另有 5.64 GB MPS。165 个 Cloth take 有 SLAM/hand/gaze 文件，缺 MPS 的
123/132/152/157/160 原始 VRS 仍在。机器路径记录在被 Git 忽略的
`local-paths.json`；这里只记录位置，不搬动原数据。

## 尚未完整恢复的部分

1. **Nov24 fixed retarget 上游**：完整 `SEW-aria-pipeline`、EgoPoser handoff
   源码及权重、`aria_egoposer`/Base Smoothing S 的精确最新实现未在现有
   GitHub 基线和本地快照中找到。已保留调用、参数（CPF/measured wrists 等）
   和验收证据，但没有用推测实现替代。当前不能宣称 VRS→fixed+BS release
   可在本机端到端复现。
2. **最新点云/深度流水线的全部依赖**：本地有较早 HD2/cloud/RGBD 源码及
   Nov25 调用脚本，但 shared `real_dp3c_tri_gz_vis_hd2_fix1124_v2.py` 等完整
   上游文件和外部模型并不齐全。历史脚本的远端路径仍需要逐项恢复。
3. **Nov25–26 最新完整 checkpoint 与 processed release**：本地保留较早部署
   权重和数据清单，不能视为最新预训练/FT 全状态、优化器或训练数据的完整备份。
4. **最新所有 runner 的字节级一致性**：已恢复的 FT 模板可追溯；其他 campaign
   的独立 scratch/pretrain runner、256-mmap 等远端性能优化和持续处理 controller
   没有全部完整源码。现存早期 variant 与新近配置明确分开。
5. GitHub 的 `external/openpi` 子模块尚未下载；本次 PCD 恢复验证不依赖它。
   没有进行 CUDA、完整原数据 loader、图像/深度重建或机器人 rollout 验收。

这些缺口不能根据日志中的“passed”补写为已恢复。进一步发现本地完整文件时，
应带来源哈希合入并重新跑相关检查。
