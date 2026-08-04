# Virtual Staining 工程化重构计划

> **状态：已执行完成（2026-08-04）。** 本文件为重构前的计划与历史记录，保留作参考。当前实际代码结构以 README.md「项目目录结构」章节为准。

> 目标：降低代码耦合，为后续「单独替换 数据处理 / 模型 / loss / 训练策略 / 推理流程」铺路。
> 约束：**不破坏当前已有功能**，运行方式保持 `python train.py` / `python train.py --resume` / `python predict.py` 不变。

---

## 1. 当前代码结构

```
virtual_staining/
├── train.py           # 241 行，训练一切逻辑
├── predict.py         # 194 行，推理一切逻辑
├── src/
│   ├── __init__.py    # 空
│   ├── config.py      # 只有路径 + 设备，train.py 根本没用它
│   ├── dataset.py     # VirtualStainingDataset
│   ├── model.py       # UNet + DoubleConv
│   ├── loss.py        # ReconstructionLoss
│   └── utils.py       # checkpoint 保存/加载 + SSIM/PSNR/Score（两类职责混在一起）
├── checkpoints/
│   └── best_score_model.pth
├── data/
│   ├── train/{DAPI, CD68, ...}
│   └── test/DAPI
└── results/
```

---

## 2. 当前代码问题（耦合点分析）

### 2.1 双路径真相源

- `src/config.py:16-20` 定义了 `CHECKPOINT_DIR / MODEL_PATH / INPUT_DIR / OUTPUT_DIR`，**只有 predict.py 使用**。
- `train.py:31-33` 又**硬编码**了一套完全相同的路径，train.py 完全不读 config.py。
- 后果：改路径要改两处，极易不一致。

### 2.2 train.py 承担全部逻辑

`train.py` 一个脚本同时负责：argparse → 路径 → 数据集 → train/val 划分 → DataLoader → 模型 → loss → optimizer → resume → 训练循环 → 验证循环 → best/last 保存。任何一块要替换都牵动整个文件。其中：

- `train.py:40-61` 验证逻辑 `validate()` 内嵌在脚本里。
- `train.py:89-121` 数据路径、batch_size、num_workers、0.9 划分比例、seed=42 全部写死在脚本。
- `train.py:127-138` 模型/loss/optimizer/epochs 全部写死在脚本。
- `train.py:145-156` resume 逻辑（文件检查 + 调用 load_checkpoint）耦合在脚本。
- `train.py:210-230` best 判定、时间戳 best 保存、best_score_model.pth 复制、last 保存四件事挤在一起。

### 2.3 训练策略无法替换

训练循环 `train.py:162-233` 是过程式代码，没有 Trainer 抽象。想换学习率调度 / AMP / early stopping / 多阶段训练，只能改这个循环，且每次改动都同时碰到数据、保存、验证逻辑。

### 2.4 utils.py 职责混杂

`src/utils.py` 同时包含：
- checkpoint 相关（`save_checkpoint` / `save_best_checkpoint` / `load_checkpoint`，11-102 行）
- 指标相关（`calculate_ssim` / `calculate_psnr` / `calculate_score`，109-135 行）

两类职责互相纠缠：改 checkpoint 逻辑会波及指标代码，反之亦然，不利于单独演进。

### 2.5 两套 SSIM 语义不一致（风险点，本次不动）

- 训练约束用的 SSIM：`src/loss.py:40-96` 的 `simple_ssim`，**全局均值版**（对一个 batch 取全局均值/方差）。
- 验证指标用的 SSIM：`src/utils.py:109-112` 的 `calculate_ssim`，即 `1 - MSE`。
- 两者含义完全不同。当前是刻意保留的 baseline 行为，本次重构**不修改**，但会在未来单独替换 loss 时重点核对，避免误伤验证指标。

### 2.6 训练/推理预处理两套实现

- 训练侧：`src/dataset.py` 用 `cv2.resize` + 手动归一化。
- 推理侧：`predict.py:30-37` 用 `torchvision.transforms.Resize(256)` + `ToTensor`。
- 存在 256 尺寸一致但插值方式可能不同的隐患。本次保持各自行为不变，仅把代码归位到独立模块，后续再统一。

### 2.7 best 保存逻辑重复

`train.py:213-221`：先 `save_best_checkpoint`（时间戳命名写入 CHECKPOINT_DIR），再 `save_checkpoint` 复制一份到 `best_score_model.pth`。两份保存逻辑重复、顺序有依赖。重构后收敛到 checkpoint 模块内部一次完成。

---

## 3. 推荐的新目录结构

```
virtual_staining/
├── train.py                   # 薄入口：解析参数 → 组装 config → 调 Trainer.fit()
├── predict.py                 # 薄入口：解析参数 → 调 Inference.predict_dir()
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py              # [改造] 统一配置：路径 + 全部超参（batch/epoch/lr/seed...）
│   ├── data/
│   │   ├── __init__.py        # 对外导出 build_train_val_loaders / build_inference_loader
│   │   ├── dataset.py         # [迁移] VirtualStainingDataset（原 src/dataset.py）
│   │   ├── transforms.py      # 训练/推理预处理函数（保持各自原有行为）
│   │   └── loaders.py         # DataLoader 构建：划分比例、seed、num_workers、batch 集中于此
│   ├── models/
│   │   ├── __init__.py        # build_model() 工厂：按 cfg 选择模型，返回 nn.Module
│   │   ├── unet.py            # [迁移] UNet + DoubleConv（原 src/model.py，结构一行不改）
│   │   └── registry.py        # [新增] 模型注册表（可选，为后续换模型做准备）
│   ├── losses/
│   │   ├── __init__.py        # build_loss() 工厂
│   │   └── reconstruction.py  # [迁移] ReconstructionLoss（原 src/loss.py，行为不改）
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── ssim_psnr.py       # [迁移] calculate_ssim/psnr/score（原 utils.py 中指标部分）
│   ├── checkpoints/
│   │   ├── __init__.py
│   │   └── saver.py           # [迁移] save_checkpoint / save_best_checkpoint / load_checkpoint
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── trainer.py         # [新增] Trainer 类：训练循环 + 验证循环 + resume + best/last 保存
│   └── inference/
│       ├── __init__.py
│       └── predictor.py       # [新增] 推理流程：加载模型 → 遍历输入 → 保存结果
└── checkpoints/               # 路径不变，已有 best_score_model.pth 继续可用
```

**设计要点：**
- 每个可替换维度一个独立子包，替换某一模块只动对应子包。
- `models/__init__.py` 提供 `build_model(cfg)` 工厂：当前返回 UNet，未来换 Attention U-Net / UNet++ 只需改 registry 与工厂。
- `losses/__init__.py` 提供 `build_loss(cfg)`：当前返回 ReconstructionLoss，未来加 Perceptual Loss 只需注册。
- `checkpoints/saver.py` 是对外唯一 checkpoint 接口，禁止在别处直接 `torch.save`。
- `metrics/` 与 `checkpoints/` 彻底分离，解决 2.4。
- `config.py` 成为唯一配置源，解决 2.1。

---

## 4. 分步修改计划

> 每步都保持「可运行」。建议按序执行，每步结束跑一次最小验证。

### Step 0 — 冻结基线（不动代码）
- 确认当前 `python train.py`（或已有训练中断点）能跑通，`predict.py` 能出结果。
- `git status` 确认工作区；如有必要先提交当前 baseline 作为回退点。
- 全局确认 `src` 模块仅被 `train.py` / `predict.py` 引用（已核实，无第三方引用）。

### Step 1 — 统一 config
- 改造 `src/config.py`：补齐全部超参（`batch_size=4`、`epochs=20`、`lr=1e-4`、`ssim_weight=0.5`、`val_ratio=0.1`、`seed=42`、`image_size=256`、`num_workers=0`）+ 原路径常量。
- `train.py` 删除本地硬编码路径常量，改从 `src.config` 读取。
- 风险：低。只涉及路径读取方式，行为不变。

### Step 2 — 拆解 utils.py
- 新建 `src/metrics/ssim_psnr.py`：迁移 `calculate_ssim / calculate_psnr / calculate_score`，**实现原样搬移**。
- 新建 `src/checkpoints/saver.py`：迁移 `save_checkpoint / save_best_checkpoint / load_checkpoint`，**逻辑与字段原样搬移**。
- `train.py` 改为从新模块 import；**删除 `src/utils.py`**。
- 风险：低。`utils.py` 无外部使用者，唯一消费者 train.py 同步更新。
- 兼容保证：checkpoint 字典字段、文件名规则、load 逻辑与现在完全一致（详见第 6 节）。

### Step 3 — 拆分 data 子包
- `src/dataset.py` 迁移到 `src/data/dataset.py`（类行为不改）。
- 新增 `src/data/transforms.py`（训练预处理函数，保持原 cv2 行为）。
- 新增 `src/data/loaders.py`：`build_train_val_loaders(cfg)`（含 0.9/0.1 划分、seed=42、batch、num_workers 全部收口到这里）与 `build_inference_loader`。
- `train.py` 的数据加载代码替换为调用 `build_train_val_loaders`。
- 风险：中低。重点核对划分 seed 与读取顺序与原脚本一致（用同一个 `manual_seed(42)` 的 generator）。
- 兼容保证：data 目录结构、文件名匹配、归一化不变，已有数据无需移动。

### Step 4 — 拆分 models 子包
- `src/model.py` 迁移到 `src/models/unet.py`（`UNet`、`DoubleConv` 类名与结构**一行不改**）。
- 新增 `src/models/registry.py` + `src/models/__init__.py` 的 `build_model(cfg)`，当前 `MODEL_REGISTRY = {"unet": UNet}`。
- `train.py` / `predict.py` 改为 `build_model(cfg)`。
- 风险：低。模型结构未变 → 已有 `best_score_model.pth` / `last_model.pth` 的 `model_state_dict` 键完全一致，可直接加载。
- 兼容保证：见第 6 节模型兼容条。

### Step 5 — 拆分 losses 子包
- `src/loss.py` 迁移到 `src/losses/reconstruction.py`（行为不改）。
- 新增 `src/losses/__init__.py` 的 `build_loss(cfg)`，当前 `LOSS_REGISTRY = {"reconstruction": ReconstructionLoss}`。
- `train.py` 改为 `build_loss(cfg)`。
- 风险：低。loss 仅被 train.py 引用，predict.py 不涉及。

### Step 6 — 抽 Trainer（核心）
- 新建 `src/trainer/trainer.py`，把 `train.py` 中的验证函数、训练循环、resume、best 判定、last/best 保存**原样搬入** `Trainer` 类：
  - `__init__(cfg, model, loss, device)`：组装 optimizer、epochs、best_score、start_epoch。
  - `fit(train_loader, val_loader, resume=False)`：resume 检查 → 循环 → 每 epoch 训练 + 验证 + best/last 保存。
  - `_validate(loader)`：原 `validate()`。
  - `_save_models(...)`：收敛 2.7 的重复保存（时间戳 best + best_score_model.pth + last_model.pth）。
- `train.py` 缩为薄入口：parse args → 组装 config → 建模型/loss/loader → `Trainer(...).fit(..., resume=args.resume)`。
- 风险：中。这是行为最敏感的一步；用 Step 6 后的对比验证（第 7 节）确保训练曲线、保存产物与重构前一致。
- 兼容保证：resume 的 epoch/optimizer/best_score 恢复逻辑保持原实现。

### Step 7 — 抽 Inference
- 新建 `src/inference/predictor.py`：`load_model(cfg)`、`predict_image(...)`、`predict_dir(...)`，把 `predict.py` 逻辑原样搬入。
- `predict.py` 缩为薄入口。
- 风险：低。predict.py 已用 config.py，迁移最直接。
- 兼容保证：仍加载 `best_score_model.pth`，输出文件名、输出目录不变。

### Step 8 — 回归验证 + 收尾
- 完整跑一遍：首次训练（小 epoch 冒烟）→ 中断 → `--resume` 续训 → 确认 epoch/best_score/optimizer 恢复 → 生成时间戳 best → `predict.py` 出图。
- 新旧 checkpoint 互测：用当前仓库已有的 `checkpoints/best_score_model.pth`（旧格式）让新 `predict.py` 加载，确认能出图。
- 更新 README 目录结构章节与 CLAUDE.md 中文件职责说明。

---

## 5. 修改风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| checkpoint 字段/文件名被改动导致旧档不可用 | 高 | Step 2 原样搬移；`load_checkpoint` 保留 `.get()` 容错；Step 8 用旧 best_score_model.pth 实测加载 |
| 训练/验证行为漂移（如 seed、loss 权重、划分） | 中 | 所有超参与逻辑在迁移中原值保留；Step 6 后对比训练输出数值 |
| 模型结构被无意改动 → 旧 checkpoint state_dict 失配 | 高 | Step 4 明确「结构一行不改」；用旧 checkpoint 加载验证键匹配 |
| import 路径遗漏（历史引用 src.utils 等） | 低 | 已确认仅 train.py / predict.py 引用 src；Step 0 再 grep 一遍 |
| 时间戳 best 命名/历史保存丢失 | 中 | `save_best_checkpoint` 命名规则原样保留，且 Step 6 收敛后做一次实际触发验证 |
| config 统一后改错默认值 | 低 | 默认值与现状逐一对照（batch=4/epochs=20/lr=1e-4/ssim_weight=0.5/seed=42） |
| 重构期间未提交基线，出错无回退点 | 中 | Step 0 先提交或记录当前可运行状态 |

---

## 6. 如何保证 checkpoint 兼容

**不变的东西（硬约束，禁止改动）：**

1. **checkpoint 字典字段**，保持原格式，新增字段可选：
   ```python
   {
     "epoch": epoch,
     "model_state_dict": model.state_dict(),
     "optimizer_state_dict": optimizer.state_dict(),
     "best_score": best_score,
     # 以下为可选：
     "timestamp": str,        # "%Y-%m-%d %H:%M:%S"
     "val_ssim": float,
     "val_psnr": float,
     "val_score": float,
   }
   ```
2. **文件名与用途**：
   - `last_model.pth`（可覆盖，`--resume` 续训用）
   - `best_score_model.pth`（可覆盖，predict 用）
   - `best_e{epoch:03d}_s{score:.4f}_{YYYYmmdd_HHMMSS}.pth`（时间戳，永不覆盖）
3. **加载容错**：`load_checkpoint` 继续用 `checkpoint.get("epoch", 0)` / `.get("best_score", -1.0)`，老格式（只有 4 个原始字段）兼容不变。
4. **模型结构**：重构期间 UNet 结构、层名、顺序完全不改 → `model_state_dict` 的键与形状不变，旧档可直接加载。
5. **恢复语义**：`load_checkpoint` 返回 `epoch + 1` 作为 `start_epoch`，best_score 覆盖当前值，optimizer 状态恢复 —— 全部原样保留。

**新旧 checkpoint 互兼容路径：**
- 旧档（当前已生成的 `best_score_model.pth`）→ 新 `predict.py`：只读 `model_state_dict` + 打印 `best_score/epoch`，可加载。
- 旧 `last_model.pth` → 新 `train.py --resume`：恢复权重/optimizer/epoch/best_score，可续训。
- 新档 → 若未来有人回退到旧代码：字段超集，旧 `load_checkpoint` 用 `.get()` 也能读。

---

## 7. 每步验收方式

| 步骤 | 验收 |
|---|---|
| Step 1 | `python train.py` 能启动，打印的路径/超参与重构前一致 |
| Step 2 | 训练保存的 checkpoint 字段、文件名与重构前完全一致 |
| Step 3 | 训练批次数量（每 epoch 步数）与划分结果与重构前一致 |
| Step 4 | 加载现有 `best_score_model.pth` 到新 `build_model` 成功、无 size mismatch |
| Step 5 | 训练 loss 数值量级与重构前一致 |
| Step 6 | 训练曲线近似、`--resume` 恢复日志（epoch/best_score）正确、best 与 last 文件均生成 |
| Step 7 | `predict.py` 对同一输入输出的图片与重构前逐像素一致 |
| Step 8 | 新旧 checkpoint 互加载 + 全链路回归 |

---

## 8. 本次重构明确不做的事

- 不改变 UNet 结构、不改 loss 公式、不改指标公式（两套 SSIM 不一致问题**记录**但本次不动）。
- 不加新功能：不加 AMP、scheduler、数据增强、perceptual loss。
- 不引入新依赖（不引入 PyYAML / hydra 等；config 保持 Python 模块，足够满足当前需求）。
- 不移动 data/ checkpoints/ results/ 目录位置。
- 不改变 `python train.py` / `--resume` / `python predict.py` 的调用方式。

## 9. 未来扩展点（重构带来的收益，供后续使用）

- 换模型：`src/models/registry.py` 注册新模型 → 重训即可。
- 加增强：改 `src/data/transforms.py` + `loaders.py`，不动 trainer。
- 换 loss：`src/losses/registry.py` 注册新 loss → 不动数据与模型。
- 改训练策略：Trainer 内替换训练循环（scheduler/AMP/early stop），不动数据/模型/loss。
- 换推理：改 `src/inference/predictor.py`，不动其余。
