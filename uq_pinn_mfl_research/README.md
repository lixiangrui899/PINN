# UQ-PINN + 去噪 + MFL 审稿人约束版项目

## 项目定位

这个项目不是“先堆一个网络，再补几句 physics/UQ 解释”的 demo，而是一套按审稿人约束组织的漏磁研究原型。它的核心目标不是尽快产出一个数字，而是先把下面几件事做对：

- 明确区分“无标签过渡版”和“有标签正式版”
- 在没有几何真值时，禁止把 proxy 任务包装成缺陷参数反演
- 不把 repeated runs、滑窗片段当成独立样本
- 把 physics loss、discrepancy term 和 UQ 指标显式留在系统里
- 让真实数据流程和 synthetic 验证流程共享同一套 CLI 和训练编排

当前项目更准确的定位是：

- 对真实划痕漏磁数据：数据审计 + 去噪 + weakly supervised / proxy / PINN 研究平台
- 对 synthetic 带标签数据：监督反演与 UQ 评估闭环验证平台

## 当前结论

截至 `2026-04-13`，项目已经达到下面这个级别：

- 已完成真实数据的结构化审计、异常报告、配对分析、标签模板生成和严格 split 清单生成
- 已完成真实数据上的 Stage B 去噪和 Stage C proxy/PINN 烟雾训练
- 已完成基于真实统计特征的 synthetic MFL 数据生成
- 已完成 synthetic 带标签数据上的 Stage D 监督反演闭环
- Stage D 已支持 `target normalization`、`per-target metrics`、`failure cases` 导出和 `leave-one-position-out` 批量评估
- UQ 指标链路已接通，但还不能说校准质量已经成熟

一句话判断：

- 这套代码已经是“可做研究实验”的原型工程
- 还不是“真实漏磁缺陷几何反演已完成”的最终成品

## 核心约束

- 不能把重复测量当独立样本
- 不能把滑窗当样本数
- 不能在没有几何标签时宣称完成缺陷参数反演
- physics-informed 模块必须显式拆成 data loss、physics loss 和 discrepancy term
- UQ 模块必须保留 NLL、PICP、MPIW 和 calibration curve 接口

## 目录

```text
uq_pinn_mfl_research/
├─ configs/
├─ manifests/
├─ metadata/
├─ outputs/
├─ reports/
└─ src/uq_pinn_mfl/
   ├─ data/
   ├─ evaluation/
   ├─ models/
   ├─ simulation/
   └─ training/
```

## 模式说明

项目只有两种合法模式：

### 1. `weakly_supervised_proxy_mode`

当 `metadata/defect_labels.csv` 不存在、字段不全、或者目标列为空时，系统自动进入这个模式。在这个模式下：

- 允许做 Stage A 审计
- 允许做 Stage B 去噪
- 允许做 Stage C 配对一致性、proxy classification、physics-guided reconstruction、embedding 学习
- 禁止进入 Stage D 监督反演
- 禁止把结果写成 full inversion

### 2. `supervised_inversion_mode`

只有当 `metadata/defect_labels.csv` 存在且字段齐全时，才允许进入这个模式。在这个模式下：

- 可以训练 `length / width / depth / area` 的监督反演模型
- 可以输出 per-target MAE/RMSE
- 若使用 UQ 变体，可以输出 NLL、PICP、MPIW、calibration curve

## 已验证的数据现状

### 真实 `划痕缺陷` 数据

默认配置 [configs/default.yaml](configs/default.yaml) 指向真实目录。当前已验证结果如下：

- 总 CSV 数：`56`
- 物理独立位置：`5`
- 轴向/径向成功配对数：`26`
- 异常总数：`36`
- `位置4` 径向明显缺失，且存在 `200K`、`200K4` 采样标签，与大多数 `10K` 文件不一致
- 未发现外部几何真值标签，因此真实项目当前模式为 `weakly_supervised_proxy_mode`

这些信息来自：

- `reports/dataset_index.csv`
- `reports/pairing_report.csv`
- `reports/naming_anomaly_report.csv`
- `reports/acquisition_tag_summary.csv`
- `reports/data_readiness_report.md`
- `reports/project_mode.json`

### synthetic 带标签数据

项目已经支持生成与真实目录风格一致的 synthetic 数据集，并自动配套标签与 config。当前已验证的一套 synthetic demo 具备：

- `8` 个物理位置
- 每个位置 `4` 个 run
- 每个 run 都有 `axial + radial`
- 共 `64` 个 CSV
- 每个文件保留 `Index + AIN 1~AIN 8` 结构
- 自动生成 `defect_labels.csv`

## 当前效果级别

### 真实数据上的效果

#### Stage A：可直接使用

真实数据的结构化审计已经可用，而且这部分是目前最稳的。它已经解决了很多后续实验最容易出问题的地方：

- 文件名解析
- 位置与方向识别
- 轴向/径向配对
- acquisition tag 差异检查
- row count 异常检查
- readiness report 自动生成

#### Stage B：已达到“可研究”的去噪原型级别

当前 smoke-test 结果见 `reports/stage_b_metrics.json`，测试集指标大致为：

- reconstruction: `0.4551`
- gradient: `0.0299`
- peak_valley: `0.4572`
- spectral: `2.6728`
- total: `0.8198`

这说明去噪训练链路已经稳定可跑，loss 也不是单一的像素重建，而是更关注波形梯度、峰谷与频谱的一致性。当前还没有进入长训练和系统对比阶段，所以它代表“训练闭环已成立”，不代表“已得到最优去噪器”。

#### Stage C：已达到“可研究”的 proxy/PINN 原型级别

当前 smoke-test 结果见 `reports/stage_c_metrics.json`，测试集大致为：

- data_loss: `1.1540`
- physics_loss: `0.9744`
- discrepancy_loss: `0.2309`
- consistency_loss: `0.1383`
- position_loss: `4.9926`
- total: `3.7287`

这说明下面这些链路都已经打通：

- axial/radial paired consistency
- physics-guided reconstruction
- discrepancy-aware surrogate
- position-level proxy classification
- embedding 学习

但需要注意，`position_loss` 仍然偏高，这意味着当前 Stage C 更适合做方法探索和表征学习，还不适合拿来宣称已经得到稳定、高泛化的 defect-state 识别器。

### synthetic 数据上的效果

#### Stage D：已达到监督反演闭环验证级别

在 `synthetic_pinn_demo` 上，blackbox 变体已经完成 `leave-one-position-out` 批量评估。结果在 `reports/synthetic_pinn_demo/stage_d_lopo_blackbox.json`。

测试集 `LOPO mean` 大致为：

- `length`
  - MAE: `3.8384`
  - RMSE: `3.9355`
- `width`
  - MAE: `2.2951`
  - RMSE: `2.3179`
- `depth`
  - MAE: `1.1965`
  - RMSE: `1.2183`
- `area`
  - MAE: `55.8296`
  - RMSE: `56.1954`

这组结果说明：

- 监督反演训练、验证、测试和批量汇总都已经能稳定运行
- `depth` 和 `width` 在 synthetic 数据上相对更稳
- `area` 的尺度更大，且跨位置波动明显，是目前最难学的目标
- 结果已经不是单次随机切分，而是严格的 `leave-one-position-out`

需要强调的是，这仍然是 synthetic 数据结果，只能说明“方法闭环可用”，不能等价替代真实数据反演精度。

## 当前能做什么

- 对真实漏磁 CSV 做结构化审计并生成报告
- 生成标签模板并自动判断是否允许进入监督反演
- 对原始 8 通道信号做缓存、标准化和滑窗切分
- 用 repeated-run averaging 生成 pseudo clean target
- 训练 Stage B 去噪模型
- 训练 Stage C blackbox / deterministic PINN / UQ-PINN proxy 模型
- 生成 synthetic MFL 数据、合成标签和配套 config
- 在 synthetic 标签数据上运行 Stage D 监督反演
- 导出 per-target 指标、failure cases 和 LOPO 汇总

## 当前不能做什么

- 不能对真实 `划痕缺陷` 数据输出可信的几何反演结果
- 不能把真实 Stage C 的 proxy 结果包装成 full inversion
- 不能把窗口数、重复测量次数当成独立样本规模
- 不能把当前的 UQ 结果表述成“已完成严格校准”

## 独立样本定义

- `raw file`：一次原始 CSV 采集文件
- `repeated measurement`：同一物理位置的不同 run
- `sliding window`：从单个长序列切出的局部片段，只能作为训练片段，不能当独立样本
- `physical defect unit`：当前默认按 `position_id` 视作物理独立缺陷单位

所有 split 都必须在 `physical defect unit` 或严格配对组层级上做，而不是在窗口层级随机打散。

## 主要模块说明

### `src/uq_pinn_mfl/config.py`

负责：

- 读取 YAML 配置
- 构造 `ProjectContext`
- 统一管理 `data_root / reports / manifests / outputs / cache / checkpoints / predictions`
- 自动创建缺失目录

### `src/uq_pinn_mfl/cli.py`

统一暴露命令：

- `audit`
- `preprocess`
- `stage-b`
- `stage-c`
- `stage-d`
- `simulate`
- `mode`

这意味着所有真实数据流程和 synthetic 流程都可以从同一个 CLI 入口运行。

### `src/uq_pinn_mfl/data/audit.py`

这是 Stage A 的核心模块，负责：

- 递归扫描 `data_root`
- 解析 `position_id / orientation / run_id / timestamp / acquisition_tag`
- 检查 CSV 表头是否为 `Index + AIN 1~AIN 8`
- 统计行数并发现 row count outlier
- 基于 `position_id + normalized_run_id` 做轴向/径向配对
- 输出 pairing anomalies 与 acquisition tag mismatch
- 自动写 readiness report

### `src/uq_pinn_mfl/data/labels.py`

负责：

- 生成 `metadata/defect_labels_template.csv`
- 检查 `metadata/defect_labels.csv` 是否存在、列是否齐全、目标是否为空
- 根据标签状态决定项目模式是 `weakly_supervised_proxy_mode` 还是 `supervised_inversion_mode`

### `src/uq_pinn_mfl/data/preprocess.py`

负责：

- 读取原始 CSV 中的 `AIN 1~AIN 8`
- 做 `zscore_per_channel` 标准化
- 缓存为 `.npz`
- 生成 `preprocessed_index.csv`
- 基于 repeated runs 计算 pseudo clean target
- 生成 `window_index.csv`
- 生成 `paired_window_index.csv`

它还定义了三个数据集类：

- `SignalWindowDataset`
- `PairedWindowDataset`
- `LabeledSignalDataset`

分别服务于 Stage B、Stage C 和 Stage D。

### `src/uq_pinn_mfl/data/splits.py`

负责严格 split 与评估调度：

- `leave_one_position_out`
- `grouped_kfold_by_defect_unit`
- `strict_paired_split`
- `build_lopo_schedule`

这部分是防止样本泄漏的关键模块之一。

### `src/uq_pinn_mfl/models/denoising.py`

负责 Stage B 去噪模型与损失，包括：

- `moving_average_denoise`
- `ResUNet1D`
- `compute_denoising_loss`

当前 loss 由四部分组成：

- reconstruction
- gradient consistency
- peak/valley preservation
- spectral consistency

### `src/uq_pinn_mfl/models/surrogate.py`

这是整个项目最核心的模型层，包含：

- `BlackBoxProxyNet`
- `BlackBoxRegressor`
- `PhysicsGuidedSurrogate`

其中 `PhysicsGuidedSurrogate` 明确分出：

- `state`
- `physics_only_reconstruction`
- `discrepancy`
- `regression_mean`
- `regression_logvar`

也就是说，physics 项和误差补偿项不是混在一起的，而是显式建模。

### `src/uq_pinn_mfl/evaluation/uq.py`

负责 UQ 评估：

- `gaussian_nll`
- `prediction_interval`
- `picp`
- `mpiw`
- `calibration_curve`

### `src/uq_pinn_mfl/training/common.py`

负责：

- 随机种子
- 自动选择 CPU/CUDA
- JSON 保存
- 按位置或组做 train/val/test 划分

### `src/uq_pinn_mfl/training/pipelines.py`

这是训练总编排模块，负责：

- `run_stage_a`
- `run_stage_b`
- `run_stage_c`
- `run_stage_d`

也是当前最关键的工程模块。最新版本已经支持：

- Stage D 目标标准化 `TargetNormalizer`
- per-target MAE/RMSE
- UQ per-target 统计
- failure cases CSV 导出
- `--lopo` 批量评估

### `src/uq_pinn_mfl/simulation/realistic_mfl.py`

负责 synthetic 数据生成。它会：

- 从真实数据估计均值、标准差和 layout
- 用 charge-pair 风格缺陷响应生成 core signal
- 按 axial/radial 生成不同响应模式
- 注入 baseline、漂移、channel bias、noise
- 保留真实风格的 multi-cycle 拼接文件结构
- 自动生成 `defect_labels.csv`
- 自动生成 `synthetic_manifest.csv`
- 自动生成一份可以直接运行的新 config

## 训练与评估命令

在项目根目录执行：

```bash
python -m pip install -e .
python -m uq_pinn_mfl.cli --config configs/default.yaml mode
python -m uq_pinn_mfl.cli --config configs/default.yaml audit
python -m uq_pinn_mfl.cli --config configs/default.yaml preprocess
python -m uq_pinn_mfl.cli --config configs/default.yaml stage-b --model resunet1d
python -m uq_pinn_mfl.cli --config configs/default.yaml stage-c --variant uq_pinn
python -m uq_pinn_mfl.cli --config configs/default.yaml stage-d --variant uq_pinn
python -m uq_pinn_mfl.cli --config configs/default.yaml stage-d --variant uq_pinn --lopo
python -m uq_pinn_mfl.cli --config configs/default.yaml simulate --mode benchmark --dataset-name synthetic_pinn_demo
```

如果没有标签，`stage-d` 会被明确阻止，而不是偷偷退化成伪反演。

## Stage A 产物

- `reports/dataset_index.csv`
- `reports/pairing_report.csv`
- `reports/naming_anomaly_report.csv`
- `reports/acquisition_tag_summary.csv`
- `reports/data_readiness_report.md`
- `reports/project_mode.json`
- `metadata/defect_labels_template.csv`
- `manifests/leave_one_position_out.csv`
- `manifests/grouped_kfold_by_defect_unit_id.csv`
- `manifests/strict_paired_split.csv`

## Stage B / C / D 产物

- Stage B
  - `reports/stage_b_metrics.json`
  - `outputs/checkpoints/stage_b_*.pt`
- Stage C
  - `reports/stage_c_metrics.json`
  - `outputs/checkpoints/stage_c_*.pt`
- Stage D
  - `reports/stage_d_metrics.json`
  - `reports/stage_d_lopo_*.json`
  - `reports/*/failure_cases_*.csv`
  - `outputs/*/checkpoints/stage_d_*.pt`

## 标签接口

项目会自动生成 `metadata/defect_labels_template.csv`。在它被补齐并另存为 `metadata/defect_labels.csv` 前：

- 禁止使用 full inversion 表述
- 禁止进入监督反演训练
- 默认只运行 weakly supervised / proxy 模式

## 风险提示

- 当前真实物理缺陷单位数量远小于原始点数和窗口数，任何高精度都必须结合严格 split 解读
- synthetic 结果只能验证方法闭环，不能替代真实数据结论
- 当前 UQ 指标链路虽然完整，但校准质量仍需系统优化
- Stage C 的 proxy 表征结果只能写成 proxy / defect-state modeling，不能直接写成几何反演

## 后续建议

如果要把这套系统进一步推进到“更接近论文成品”的级别，优先级建议如下：

1. 为真实数据补齐可信的 `defect_labels.csv`
2. 对 Stage B 和 Stage C 做正式长训练与可视化对比
3. 在 Stage D 上补充 deterministic PINN 与 UQ-PINN 的 LOPO 对比实验
4. 增加 per-target 可视化、误差分布图和 calibration 图
5. 做更系统的 ablation：是否使用 physics term、是否使用 discrepancy、是否使用 target normalization
