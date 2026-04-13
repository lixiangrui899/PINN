# UQ-PINN + 去噪 + MFL 审稿人约束版项目

这个项目不是“先堆模型再补解释”的 demo，而是按审稿人约束把工程拆成两个明确模式：

- 无标签过渡版：只允许做数据审计、去噪、paired axial/radial consistency、physics-constrained surrogate 和 uncertainty-aware proxy 输出。
- 有标签正式版：只有当 `metadata/defect_labels.csv` 存在且字段齐全时，才允许进入监督反演训练。

## 核心约束

- 不能把重复测量当独立样本。
- 不能把滑窗当样本数。
- 不能在没有几何标签时宣称完成缺陷参数反演。
- physics-informed 模块必须显式拆成 data loss、physics loss 和 discrepancy term。
- UQ 模块必须保留 NLL、PICP、MPIW 和 calibration curve 接口。

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
   └─ training/
```

## 如何运行

在项目根目录下执行：

```bash
python -m pip install -e .
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
- `metadata/defect_labels_template.csv`
- `manifests/leave_one_position_out.csv`
- `manifests/grouped_kfold_by_defect_unit_id.csv`
- `manifests/strict_paired_split.csv`

## 独立样本定义

- `raw file`：一次原始 CSV 采集文件
- `repeated measurement`：同一物理位置的不同 run
- `sliding window`：从单个长序列切出的局部片段，只能作为训练片段，不能当独立样本
- `physical defect unit`：当前默认按 `position_id` 视作物理独立缺陷单位

所有 split 都必须在 `physical defect unit` 或严格配对组的层级上做，而不是在窗口层级随机打散。

## 标签接口

项目会自动生成 `metadata/defect_labels_template.csv`。在该文件被补齐并另存为 `metadata/defect_labels.csv` 前：

- 禁止使用 full inversion 表述
- 禁止进入监督反演训练
- 默认只运行 weakly supervised / proxy 模式

## 当前默认模型边界

- Stage B：1D ResUNet 去噪，支持 repeated-run averaging pseudo target
- Stage C：black-box / deterministic PINN / UQ-PINN 三种代理模式
- Stage D：监督反演接口，只有标签齐全时才可进入
- Stage D 现在支持 target normalization、per-target metrics 和 leave-one-position-out 批量评估

## 合成信号生成

项目支持基于真实 `划痕缺陷` 数据的统计特征生成 synthetic MFL CSV：

- 输出文件仍然是 `Index + AIN 1~AIN 8`
- 保留 `位置/轴向/径向/run_id/timestamp/acquisition_tag` 结构
- 自动生成 `defect_labels.csv`、`synthetic_manifest.csv` 和可直接运行的新 config

示例：

```bash
python -m uq_pinn_mfl.cli --config configs/default.yaml simulate --mode benchmark --dataset-name synthetic_pinn_demo
python -m uq_pinn_mfl.cli --config configs/generated/synthetic_pinn_demo.yaml audit
python -m uq_pinn_mfl.cli --config configs/generated/synthetic_pinn_demo.yaml stage-d --variant blackbox
python -m uq_pinn_mfl.cli --config configs/generated/synthetic_pinn_demo.yaml stage-d --variant uq_pinn --lopo
```

## 风险提示

- 当前物理缺陷单位数量很可能远小于原始点数和窗口数，任何高精度都必须结合严格 split 解读。
- 位置分类、对比嵌入、配对一致性学习只能被表述为 proxy / defect-state modeling，不能包装成几何反演。
