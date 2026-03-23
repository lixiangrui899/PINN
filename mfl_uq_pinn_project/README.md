# MFL UQ PINN Project (Week 1)

本仓库实现漏磁检测（MFL）正问题的硬边界 PINN 框架与数据生成预处理的第一周代码任务。  
This repo delivers week‑1 code for a hard‑boundary constrained PINN workflow and synthetic data prep for MFL.

## Repo Layout
```
mfl_uq_pinn_project/
├── requirements.txt
├── README.md
├── src/
│   ├── base_pinn/          # Subtask 1: 1D Poisson hard-boundary PINN validation
│   ├── mfl_forward/        # Subtask 2: Axisymmetric MFL forward PINN
│   └── data_processing/    # Subtask 3: Dipole simulation + COMSOL conversion + preprocessing
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── base_pinn/
│   └── mfl_forward/
└── results/
    ├── base_pinn/
    └── mfl_forward/
```

## Setup
1. 创建虚拟环境（可选） / (Optional) create venv  
2. 安装依赖 / Install deps:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run (Week 1 Scope)
1. 验证子任务1 / Validate Subtask 1 (1D Poisson hard boundary):
   ```bash
   python src/base_pinn/validate_base_pinn.py
   ```
2. 验证子任务2 / Validate Subtask 2 (axisymmetric MFL forward):
   ```bash
   python src/mfl_forward/validate_mfl_forward.py
   ```
3. 生成数据集 / Generate dataset (dipole sims + preprocessing):
   ```bash
   python src/data_processing/generate_dataset.py
   ```

## Week 1 Deliverables & Criteria
- Subtask 1: boundary residual ≤ 1e-6; MSE to analytical ≤ 1e-4; max abs err ≤ 1e-2; auto plots (loss, prediction vs. truth, boundary residual) in `results/base_pinn`; model ckpt in `models/base_pinn`.
- Subtask 2: boundary residual ≤ 1e-5; MSE vs. dipole truth ≤ 5e-3; auto plots (loss, heatmap) in `results/mfl_forward`; model ckpt in `models/mfl_forward`.
- Subtask 3: 200 samples dipole dataset; split 7:2:1 with seed 42; normalized; saved as `data/processed/mfl_standard_defect_dataset.npz`. COMSOL converter provided.

## Notes
- 所有训练脚本均支持断点续训（加载现有模型权重继续训练）。  
- 关键函数与类包含中英文注释，符合 PEP8。  
- 仅完成本周范围，不含反问题与贝叶斯 UQ。
