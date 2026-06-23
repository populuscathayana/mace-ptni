# PtNi MACE 重构总览

本目录是新的 PtNi MACE 工作流入口。旧的 `outputs/` 保留为历史备份，新代码、新说明和新网页文档都从 `ptni_mace_workflow/` 与 `mace_workspace/` 开始。

## 总流程

```text
OUTCAR
  -> POTCAR/重复/元素筛选
  -> MACE extxyz
  -> train/valid/test split
  -> fine-tune 或 scratch 训练
  -> best-loss .model 导出
  -> train/valid/test 误差检验
  -> 晶格、NEB、Pt111 PES、距离稳定性、NP 外推 benchmark
  -> vacancy-mediated MCMD 动力学雏形
```

## 四个模块

| 模块 | 目录 | 主要用途 |
| --- | --- | --- |
| 前处理 | `ptni_mace_workflow/preprocess/` | OUTCAR 收集、去重、POTCAR 筛选、extxyz 转换、数据集切分、NP package 准备 |
| 训练 | `ptni_mace_workflow/training/` | MACE fine-tune、scratch、训练监控、best checkpoint 转 `.model` |
| 误差检验 | `ptni_mace_workflow/evaluation/` | 低显存预测、train/valid/test 打分、离群点、parity density 图 |
| 外推验证 | `ptni_mace_workflow/benchmarks/` | Pt/Ni 晶格、应变 NEB、Pt111 势能面、slab 距离稳定性、NP 单点、NP relax+NEB |
| 动力学雏形 | `ptni_mace_workflow/mcmd/` | vacancy-mediated hop、ASE+MACE MD、显式 CI-NEB 能垒和速率加权事件选择 |

## 统一运行根目录

所有新入口默认使用：

```text
mace_workspace/
```

它和旧 `work/` 并列。未来训练、评估和 benchmark 的输出都应写到 `mace_workspace/runs/`，不要再写入仓库根目录下的 `logs/`、`checkpoints/`、`results/`。

## 常用命令

迁移规范数据：

```bash
python ptni_mace_workflow/tools/migrate_canonical_data.py --workspace mace_workspace
```

启动 fine-tune：

```bash
MACE_WORKSPACE=mace_workspace \
SAVE_ALL_CHECKPOINTS=1 WANDB=1 \
WANDB_PROJECT=ptni-mace \
WANDB_NAME=ptni_binary_mace_ft \
bash ptni_mace_workflow/training/train_mace_ptni_ft.sh \
  --dataset ptni_split \
  --run-name ptni_binary_mace_ft
```

启动 scratch：

```bash
MACE_WORKSPACE=mace_workspace \
SAVE_ALL_CHECKPOINTS=1 WANDB=1 \
WANDB_PROJECT=ptni-mace \
WANDB_NAME=ptni_binary_mace_scratch \
bash ptni_mace_workflow/training/train_mace_ptni_scratch.sh \
  --dataset ptni_split \
  --run-name ptni_binary_mace_scratch
```

低显存评估：

```bash
bash ptni_mace_workflow/evaluation/evaluate_splits_lowmem.sh \
  --workspace mace_workspace \
  --model-tag ft_best_loss \
  --dataset ptni_split \
  --device cuda
```

统一 benchmark：

```bash
bash ptni_mace_workflow/benchmarks/run_benchmark_suite.sh \
  --workspace mace_workspace \
  --model-tag ft_best_loss \
  --suite lattice,strained_neb,pt111_pes,distance_scan,np_singlepoint,np_relax_neb \
  --device cuda
```

vacancy-mediated MCMD 初始位点准备：

```bash
python -m ptni_mace_workflow.mcmd.run_vacancy_mcmd \
  --workspace mace_workspace \
  --input mace_workspace/inputs/mcmd/POSCAR \
  --run-name np_vacancy_mcmd_smoke \
  --prepare-sites-only \
  --site-output vasp \
  --site-mode np \
  --site-np-boundary one-shell \
  --overwrite
```

检查 `site_reports/step_0000_with_He.vasp` 后，显式指定 vacancy 运行：

```bash
python -m ptni_mace_workflow.mcmd.run_vacancy_mcmd \
  --workspace mace_workspace \
  --input mace_workspace/inputs/mcmd/POSCAR \
  --model-tag ft_best_loss \
  --run-name np_vacancy_mcmd_smoke \
  --vacancy-site-index 0 \
  --mc-steps 1 \
  --md-steps 0 \
  --neb-images 3 \
  --neb-steps 1 \
  --device cpu \
  --overwrite
```

构建网页说明：

```bash
python ptni_mace_workflow/tools/build_docs_site.py
```

网页输出：

```text
mace_workspace/reports/docs_site/index.html
```

## Windows 和 WSL 路径

Windows 路径示例：

```text
C:\Users\A\Documents\Codex\2026-06-08\vasp-dft-mace-outcar-slabd-ptni\mace_workspace
```

WSL 路径示例：

```text
/mnt/c/Users/A/Documents/Codex/2026-06-08/vasp-dft-mace-outcar-slabd-ptni/mace_workspace
```

在 WSL 中运行 MACE/ASE 时，建议先进入项目目录，再运行新脚本：

```bash
cd /mnt/c/Users/A/Documents/Codex/2026-06-08/vasp-dft-mace-outcar-slabd-ptni
conda activate mace-ptni
```
