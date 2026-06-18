# 06 从旧 outputs 迁移说明

旧的 `outputs/` 目录保留为备份。本轮重构不删除、不移动、不改写旧文档和旧脚本。

## 新旧路径对照

| 旧位置 | 新位置 |
| --- | --- |
| `outputs/scripts/collect_outcars.py` 等前处理脚本 | `ptni_mace_workflow/preprocess/` |
| `outputs/scripts/train_mace_ptni.sh` | `ptni_mace_workflow/training/train_mace_ptni_ft.sh` |
| `outputs/scripts/train_mace_ptni_scratch.sh` | `ptni_mace_workflow/training/train_mace_ptni_scratch.sh` |
| `outputs/scripts/evaluate_mace_extxyz_one_by_one.py` | `ptni_mace_workflow/evaluation/evaluate_mace_extxyz_one_by_one.py` |
| `outputs/scripts/score_mace_predictions_extxyz.py` | `ptni_mace_workflow/evaluation/score_mace_predictions_extxyz.py` |
| `outputs/scripts/run_best_loss_model_benchmark.sh` | `ptni_mace_workflow/benchmarks/run_benchmark_suite.sh` |
| `outputs/scripts/run_np_neb_singlepoint_benchmark.sh` | `ptni_mace_workflow/benchmarks/np_neb/run_np_neb_singlepoint_benchmark.sh` |
| `outputs/scripts/run_np_neb_relax_neb_two_models.sh` | `ptni_mace_workflow/benchmarks/np_neb/run_np_neb_relax_neb_two_models.sh` |
| `outputs/docs_site/` | `mace_workspace/reports/docs_site/` |

## 数据迁移

执行：

```bash
python ptni_mace_workflow/tools/migrate_canonical_data.py --workspace mace_workspace
```

该命令只复制规范数据，不删除旧数据。复制结果记录在：

```text
mace_workspace/datasets/manifests/migration_manifest.csv
```

## 模型迁移

当前迁移工具默认复制三个基准模型：

| 旧模型 | 新标签 |
| --- | --- |
| `checkpoints/ptni_binary_mace_ft_run-123_best_loss.model` | `mace_workspace/models/ft_best_loss/model.model` |
| `checkpoints/ptni_binary_mace_scratch_run-123_best_loss.model` | `mace_workspace/models/scratch_best_loss/model.model` |
| `checkpoints/ptni_binary_mace_ft_best_loss.model` | `mace_workspace/models/ft_np_baseline/model.model` |

以后新的 best-loss 模型应通过以下命令导出：

```bash
bash ptni_mace_workflow/training/export_best_model_from_run.sh \
  --workspace mace_workspace \
  --run-name ptni_binary_mace_ft \
  --model-tag ft_best_loss
```

## 迁移后的工作习惯

- 旧命令仍可作为参考，但新实验优先使用 `ptni_mace_workflow/`。
- 新训练统一写到 `mace_workspace/runs/training/`。
- 新评估统一写到 `mace_workspace/runs/evaluation/`。
- 新外推验证统一写到 `mace_workspace/runs/benchmarks/`。
- 新文档统一写到 `ptni_mace_workflow/docs/`，网页由 `ptni_mace_workflow/tools/build_docs_site.py` 生成。

## 检查清单

- `outputs/` 仍存在。
- `mace_workspace/datasets/ptni_split/train.extxyz` 存在。
- `mace_workspace/models/ft_best_loss/model.model` 存在。
- `mace_workspace/reports/docs_site/index.html` 可生成。
- `ptni_mace_workflow/` 下的新入口脚本通过 `bash -n`。
