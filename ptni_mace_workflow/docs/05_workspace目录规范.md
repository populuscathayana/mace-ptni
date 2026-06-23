# 05 workspace 目录规范

`mace_workspace/` 是新工作流的唯一默认运行根目录。它的目标是让训练、评估和外推任务互不污染。

## 目录结构

```text
mace_workspace/
  datasets/
    ptni_split/
    NP_benchmark_package/
    manifests/
  inputs/
    pt111/
    strained_neb/
    np_structures/
    mcmd/
  models/
    ft_best_loss/
    scratch_best_loss/
    ft_np_baseline/
  runs/
    training/<run_name>/
    evaluation/<model_tag>/<dataset>/
    benchmarks/<benchmark_name>/<model_tag>/
    mcmd/<run_name>/
  reports/
    docs_site/
  tmp/
```

## 数据

`datasets/` 只放规范数据，不放临时测试输出。

| 子目录 | 内容 |
| --- | --- |
| `ptni_split/` | MACE 训练用 train/valid/test extxyz |
| `NP_benchmark_package/` | NP NEB 外推验证包 |
| `manifests/` | 数据迁移和校验 manifest |

## 输入

`inputs/` 放 benchmark 的静态输入。

| 子目录 | 内容 |
| --- | --- |
| `pt111/` | Pt111 hex grid POSCAR 和 `hex_point_origin` |
| `strained_neb/` | `POSCAR-is`、`POSCAR-ts`、`POSCAR-fs` |
| `np_structures/` | 手动 NP 结构优化测试输入 |
| `mcmd/` | vacancy-mediated MCMD 原型输入结构 |

## 模型

每个模型标签一个目录：

```text
mace_workspace/models/<model_tag>/
  model.model
  model_manifest.json
```

建议标签：

| 标签 | 含义 |
| --- | --- |
| `ft_best_loss` | 不含 NP 训练集的 fine-tune best-loss 模型 |
| `scratch_best_loss` | 不含 NP 训练集的 scratch best-loss 模型 |
| `ft_np_baseline` | 含 NP 训练集的旧 fine-tune 基准模型 |

## 运行结果

| 运行类型 | 目录 |
| --- | --- |
| 训练 | `runs/training/<run_name>/` |
| 误差检验 | `runs/evaluation/<model_tag>/<dataset>/` |
| 外推验证 | `runs/benchmarks/<benchmark_name>/<model_tag>/` |
| 动力学雏形 | `runs/mcmd/<run_name>/` |

每个入口脚本至少写一个 manifest：

- 训练：`run_manifest.json`
- 评估：`run_manifest.csv`
- benchmark：`benchmark_suite_manifest.csv`
- MCMD：`run_manifest.json`
- 数据迁移：`datasets/manifests/migration_manifest.csv`

## 禁止的新行为

新流程不要再默认写入：

```text
logs/
checkpoints/
results/
work/model_benchmark_best_loss/
work/np_neb_singlepoint_benchmark/
work/np_neb_relax_neb_two_models/
```

这些目录保留为历史结果，不作为新工作流默认输出。
