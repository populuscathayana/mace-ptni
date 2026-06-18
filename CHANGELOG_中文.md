# 更新日志

本项目采用 SemVer 版本号。每次代码或文档工作流发生实质修改时，同步更新 `VERSION` 和本文件。

## v0.1.0 - 2026-06-18

### 新增

- 建立 `ptni_mace_workflow/` 模块化工作流目录，区分前处理、训练、误差检验和外推任务验证。
- 建立 `mace_workspace/` 作为统一运行根目录，隔离数据集、模型、训练、评估和 benchmark 输出。
- 新增 workspace-aware 入口脚本：
  - `ptni_mace_workflow/training/train_mace_ptni_ft.sh`
  - `ptni_mace_workflow/training/train_mace_ptni_scratch.sh`
  - `ptni_mace_workflow/training/export_best_model_from_run.sh`
  - `ptni_mace_workflow/evaluation/evaluate_splits_lowmem.sh`
  - `ptni_mace_workflow/benchmarks/run_benchmark_suite.sh`
- 新增中文模块化说明文档和静态网页生成脚本。
- 新增 `.gitignore`，默认排除大数据、模型、checkpoint、训练结果、W&B 日志和网页构建产物。
- 新增 GitHub Pages Actions workflow，push 后自动构建并部署文档网站。

### 约定

- Git 仓库只保存代码、文档、轻量配置和部署工作流。
- `*.extxyz`、`*.model`、`*.pt`、`work/`、`mace_workspace/`、`checkpoints/` 不进入 Git。
- GitHub Pages 只发布文档网站，不发布数据集、模型或运行结果。
