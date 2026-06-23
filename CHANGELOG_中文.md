# 更新日志

本项目采用 SemVer 版本号。每次代码或文档工作流发生实质修改时，同步更新 `VERSION` 和本文件。

## v0.3.2 - 2026-06-24

### 调整

- MCMD 默认执行顺序改为完成 MC-NEB hop 后再运行 MD relaxation 段，即 `--md-position after`。
- 文档明确 `--neb-steps` 是每个候选 hop 的 CI-NEB FIRE 最大优化步数，不是 MC 步数。

## v0.3.1 - 2026-06-23

### 调整

- MCMD 的 NP site boundary 默认从 `strict-hull` 调整为 `one-shell`，使 vacancy 候选来自 close-packed 壳层而不是只偏向颗粒内部空位。
- 文档补充“每步随机抽一个邻位 hop”“MC 后执行短 MD/relax”“紧凑输出”的推荐参数。

## v0.3.0 - 2026-06-23

### 新增

- 新增 `ptni_mace_workflow/mcmd/` vacancy-mediated MCMD 原型模块。
- MCMD 主入口支持显式 vacancy site、空位邻位 hop 事件生成、ASE+MACE MD、显式 ASE CI-NEB 能垒计算和 `nu * exp(-Ea/kBT)` 速率加权事件选择。
- 新增 NEB event cache、`events.csv`、`mcmd_steps.csv`、`trajectory.extxyz`、`site_reports/` 和 `summary.md` 输出约定。
- `mace_workspace/` 目录规范新增 `inputs/mcmd/` 和 `runs/mcmd/<run_name>/`。

### 文档

- 新增 `08_MCMD框架.md`，说明空位指定、hop 判据、CI-NEB 参数、MD 参数、输出和注意事项。
- 更新总览、外推任务验证说明和网页文档索引。

## v0.2.0 - 2026-06-22

### 新增

- 新增 `distance_scan` benchmark，用于扫描 PtNi slab 顶部唯一可动原子沿第三晶格矢量方向远离表面时的 MACE 单点能稳定性。
- 新脚本 `ptni_slab_mobile_atom_distance_scan_mace.py` 支持自动识别 selective-dynamics 中唯一可动原子、加厚真空、输出绝对位移和相对层间距两类横坐标。
- `run_benchmark_suite.sh` 新增 `--suite distance_scan`，默认读取 `mace_workspace/inputs/PtNi-diffusion/111/POSCAR` 和 `100/POSCAR`。
- distance scan 输出 CSV、Markdown、PNG 和本地 HTML 报告。

### 文档

- 更新外推任务验证说明，加入 PtNi slab 距离稳定性 benchmark 的输入、参数、输出和判断方式。
- 更新网页文档构建元信息和总览文档。

## v0.1.4 - 2026-06-18

### 新增

- `train_mace_ptni_ft.sh` 和 `train_mace_ptni_scratch.sh` 支持显式命令行参数 `--epochs` / `--max-num-epochs`。
- 两个训练入口支持显式早停参数 `--patience` / `--early-stop-patience`，例如 `--patience 10` 表示 validation 10 个 epoch 无改善后早停。

### 文档

- 更新训练说明，加入 fine-tune 和 scratch 的 epoch/early-stop 示例。

## v0.1.3 - 2026-06-18

### 调整

- 精简 Git 与 GitHub 维护教程，只保留日常提交、添加新文件、push 和 Pages 自动更新的基本流程。
- 移除回档相关长说明；需要回档时再单独处理。

## v0.1.2 - 2026-06-18

### 新增

- 扩展 Git 维护教程，增加“新增文件或小型数据如何纳入 Git 上传”的判断标准和命令。
- 增加回档教程，覆盖查看旧版本、恢复单个文件、修改未 push 的最后一次提交、以及对已 push commit 使用 `git revert` 的安全流程。

### 清理

- 删除本地可再生成的 `_site/` 和 Python `__pycache__/` 缓存目录。

## v0.1.1 - 2026-06-18

### 新增

- 新增 `07_Git与GitHub维护教程.md`，面向 Git 新手说明日常修改、暂存、提交、tag、push、GitHub Pages 和认证方式。
- 将 Git 维护教程加入网页文档目录，GitHub Pages 自动部署后可在线查看。
- 新增并维护 `GIT_SUBMIT_COMMANDS.md`，作为 Codex 无法直接 push 时的本地提交/推送指令文件。

### 调整

- `.gitignore` 新增 `outputs/`，旧输出目录作为本地备份，不再干扰 `git status`。

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
