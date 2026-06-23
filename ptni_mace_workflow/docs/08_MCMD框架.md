# 08 MCMD 动力学雏形

本模块是 vacancy-mediated MCMD 的早期原型。它不是完整生产级 KMC/MD 框架，而是把以下几件事串起来，方便检查 PtNi NP 或 slab 中空位介导邻位迁移的物理合理性：

1. 用 `reconstruct_close_packed_sites.py` 重构潜在 close-packed 空位位点。
2. 显式指定一个初始 vacancy site。
3. 在 vacancy 第一近邻壳层中生成 atom-to-vacancy hop 候选。
4. 对每个候选 hop 执行 ASE CI-NEB，得到显式过渡态能垒。
5. 按 `nu * exp(-Ea/kBT)` 做速率加权事件选择。
6. 默认在完成 MC-NEB hop 之后执行 ASE+MACE MD relaxation 段。

## 主入口

```bash
python -m ptni_mace_workflow.mcmd.run_vacancy_mcmd \
  --workspace mace_workspace \
  --input mace_workspace/inputs/mcmd/POSCAR \
  --model-tag ft_best_loss \
  --run-name np_vacancy_mcmd_test \
  --vacancy-site-index 0 \
  --temperature 800 \
  --mc-steps 20 \
  --md-steps 200 \
  --md-timestep-fs 1.0 \
  --neb-images 5 \
  --neb-fmax 0.05 \
  --device cuda
```

初始位点准备：

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

检查 `mace_workspace/runs/mcmd/np_vacancy_mcmd_smoke/site_reports/step_0000_with_He.vasp`，确认 vacancy 编号后再显式运行：

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

## 空位定义

MCMD 使用的是 close-packed 壳层 lattice-site 逻辑，不是只搜索颗粒内部 vacancy。对于 NP，默认边界策略是：

```bash
--site-np-boundary one-shell
```

这会保留壳层上的 close-packed 候选点；真正的内部空位也会包含在这个候选集合中。若只想找 hull 内部空位，才使用 `--site-np-boundary strict-hull`。

推荐显式指定 vacancy：

| 参数 | 含义 |
| --- | --- |
| `--vacancy-site-index 0` | 使用重构 site list 中第 0 个位点。注意这是 MCMD CLI 的 zero-based index。 |
| `--vacancy-cartesian X Y Z` | 直接给出空位笛卡尔坐标，单位 A。 |

`reconstruct_close_packed_sites.py` 的报告中仍保留它原始的一基 `index`，MCMD 输出中对应为 `vacancy_source_index`。

## Hop 事件

默认事件是：

```text
atom_i + vacancy_site -> atom_i moves into vacancy_site
new vacancy = atom_i old position
```

邻位判据：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--hop-shell-low` | `0.70` | hop 距离下限，乘以重构得到的 `d_nn` |
| `--hop-shell-high` | `1.30` | hop 距离上限，乘以 `d_nn` |
| `--max-events-per-step` | 空 | 每个 MC 步最多评估多少个候选事件 |

若要采用“每一步只从邻位候选中随机抽一个试探点”的轻量 rKMC smoke 逻辑：

```bash
--event-order random --max-events-per-step 1
```

默认禁止跨周期边界 hop。若 direct 位移和 minimum-image 位移差异超过 `--pbc-cross-tol`，该事件会被跳过。只有确认插值路径合理时才建议使用：

```bash
--allow-pbc-hop
```

## CI-NEB 和速率

每个候选事件会写入：

```text
mace_workspace/runs/mcmd/<run_name>/neb_cache/<event_id>/
  neb_initial.extxyz
  neb_initial_path.extxyz
  neb_final.extxyz
  energy_profile.csv
  summary.json
```

默认设置：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--neb-images` | `5` | NEB 总 image 数，包含端点 |
| `--neb-steps` | `100` | 每个候选 hop 的 CI-NEB FIRE 最大优化步数，不是 MC step |
| `--neb-fmax` | `0.05` | NEB 收敛力阈值，eV/A |
| `--endpoint-relax-mode` | `none` | 默认不在每个 MC hop 前完全优化端点 |
| `--attempt-frequency` | `1e13` | 速率前因子，s^-1 |

能垒定义：

```text
Ea_forward = max(E_images) - E_initial
Ea_reverse = max(E_images) - E_final
DeltaE = E_final - E_initial
rate = nu * exp(-Ea_forward / kBT)
```

如果要把未收敛 NEB 事件排除出 MC 选择：

```bash
--require-neb-converged
```

## MD 部分

第一版只实现 `ASE + MACE`：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `--md-steps` | `0` | 每个 MC step 对应的 MD relaxation 步数 |
| `--md-position` | `after` | 默认在 MC-NEB hop 执行后做 MD relaxation |
| `--md-ensemble` | `langevin` | `langevin` 或 `nve` |
| `--md-timestep-fs` | `1.0` | MD 步长 |
| `--md-friction-per-fs` | `0.01` | Langevin friction |

默认推荐流程是：

```text
reconstruct close-packed sites
  -> randomly choose one neighboring vacancy-mediated hop
  -> run CI-NEB for this hop
  -> execute the accepted hop
  -> run MD relaxation
```

LAMMPS 后端暂不实现。原因是不同机器上的 LAMMPS-MACE pair style、模型加载方式和单位制需要单独确认；当前原型先保证 ASE+MACE 可检查、可复现。

## 输出

```text
mace_workspace/runs/mcmd/<run_name>/
  run_manifest.json
  mcmd_steps.csv
  events.csv
  md_steps.csv
  trajectory.extxyz
  summary.md
  site_reports/
  neb_cache/
```

关键表格：

| 文件 | 内容 |
| --- | --- |
| `events.csv` | 每个候选 hop 的 atom、vacancy、NEB barrier、rate 和是否被选中 |
| `mcmd_steps.csv` | 每个 MC step 选中的事件、接受后能量、最大力和累计时间 |
| `md_steps.csv` | 可选 MD 段的能量、温度和最大力 |
| `trajectory.extxyz` | 初始结构、MD frame 和 accepted MC state |
| `site_reports/*_with_He.vasp` | 用 He 可视化的重构空位候选，不参与真实计算 |

对于只想看每步结果和可选空位的紧凑输出，推荐：

```bash
--neb-output compact --site-output vasp
```

这样不会写出每个 NEB 事件的完整 extxyz 路径，只保留 `events.csv`、`mcmd_steps.csv`、`trajectory.extxyz` 和每步可选空位的 VASP 可视化文件。

## 注意事项

- 这是 MCMD/KMC-like 原型，不声明物理时间已经达到严格 KMC 级别。
- 初始 vacancy 推荐手动指定，避免把表面空隙或边界 artifact 当成迁移空位。
- 默认不优化每个 hop 的端点，是为了保留 MD 热涨落；若要做 benchmark 风格势垒，可用 `--endpoint-relax-mode full`。
- CI-NEB 的 climbing image 不是“不受力”，而是使用投影后的 NEB 力：沿路径方向爬升，垂直路径方向松弛。
