# 08 MCMD 动力学雏形

本模块是 vacancy-mediated MCMD 的早期原型。它不是完整生产级 KMC/MD 框架，而是把以下几件事串起来，方便检查 PtNi NP 或 slab 中空位介导邻位迁移的物理合理性：

1. 用 `reconstruct_close_packed_sites.py` 重构潜在 close-packed 空位位点。
2. 显式指定一个初始 vacancy site。
3. 在 vacancy 第一近邻壳层中生成 atom-to-vacancy hop 候选。
4. 对每个候选 hop 执行 ASE CI-NEB，得到显式过渡态能垒。
5. 按 `nu * exp(-Ea/kBT)` 做速率加权事件选择。
6. 可选在每个 MC 步之间插入 ASE+MACE MD。

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

CPU smoke test：

```bash
python -m ptni_mace_workflow.mcmd.run_vacancy_mcmd \
  --workspace mace_workspace \
  --input mace_workspace/inputs/mcmd/POSCAR \
  --model-tag ft_best_loss \
  --run-name np_vacancy_mcmd_smoke \
  --auto-vacancy highest-score \
  --mc-steps 1 \
  --md-steps 0 \
  --neb-images 3 \
  --neb-steps 1 \
  --device cpu \
  --overwrite
```

## 空位定义

推荐显式指定 vacancy：

| 参数 | 含义 |
| --- | --- |
| `--vacancy-site-index 0` | 使用重构 site list 中第 0 个位点。注意这是 MCMD CLI 的 zero-based index。 |
| `--vacancy-cartesian X Y Z` | 直接给出空位笛卡尔坐标，单位 A。 |
| `--auto-vacancy highest-score` | 仅用于 smoke test，自动选 close-packed reconstruction 排序第一的位点。 |

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
| `--neb-steps` | `100` | FIRE 最大步数 |
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
| `--md-steps` | `0` | 每个 MC step 前执行多少步 MD |
| `--md-ensemble` | `langevin` | `langevin` 或 `nve` |
| `--md-timestep-fs` | `1.0` | MD 步长 |
| `--md-friction-per-fs` | `0.01` | Langevin friction |

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

## 注意事项

- 这是 MCMD/KMC-like 原型，不声明物理时间已经达到严格 KMC 级别。
- 初始 vacancy 推荐手动指定，避免把表面空隙或边界 artifact 当成迁移空位。
- 默认不优化每个 hop 的端点，是为了保留 MD 热涨落；若要做 benchmark 风格势垒，可用 `--endpoint-relax-mode full`。
- CI-NEB 的 climbing image 不是“不受力”，而是使用投影后的 NEB 力：沿路径方向爬升，垂直路径方向松弛。

