# 文件夹整理与整合说明（2026-04-08）

## 整合目标

将仓库按职责划分为 `src/`、`docs/`、`outputs/` 三层结构，减少根目录噪音，统一代码入口与数据出口路径。

## 新目录结构

```text
toric_code/
├── AGENTS.md
├── CLAUDE.md
├── src/
│   ├── core/
│   │   ├── CMI_calculation.py
│   │   ├── bMPS_contraction.py
│   │   ├── H_calculation.py
│   │   └── Syndrome_pure_error.py
│   └── scripts/
│       ├── plot_CMI_vs_p.py
│       ├── plot_CMI_bMPS.py
│       └── plot_CMI_vs_r_three_p_fast2h.py
├── docs/
│   ├── PROJECT.md
│   ├── PROBLEM.md
│   ├── PLAN.md
│   ├── PHYSICS.md
│   ├── paper.md
│   └── FOLDER_INTEGRATION.md
├── outputs/
│   ├── *.csv
│   └── *.png
└── archive/
    └── 2026-04-08_cleanup/
```

## 保留策略（需要）

### 主代码
- `src/core/`：核心计算与模型
- `src/scripts/`：可执行任务脚本

### 主文档
- `docs/`：项目说明、问题、计划与理论背景

### 主结果
- `outputs/`：当前有效结果数据与图像

## 归档策略（不删除，仅下沉）

- 历史脚本、旧结果、中间数据统一放入：
  - `archive/2026-04-08_cleanup/scripts/`
  - `archive/2026-04-08_cleanup/results/`
  - `archive/2026-04-08_cleanup/data/`
  - `archive/2026-04-08_cleanup/cache/`

## 运行方式（新）

从仓库根目录运行：

```bash
python -m src.scripts.plot_CMI_vs_p --help
python -m src.scripts.plot_CMI_bMPS
python -m src.scripts.plot_CMI_vs_r_three_p_fast2h
```

说明：
- 脚本默认把结果写入 `outputs/`。
- `src/` 已配置为包结构（含 `__init__.py`），支持模块化导入。

## 去重原则

- 同类任务只保留一个“当前主入口”脚本。
- 历史版本不删除、仅归档，确保可追溯。
- 根目录仅保留配置与一级目录入口。

