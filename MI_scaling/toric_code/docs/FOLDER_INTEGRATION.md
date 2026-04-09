# 文件夹整理与整合说明（2026-04-09）

## 整合目标

将仓库按职责划分为 `src/`、`docs/`、`outputs/` 三层结构，降低根目录噪音；统一文档命名，并清理重复缓存文件。

## 当前目录结构

```text
toric_code/
├── AGENTS.md
├── README.md
├── src/
│   ├── core/
│   │   ├── cmi_calculation.py
│   │   ├── bmps_contraction.py
│   │   ├── h_calculation.py
│   │   └── syndrome_pure_error.py
│   └── scripts/
│       ├── plot_cmi_vs_p.py
│       ├── plot_cmi_bmps.py
│       ├── plot_cmi_vs_r_three_p_fast2h.py
│       ├── plot_cmi_vs_r_three_p_chunked.py
│       ├── refine_fast2h_chi32.py
│       ├── cmi_keypoints_chi_scan.py
│       └── cmi_keypoints_multiseed.py
├── docs/
│   ├── PROJECT.md
│   ├── PROBLEM.md
│   ├── PLAN.md
│   ├── PHYSICS.md
│   ├── PAPER.md
│   └── FOLDER_INTEGRATION.md
├── outputs/
│   ├── *.csv
│   └── *.png
└── archive/
    └── 2026-04-08_cleanup/
```

## 本次清理内容

- 合并后删除重复说明文件：`CLAUDE.md`（内容已并入 `AGENTS.md`）。
- 删除全部 Python 缓存：`__pycache__/` 与 `*.pyc`。
- 统一文档命名：`docs/paper.md` → `docs/PAPER.md`。
- 清理空目录：`archive/2026-04-08_cleanup/cache/`。

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

## 命名约定

- 文档使用大写主题名：`PROJECT.md`、`PLAN.md`、`PAPER.md`。
- 核心代码保持现有物理语义命名，不做行为性重构。
- 输出文件保持“任务语义 + 参数标签”风格，避免 `final/new/tmp` 等不透明后缀。

## 运行方式（当前）

从仓库根目录运行：

```bash
python -m src.scripts.plot_cmi_vs_p --help
python -m src.scripts.plot_cmi_bmps
python -m src.scripts.plot_cmi_vs_r_three_p_fast2h
python -m src.scripts.plot_cmi_vs_r_three_p_chunked --help
python -m src.scripts.refine_fast2h_chi32 --help
```

## 去重原则

- 同类任务只保留一个“当前主入口”脚本。
- 历史版本不删除、仅归档，确保可追溯。
- 根目录仅保留项目入口与治理文档。
