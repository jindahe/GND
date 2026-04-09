# toric_code

二维环面码（toric code）在去相位噪声下的 CMI/MI 数值研究项目。  
当前仓库已按职责重构为 `src/`、`docs/`、`outputs/` 三层结构。

## 目录结构

- `src/core/`：核心计算模块（CMI、bMPS、H(m) 等）
- `src/scripts/`：可执行脚本入口
- `docs/`：项目文档（问题、计划、物理背景）
- `outputs/`：结果数据与图像输出
- `archive/`：历史版本和中间结果归档

## 快速开始

在仓库根目录执行：

```bash
python -m src.scripts.plot_cmi_vs_p --help
```

## 常用命令

### 1) 运行 CMI vs p 扫描

```bash
python -m src.scripts.plot_cmi_vs_p \
  --r-values 1,2,3,4,5,6 \
  --method auto \
  --num-samples 500 \
  --no-show
```

默认输出到 `outputs/CMI_vs_p_results.csv` 和 `outputs/CMI_vs_p.png`。

### 2) 运行大 r 的 bMPS 扫描

```bash
python -m src.scripts.plot_cmi_bmps
```

默认输出到 `outputs/CMI_bMPS_results.csv` 和 `outputs/CMI_bMPS_vs_p.png`。

### 3) 运行 fast2h 版本（r=1..6, p=0.05/0.11/0.15）

```bash
python -m src.scripts.plot_cmi_vs_r_three_p_fast2h
```

默认输出到：
- `outputs/CMI_vs_r_three_p_fast2h_raw.csv`
- `outputs/CMI_vs_r_three_p_fast2h_summary.csv`
- `outputs/CMI_vs_r_p005_011_015_fast2h.png`

## 文档入口

- 研究背景与实现说明：`docs/PROJECT.md`
- 当前问题与瓶颈：`docs/PROBLEM.md`
- 开发计划与并行方案：`docs/PLAN.md`
- 物理公式与解释：`docs/PHYSICS.md`
- 文件整合说明：`docs/FOLDER_INTEGRATION.md`

