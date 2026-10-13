# RBMF-CS 与 MC-EIR 科研项目

## GitHub 仓库内容

仓库包含源码、配置、脚本、测试及项目文档。`outputs/` 中的本地数据与实验输出（约 16 GB）、`tmp/` 临时文件和 `.venv/` 虚拟环境不上传；本文档中的历史结果路径指向本地工作区。运行依赖真实数据的实验前，需要另行准备对应输入文件；源码仓库本身不包含这些数据。

## 可审计比较实验（2026-10-12 完成）

比较框架、实验协议、历史诊断、重构/部署基线、消融、独立污染与保存结果重绘已实现。2013 验证 LF/海冰已补齐，通过60项输入核验；参数搜索现使用2013全年，禁止以2014调参。2014已用于开发，后续仍只能作探索性评估。当前不支持“RBMF-CS或MC-EIR普遍优于基线”的结论。

- 新数据只读合并：`outputs/comparison_inputs_2013_v1`；配置：`configs/comparison_2013_v1.json`；补充协议：`docs/comparison_protocol_2013_amendment.md`。
- 验证搜索：`outputs/comparison_validation_v1`，150个重构候选与27个部署候选。进度以其 `manifest.json` 为准；只有 `selection.json` 和独立核验通过才能使用选定参数。
- 验证搜索已通过35,953项独立检查；选定参数pilot和重复运行通过77,671项检查、44,480数组对照。正式1200案例全部完成：199,200份保存预测重算、1,558,811项检查通过，无错误或警告。
- 最终报告：[comparison_formal_report.md](docs/comparison_formal_report.md)；正式结果：`outputs/comparison_formal_v1`；CSV与九类PDF/SVG/300dpi PNG：`outputs/comparison_formal_v1/figures`。验收记录：`acceptance.json`、`artifact_verification.json`、`visual_review.json`。
- 主要结果：clean、12节点时 RBMF RMSE 0.528708℃，POD-LASSO 0.439841℃、Gaussian-SBL 0.451626℃、Ridge 0.474100℃。MC-EIR 0.528791℃，未观察到其误差优势；潜在场区间明显欠覆盖。结果属于IAP参考格点模拟采样、2014探索性评估，不是独立野外确认性验证。
- 持久执行状态：`outputs/comparison_completion_v1/status.json`，阶段日志同目录。已完成案例无需再运行；重绘只读保存结果。

- 协议：`docs/comparison_protocol.md`；历史未调参pilot报告：`docs/comparison_report.md`，原样保留。
- 历史完整pilot：`outputs/comparison_pilot_v2`；选定参数完整预算pilot：`outputs/comparison_selected_pilot_v1`，固定种子对照：`outputs/comparison_selected_repeat_v1`。
- 历史诊断：`outputs/comparison_diagnosis_v2/diagnosis.json`；原始`outputs/polar_deployment`保留。
- 重构：LF-only、Ridge、LASSO、Gaussian-SBL-VB、确定性MF残差修正、RBMF及去鲁棒/去MF。
- 部署：Random、Maximin、约束POD-QR、负指示器熵权POD-QR适配、条件熵、后验方差、全场方差下降、MC-EIR及四个评分消融；无约束代价实验另列。
- 关键代码：`src/rbmf_cs/comparison.py`、`comparison_statistics.py`。去MF真正删除兼容因子；去鲁棒固定局部精度1且删除其Gamma项。默认原模型行为不变。

```powershell
.venv\Scripts\python -m pip install -e ".[comparison]"
$env:PYTHONPATH='src'
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python scripts/plot_comparison.py outputs/comparison_formal_v1 --output outputs/figures_new
```

输出目录须不存在。`preflight`在缺数据时退出码2是有意阻断；不自动生成或下载替代真实数据。绘图只读结果，输出CSV/SVG/PDF/300dpi PNG。每次实验保存输入副本/hash、源码zip、配置、精确环境、预生成扰动、逐步预测与诊断；新运行的worker从归档源码执行，工作区后续修改不会影响正在运行的任务。

补齐验证数据后的执行顺序为 `tune_comparison_batch.py` → `verify_comparison_tuning.py` → 带 `--selection` 和 `--full-pilot-budgets` 的 pilot → 验收与重复验证 → 正式批量。正式入口必须同时提供 `--selection .../selection.json` 和对应参数的 `--pilot-verification .../verification.json`。详细参数空间、校准边界与失败政策在上述协议中；历史未调参 pilot 报告按原样保留。

根据用户提供的英文论文补全稿实现阶段1–3；现已完成真实格点数据对齐、验证集调参和上述探索性比较。公式映射、训练专用POD、四类硬约束及逐点观测后的完整重拟合闭环均保留。

## 安装

Python 3.10+，依赖 NumPy、SciPy；测试使用标准库 unittest。

```powershell
cd D:\project\RBMF-CS
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install --no-deps -e .
```

已有依赖可直接运行下面脚本，脚本自行加载项目 src。独立 Python 程序导入本包时应 editable install 或设置 PYTHONPATH。当前记录环境为 Python 3.13.7 / NumPy 2.4.4 / SciPy 1.17.1；精确依赖见 requirements-tested.txt。

## 测试与阶段 3 闭环 demo

```powershell
python scripts/run_tests.py
python scripts/run_closed_loop_demo.py --config configs/closed_loop_demo.json --output outputs/closed_loop_demo
python scripts/validate_closed_loop_run.py outputs/closed_loop_demo
python scripts/check_closed_loop_reproducibility.py
```

使用虚拟环境时，将 python 替换为 `.venv\Scripts\python`。

闭环每轮执行：当前观测 → 完整 RBMF-CS → 筛选整个 prospective deployment 的硬约束 → feasible 集内计算并归一化 MC-EIR → 选择一个节点 → 调用观测器取得 y → 完整 VB warm-start refit → 下一轮重新评分。

默认 synthetic demo 从 `[0,1,12,13]` connected seed 出发，节点 0 为 sensing anchor，固定非感知 gateway 位于 `[-5,0]` km。全部数据、推断、工程预算、成本和评分参数见 `configs/closed_loop_demo.json`。observation callback 只在 seed 观测及实际选择时访问 HF；候选规划无未来 y 或 residual 输入。

当前是 **synthetic 软件验证**，场值为 arbitrary units，不是实测极地数据实验，不提供基线优越性结论。

## 输出内容

| 文件/目录 | 内容 |
|---|---|
| `outputs/test_results.json` | 全部阶段测试的实际通过/失败状态 |
| `outputs/closed_loop_demo/run.json` | 版本、配置/源码 hash、状态、最终节点和可行性、posterior 文件清单 |
| `outputs/closed_loop_demo/config.json` | 完整运行配置副本 |
| `outputs/closed_loop_demo/deployment.json` | 每次候选筛选原因、raw/normalized scores、归一化尺度、选择与停止、score/acquire/refit 事件 |
| `outputs/closed_loop_demo/trajectory.csv` | 新增节点、冰况、EIR、能耗、连通性、不确定性和推断耗时的表格 |
| `outputs/closed_loop_demo/fit_*.json` | 每轮全部 VB 因子、ELBO、收敛状态、warm-start 初始因子 |
| `outputs/closed_loop_demo/posterior_*.npz` | 每轮 mean、variance map、μ_alpha、Σ_alpha；model/physical 单位 |
| `outputs/closed_loop_demo/acquisitions.json` | 实际 acquisition 次序与观测；null 表示缺测 |
| `outputs/closed_loop_demo/synthetic_data.npz` | synthetic train/test LF/HF、坐标、冰况 |
| `outputs/closed_loop_demo/training_artifacts.npz` | 训练专用 basis/mean/scale/covariance factor 及已知成本 |
| `outputs/closed_loop_demo/independent_audit.json` | 独立 SVD ΔU、图连通性和 shortest-path 能耗审计结果 |
| `outputs/closed_loop_reproducibility/report.json` | 两个新进程的完整决策、posterior 和数据比对 |

默认输出路径可通过 --output 改变。一次运行的 run.json 文件清单界定有效 posterior；如需保留多次实验，使用不同输出目录。

## 接口与实现

```python
from rbmf_cs import (
    RBMFCS, VBConfig, TrainingCovariance, ConstraintConfig,
    DeploymentConstraints, PlanningConfig, MCEIR, run_closed_loop,
)

model = RBMFCS(fixed_basis, VBConfig())
constraints = DeploymentConstraints(
    coordinates_km, ice_fraction, ConstraintConfig(),
    gateways=gateway_coordinates_km, anchors=anchor_ids,
)
planner = MCEIR(
    TrainingCovariance.fit(training_anomalies),
    base_cost, maintenance_cost, failure_risk, PlanningConfig(),
    historical_reliability=known_reliability,
)
result = run_closed_loop(
    model, lf_anomaly, seed_ids, seed_y_anomaly,
    constraints, planner, acquire_observation,
)
print(result['stop_reason'])
```

`acquire_observation(node_id)` 返回同一目标快照的观测。`result['snapshots']` 保存每轮 posterior 和诊断；最终模型仍可调用 predict、posterior_variance、covariance_product。可选 `refresh_environment(round, selected, constraints)` 在每轮前更新冰况/运营参数；已有部署违规时停止扩展。

- `src/rbmf_cs/model.py`：完整六组 VB、Gaussian compatibility factor、ELBO、warm-start。
- `src/rbmf_cs/basis.py`：只在训练数据拟合 POD、共用 anomaly normalization 和 covariance factor。
- `src/rbmf_cs/planning.py`：条件结构熵、非正交基底 ΔU、Red、归一化、soft costs、EIR 与最大值/阈值选择。
- `src/rbmf_cs/constraints.py`：冰况、节点预算、任务能耗和 augmented graph 连通性。
- `src/rbmf_cs/deployment.py`：实际单点 acquisition → 全因子 refit → rescore 的 Algorithm 2。

Γ/λ/τ/τ_delta 均为 Gamma shape–rate；两类 discrepancy 后验方差保留。默认不构造完整 N×N 场协方差。Gaussian planning 使用 `r=1/(Eτ·ω)`，与 Student-t marginal noise variance 区别明确；真实观测到来后重拟合全部因子。

## 物理和工程口径

距离模式明确选 cartesian_km 或 latlon_degrees；后者用 haversine 转 km。冰况为 fraction。感知 anchor 占节点数、能耗并检查 ice；固定非感知 gateway 不占 sensor budget，但固定任务能耗计入预算。

能耗是同一 mission_seconds 内**整个网络的任务预算**：sense/process 功率乘时长，加每源 packets_per_sensor 经最小 transmit-energy 路由到 gateway 的逐跳能耗。中继转发计费，新增节点后旧节点路由和负载也重新计算。当前模型不含 receive/idle radio energy，数值均为 synthetic 配置。

每轮 cumulative_mission_energy_j 是当前 whole-network total，不能把各轮 total 再相加。energy_change_j 为相邻差值，可因路由改善而负值；seed total 加全部差值等于最终 total。

硬约束先于评分，任一违规都排除；归一化仅在当轮 feasible 集内以 max+epsilon 进行，零增益仍为零。负 EIR 不被截断。NaN 新观测不补零，物理节点仍留在网络且照常耗能；观测行删除后重新运行 VB。

## 阶段 2 demo 与论文来源

原阶段 2 的预定随机顺序 assimilation demo 仍可运行：

```powershell
python scripts/run_synthetic_demo.py
python scripts/check_reproducibility.py
```

默认 discrepancy prior Gamma(2,2) 及先前弱先验未收敛记录见 `docs/validation_report.md` 和 `outputs/initial_weak_prior_run`。这不是已经在真实数据上优化的先验。

公式对应、变量维度、近似/未指定参数见 `docs/math_to_code.md`；数组/真实数据接入约定见 `docs/data_contract.md`；阶段 3 验证见 `docs/stage3_validation.md`。原论文只读，提取的全文、157 个 native OMML 与来源 hash 在 `docs/source/`。

## 后续范围

2013年全年验证LF温度、配套盐度和海冰已补齐到 `D:\Dataset\RBMF-CS_polar_validation_2013_20261011`，并以原冻结训练基准及共同域对齐，见 `docs/polar_validation_2013_report.md`。下载/审计工具为 `scripts/prepare_polar_validation.py`（默认完整流程，`--stage verify`复核）。后续参数选择只能使用训练/2013验证，2014禁止调参；此前已查看的2014部署结果应标为探索性评价，另留未查看时段作严格确认性测试。

真实极地场的首轮部署数值实验已完成：配置 `configs/polar_deployment.json`，运行 `scripts/run_polar_deployment.py`，复核 `scripts/check_polar_deployment.py`。VS Code中可按Ctrl+Shift+B运行对应任务；地图、曲线和指标保存于 `outputs/polar_deployment`。实验使用真实场模拟格点传感器，硬件参数为情景假设；三种选点方法当前未显示明确优势，1月重构弱于仅ORAS5，详见 `docs/polar_deployment_report.md`。

50 m真实极地样本的NetCDF读取、物理单位转换、训练期异常基准和空间对齐已由 `scripts/align_polar_sample.py` 实现，结果存于 `D:\Dataset\RBMF-CS_polar_sample\aligned_50m`。安装额外依赖可用 `pip install -e ".[real-data]"`；重跑对齐后执行 `python scripts/check_polar_alignment.py` 核对持久化结果。训练期为2000–2012，验证期2013，2014全年排除在POD及预处理拟合之外。详细说明见 `docs/real_data_alignment_report.md`。

通用产品适配、POD/BCS/SBL/LASSO/OMP/deterministic MF 重构基线、Random/QR/entropy-only/uncertainty 部署基线，以及正式时空 holdout、消融、鲁棒性、敏感性和校准实验仍属于阶段 4。真实LF/HF精度关系与观测同化独立性尚需验证；不得由test HF生成LF。

当前 posterior variance 是 fixed-basis mean-field latent field uncertainty，未包含基底学习、截断和未来测量噪声；完整 VB refit 后不保证不确定性或重构误差逐步下降。
