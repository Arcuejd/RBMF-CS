# 验证集选择后的 RBMF-CS / MC-EIR 探索性比较

运行状态：complete。完成预注册批量执行不等于获得独立测试证据。2014 年已用于诊断和开发，本轮仍为探索性评估；没有未见的独立配对测试年份。

## 数据、规模与隔离

2000–2012 训练、2013 全年验证、2014 四个已有月份探索性评估。2013 补充 LF/海冰通过独立输入审计；训练工件、HF 和网格保持不变。调参工作进程输入副本不含 2014 真值和 LF。原始数据及历史结果均保留。

种子 [0, 1]；月份 [201401]；预算 [4, 6, 8, 10, 12]；场景 ['clean', 'gaussian', 'student', 'outlier', 'missing', 'lf_bias', 'train_gaussian', 'train_student', 'train_outlier', 'train_missing']。共 20 个案例。状态计数：{'completed': 20}。进程异常记录数 0；非收敛、提前停止另在逐预算结果中保留。

同一案例全方法共享按节点预生成的扰动，观测器只公开已选节点。重构比较使用相同约束 Maximin 节点；部署比较固定 RBMF-CS。固定评价格点不采样，面积权重 cos(latitude)，NRMSE 分母为该域训练异常的加权 RMS。数据准备、参数空间、公式和来源见 comparison_protocol.md 及 2013 补充协议。

高斯标准差 0.02℃；t(3) 按等方差缩放；离群、缺测、LF 偏移和训练污染独立。LF +0.2℃ 是固定方向的敏感性分析，可能抵消原有负偏差。训练污染改变 POD/covariance 输入，保持原气候态和尺度，不能推广为所有预处理均污染。

## 验证选择

| 方法 | 搜索数 | 可用候选 | 选定参数 |
|---|---:|---:|---|
| deterministic_mf | 30 | 30 | `{"rank": 4, "lambda": 0.01, "inference": {}}` |
| gaussian_sbl_vb | 15 | 15 | `{"rank": 4, "lambda": 0.1, "inference": {"a_gamma": 0.001, "b_gamma": 0.001}}` |
| pod_lasso | 30 | 20 | `{"rank": 16, "lambda": 0.1, "inference": {}}` |
| pod_ridge | 30 | 30 | `{"rank": 16, "lambda": 0.01, "inference": {}}` |
| rbmf | 45 | 45 | `{"rank": 32, "lambda": 0.1, "inference": {"a_delta": 2.0, "b_delta": 20.0, "nu": 3.0}}` |
| MC-EIR | 27 | 27 | `{"beta": 0.25, "eta": 0.0, "entropy_reference_variance": 0.001}` |

全部候选记录均保留。验证目标为 12 月 × 3 配对种子 × 3 节点预算的平均面积 RMSE，任何未收敛/失败使候选不可选。消融继承完整方法的验证参数，不利用探索性评估调参。

## 12 节点比较

下表误差仅汇总收敛且实际达到该预算的案例；成功数/计划数必须一起阅读。未达到预算的结果没有并入同预算误差。CSV 另含 MAE、NRMSE、实际节点、能耗、成本、可行率和分项运行时间。

### reconstruction

| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 收敛同预算/计划 | clean 可行率 |
|---|---:|---:|---:|---:|
| lf_only | 0.695049 | 0.695049 | 2/2 | 1.000000 |
| pod_ridge | 0.582464 | 0.581788 | 2/2 | 1.000000 |
| pod_lasso | 0.547070 | 0.546706 | 2/2 | 1.000000 |
| gaussian_sbl_vb | 0.545334 | 0.543708 | 2/2 | 1.000000 |
| deterministic_mf | 0.592733 | 0.592411 | 2/2 | 1.000000 |
| rbmf | 0.748298 | 0.748297 | 2/2 | 1.000000 |
| no_robust | 0.748297 | 0.748299 | 2/2 | 1.000000 |
| no_mf | 0.686099 | 0.678500 | 2/2 | 1.000000 |

### deployment

| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 收敛同预算/计划 | clean 可行率 |
|---|---:|---:|---:|---:|
| random | 0.748406 | 0.748403 | 2/2 | 1.000000 |
| maximin | 0.748305 | 0.748304 | 2/2 | 1.000000 |
| pod_qr | 0.748472 | 0.748470 | 2/2 | 1.000000 |
| entropy_qr | 0.748458 | 0.748460 | 2/2 | 1.000000 |
| conditional_entropy | 0.748444 | 0.748443 | 2/2 | 1.000000 |
| posterior_variance | 0.748412 | 0.748410 | 2/2 | 1.000000 |
| variance_reduction | 0.748412 | 0.748410 | 2/2 | 1.000000 |
| mc_eir | 0.748475 | 0.748473 | 2/2 | 1.000000 |
| no_entropy | 0.748412 | 0.748409 | 2/2 | 1.000000 |
| no_uncertainty | 0.748412 | 0.748411 | 2/2 | 1.000000 |
| no_redundancy | 0.748475 | 0.748473 | 2/2 | 1.000000 |
| no_cost | 0.748413 | 0.748412 | 2/2 | 1.000000 |

## 配对差值与适用边界

差值为表中方法减完整参考方法（重构 RBMF、部署 MC-EIR），负数有利于表中方法。以下是 clean 最大预算；全部场景和预算见 paired_statistics.csv/json。只使用四个月均有可用配对的种子，同时报告分母。

| 分支/方法 | 配对平均差 ℃ | 固定环境种子条件95%区间 | 时间/空间分块95%敏感性区间 | 配对种子/计划 |
|---|---:|---|---|---:|
| reconstruction/deterministic_mf | -0.155566 | [-0.155565667086214, -0.155565667086214] | None | 2/2 |
| reconstruction/gaussian_sbl_vb | -0.202964 | [-0.20296415751711072, -0.20296415751711072] | None | 2/2 |
| reconstruction/lf_only | -0.053249 | [-0.0532492647070546, -0.0532492647070546] | None | 2/2 |
| reconstruction/no_mf | -0.062199 | [-0.06219889511301546, -0.06219889511301546] | None | 2/2 |
| reconstruction/no_robust | -0.000001 | [-1.0249077690982489e-06, -1.0249077690982489e-06] | None | 2/2 |
| reconstruction/pod_lasso | -0.201228 | [-0.2012279942405264, -0.2012279942405264] | None | 2/2 |
| reconstruction/pod_ridge | -0.165834 | [-0.16583430555776446, -0.16583430555776446] | None | 2/2 |
| deployment/conditional_entropy | -0.000031 | [-3.1269533436906194e-05, -3.1269533436906194e-05] | None | 2/2 |
| deployment/entropy_qr | -0.000017 | [-1.6780645447744824e-05, -1.6780645447744824e-05] | None | 2/2 |
| deployment/maximin | -0.000170 | [-0.00017026708701506355, -0.00017026708701506355] | None | 2/2 |
| deployment/no_cost | -0.000062 | [-6.162261267739222e-05, -6.162261267739222e-05] | None | 2/2 |
| deployment/no_entropy | -0.000063 | [-6.336121695105223e-05, -6.336121695105223e-05] | None | 2/2 |
| deployment/no_redundancy | 0.000000 | [0.0, 0.0] | None | 2/2 |
| deployment/no_uncertainty | -0.000063 | [-6.330920551678165e-05, -6.330920551678165e-05] | None | 2/2 |
| deployment/pod_qr | -0.000003 | [-3.2082874491523583e-06, -3.2082874491523583e-06] | None | 2/2 |
| deployment/posterior_variance | -0.000063 | [-6.299781108698443e-05, -6.299781108698443e-05] | None | 2/2 |
| deployment/random | -0.000069 | [-0.00014683158582962097, 7.919121225663517e-06] | None | 2/2 |
| deployment/variance_reduction | -0.000063 | [-6.299781108698443e-05, -6.299781108698443e-05] | None | 2/2 |

在 clean 最大预算的可用配对结果中，平均误差低于 rbmf 的基线为：deterministic_mf, gaussian_sbl_vb, lf_only, pod_lasso, pod_ridge。这是当前固定环境下的描述，不是跨年份统计优越性声明。

在 clean 最大预算的可用配对结果中，平均误差低于 mc_eir 的基线为：conditional_entropy, entropy_qr, maximin, pod_qr, posterior_variance, random, variance_reduction。这是当前固定环境下的描述，不是跨年份统计优越性声明。

种子不是独立环境样本。空间按2°×5°分块、时间按相邻两个观测月份循环分块；只有四个相关月份，区间仅作探索性敏感性分析。独立环境 p 值和 Holm 校正 p 值留空；没有制造多方法显著性。

## 失败、约束代价与消融

| 分支/方法/场景 | 预算 | 收敛同预算/计划 |
|---|---:|---:|
| 无 | — | 全部达到预算并收敛 |

原始状态、错误及未收敛诊断见 metrics.csv、failures.json、运行 ledger 和逐案例记录。POD-QR 残差为零时停止，不将数值舍入误差当作有效 pivot；超过所选 POD 秩的预算可能不可用。熵权 QR 使用论文负指示器信息权重乘残差范数，属于工程约束适配，未声称原论文严格复现。

| 约束代价方法 | clean RMSE ℃ | 可行率（按原约束审计） | 成本 | 能耗 J |
|---|---:|---:|---:|---:|
| pod_qr | 0.748472 | 1.000000 | 13.188000 | 5799.571709 |
| unconstrained_pod_qr | 0.748450 | 0.000000 | 14.244000 | 不可用 |
| mc_eir | 0.748475 | 1.000000 | 12.000000 | 8127.216778 |
| unconstrained_mc_eir | 0.748421 | 0.000000 | 12.000000 | 不可用 |

无约束版本单列，不纳入主比较。去多保真删除兼容因子及相应更新/ELBO 项；去鲁棒似然固定局部精度为1并删除局部 Gamma 项。四个部署消融删除对应评分项，仍保留全部硬约束。消融效果不预设为正。工程成本/能耗参数是场景假设，不是实测设备验证。

## 不确定性

| 方法 | Gaussian 潜在场覆盖 | 潜在场宽度 ℃ | 未来观测覆盖 | 未来观测宽度 ℃ |
|---|---:|---:|---:|---:|
| rbmf | 0.090466 | 0.107259 | 0.954032 | 3.310984 |
| gaussian_sbl_vb | 0.325790 | 0.440162 | 0.894735 | 1.810222 |
| no_robust | 0.090466 | 0.107294 | 0.921045 | 2.598217 |
| no_mf | 0.272687 | 0.415624 | 0.362174 | 0.541316 |

验证选择的冗余惩罚系数 eta=0，完整方法的选点评分已不含冗余惩罚。因此去冗余消融与完整方法评分相同，本轮不能用这一相同结果判断非零冗余惩罚在其他环境中的贡献。

潜在场区间来自系数后验；未来观测区间另积分噪声精度和新局部精度，用独立预生成未来观测评价。覆盖率和宽度必须一起解释。遗漏基底学习、截断、模型差异和 mean-field 交叉协方差，不能凭较宽的未来观测区间声称潜在场校准良好。没有使用探索性真值校准区间。

## 验收、运行时间与复现

保存结果核验：passed=True，25991 项检查，3320 份预测重新计算指标。固定种子重跑证据见选定参数 pilot 的 repeat_verification.json；单元参考测试见测试日志。正式并发数 6，批次墙钟 174.39 秒，案例耗时累计 904.00 秒。预处理、选点、重构时间分开记录；并发和审计 I/O 会影响总墙钟，不据此声称算法固有速度优势。

运行目录：`outputs/comparison_selected_pilot_v1`；验证选择：`outputs/comparison_validation_v1`；图表：`outputs/comparison_selected_pilot_v1/figures`。源码、输入副本、版本、种子、逐步预测/后验/迭代/约束均归档，部署大日志为 `.json.gz`。

重绘不拟合模型，不改数据、不平滑、不强制单调。空间图按最早月份、最小种子、clean、最大计划预算固定选择，使用共同色标。

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python scripts/tune_comparison_batch.py --config configs/comparison_2013_v1.json --output outputs/validation_new --workers 6
.venv\Scripts\python scripts/run_comparison.py --mode pilot --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --full-pilot-budgets --workers 6 --output outputs/selected_pilot_new
.venv\Scripts\python scripts/verify_comparison.py outputs/selected_pilot_new
.venv\Scripts\python scripts/run_comparison.py --mode formal --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --pilot-verification outputs/selected_pilot_new/verification.json --workers 6 --output outputs/formal_new
.venv\Scripts\python scripts/verify_comparison.py outputs/comparison_selected_pilot_v1
.venv\Scripts\python scripts/plot_comparison.py outputs/comparison_selected_pilot_v1 --output outputs/figures_new
.venv\Scripts\python scripts/report_comparison_formal.py outputs/comparison_selected_pilot_v1 --figures outputs/figures_new --tuning outputs/comparison_validation_v1 --output docs/report_new.md
```

输出目录必须不存在。复现固定数值允许绝对误差1e-12，运行时间不要求一致；归档 runtime 提供当次原始源码。历史诊断与未调参 pilot 见 comparison_report.md，不能与本次已验证参数结果混为一组。
