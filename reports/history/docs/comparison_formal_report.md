# 验证集选择后的 RBMF-CS / MC-EIR 探索性比较

运行状态：complete。完成预注册批量执行不等于获得独立测试证据。2014 年已用于诊断和开发，本轮仍为探索性评估；没有未见的独立配对测试年份。

核心结果：无污染、12节点时 RBMF-CS 面积 RMSE=0.528708℃，优于 LF-only 的 0.706389℃，但没有优于同节点和观测下的全部 POD 基线。MC-EIR RMSE=0.528791℃；各部署方法的误差差异很小，本轮未支持其误差优势。优势、失败和适用边界须按下述配对结果、污染场景及约束共同解释。

## 数据、规模与隔离

2000–2012 训练、2013 全年验证、2014 四个已有月份探索性评估。2013 补充 LF/海冰通过独立输入审计；训练工件、HF 和网格保持不变。调参工作进程输入副本不含 2014 参考场和 LF。原始数据及历史结果均保留。

本实验以 IAP 格点温度作为 HF 参考场、ORAS5 作为 LF 候选，从参考格点模拟传感器采样并加入预先生成的污染；不是独立 Argo/ITP 点观测的野外验证。HF/LF 是建模约定，尚未证明 IAP 在全部目标位置精度高于 ORAS5；产品内部同化相关性也不能由项目时间划分消除。误差与覆盖率均相对于 IAP 参考场解释。

种子 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]；月份 [201401, 201404, 201407, 201410]；预算 [4, 6, 8, 10, 12]；场景 ['clean', 'gaussian', 'student', 'outlier', 'missing', 'lf_bias', 'train_gaussian', 'train_student', 'train_outlier', 'train_missing']。共 1200 个案例。状态计数：{'completed': 1200}。进程异常记录数 0；非收敛、提前停止另在逐预算结果中保留。

同一案例全方法共享按节点预生成的扰动，观测器只公开已选节点。重构比较使用相同约束 Maximin 节点；部署比较固定 RBMF-CS。固定评价格点不采样，面积权重 cos(latitude)，NRMSE 分母为该域训练异常的加权 RMS。数据准备、参数空间、公式和来源见 comparison_protocol.md 及 2013 补充协议。

高斯标准差 0.02℃；t(3) 按等方差缩放；离群、缺测、LF 偏移和训练污染独立。仅考察协议固定污染尺度，没有依据结果增加强度以制造优势。LF +0.2℃ 是固定方向的敏感性分析，可能抵消原有负偏差。训练污染改变 POD/covariance 输入，保持原气候态和尺度，不能推广为所有预处理均污染。部署的方差下降采用冻结超参数的高斯近似，没有对未来完整 VB 重拟合做精确积分。

## 验证选择

| 方法 | 搜索数 | 可用候选 | 选定参数 |
|---|---:|---:|---|
| deterministic_mf | 30 | 30 | `{"rank": 4, "inference": {}, "lambda": 0.01}` |
| gaussian_sbl_vb | 15 | 15 | `{"rank": 4, "inference": {"a_gamma": 0.001, "b_gamma": 0.001}}` |
| pod_lasso | 30 | 20 | `{"rank": 16, "inference": {}, "lambda": 0.1}` |
| pod_ridge | 30 | 30 | `{"rank": 16, "inference": {}, "lambda": 0.01}` |
| rbmf | 45 | 45 | `{"rank": 32, "inference": {"a_delta": 2.0, "b_delta": 20.0, "nu": 3.0}}` |
| MC-EIR | 27 | 27 | `{"beta": 0.25, "eta": 0.0, "entropy_reference_variance": 0.001}` |

全部候选记录均保留。验证目标为 12 月 × 3 配对种子 × 3 节点预算的平均面积 RMSE，任何未收敛/失败使候选不可选。消融继承完整方法的验证参数，不利用探索性评估调参。

## 12 节点比较

下表误差仅汇总收敛且实际达到该预算的案例；成功数/计划数必须一起阅读。未达到预算的结果没有并入同预算误差。CSV 另含 MAE、NRMSE、实际节点、能耗、成本、可行率和分项运行时间。

### reconstruction

| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 收敛同预算/计划 | clean 可行率 |
|---|---:|---:|---:|---:|
| lf_only | 0.706389 | 0.706389 | 120/120 | 1.000000 |
| pod_ridge | 0.474100 | 0.474289 | 120/120 | 1.000000 |
| pod_lasso | 0.439841 | 0.439662 | 120/120 | 1.000000 |
| gaussian_sbl_vb | 0.451626 | 0.451314 | 120/120 | 1.000000 |
| deterministic_mf | 0.628681 | 0.628510 | 120/120 | 1.000000 |
| rbmf | 0.528708 | 0.528710 | 120/120 | 1.000000 |
| no_robust | 0.528710 | 0.528711 | 120/120 | 1.000000 |
| no_mf | 0.538041 | 0.540687 | 120/120 | 1.000000 |

### deployment

| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 收敛同预算/计划 | clean 可行率 |
|---|---:|---:|---:|---:|
| random | 0.528754 | 0.528754 | 120/120 | 1.000000 |
| maximin | 0.528711 | 0.528713 | 120/120 | 1.000000 |
| pod_qr | 0.528830 | 0.528831 | 120/120 | 1.000000 |
| entropy_qr | 0.528869 | 0.528872 | 120/120 | 1.000000 |
| conditional_entropy | 0.528766 | 0.528767 | 120/120 | 1.000000 |
| posterior_variance | 0.528803 | 0.528805 | 120/120 | 1.000000 |
| variance_reduction | 0.528803 | 0.528805 | 120/120 | 1.000000 |
| mc_eir | 0.528791 | 0.528792 | 120/120 | 1.000000 |
| no_entropy | 0.528823 | 0.528824 | 120/120 | 1.000000 |
| no_uncertainty | 0.528813 | 0.528813 | 120/120 | 1.000000 |
| no_redundancy | 0.528791 | 0.528792 | 120/120 | 1.000000 |
| no_cost | 0.528762 | 0.528763 | 120/120 | 1.000000 |

### 固定污染尺度下的比较

以下为 12 节点的面积 RMSE（℃）；训练污染与在线污染独立，没有叠加。完整方法、基线和消融均用保存结果汇总。

| 场景 | RBMF | 去鲁棒 | 去多保真 | POD-LASSO | Gaussian-SBL | MC-EIR |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.528708 | 0.528710 | 0.538041 | 0.439841 | 0.451626 | 0.528791 |
| gaussian | 0.528710 | 0.528711 | 0.540687 | 0.439662 | 0.451314 | 0.528792 |
| student | 0.528710 | 0.528712 | 0.542786 | 0.440061 | 0.451846 | 0.528790 |
| outlier | 0.528743 | 0.528744 | 0.557700 | 0.446223 | 0.455207 | 0.528806 |
| missing | 0.528820 | 0.528820 | 0.556274 | 0.459878 | 0.465629 | 0.528878 |
| lf_bias | 0.528691 | 0.528693 | 0.538041 | 0.439841 | 0.451626 | 0.528773 |
| train_gaussian | 0.528709 | 0.528711 | 0.531735 | 0.439587 | 0.451597 | 0.528800 |
| train_student | 0.528708 | 0.528710 | 0.532891 | 0.439632 | 0.451784 | 0.528797 |
| train_outlier | 0.528728 | 0.528726 | 0.561387 | 0.440445 | 0.451558 | 0.528823 |
| train_missing | 0.528796 | 0.528791 | 0.569941 | 0.453213 | 0.452435 | 0.528875 |

## 配对差值与适用边界

差值为表中方法减完整参考方法（重构 RBMF、部署 MC-EIR），负数有利于表中方法。以下是 clean 最大预算；全部场景和预算见 paired_statistics.csv/json。只使用四个月均有可用配对的种子，同时报告分母。

| 分支/方法 | 配对平均差 ℃ | 固定环境种子条件95%区间 | 时间/空间分块95%敏感性区间 | 配对种子/计划 |
|---|---:|---|---|---:|
| reconstruction/deterministic_mf | 0.099972 | [0.0999724, 0.0999724] | [-0.0485749, 0.241427] | 30/30 |
| reconstruction/gaussian_sbl_vb | -0.077082 | [-0.0770819, -0.0770819] | [-0.163659, -0.0142931] | 30/30 |
| reconstruction/lf_only | 0.177681 | [0.177681, 0.177681] | [0.0445778, 0.307004] | 30/30 |
| reconstruction/no_mf | 0.009333 | [0.00933296, 0.00933296] | [-0.0721165, 0.0893271] | 30/30 |
| reconstruction/no_robust | 0.000002 | [1.76298e-06, 1.76298e-06] | [-8.06413e-05, 7.50987e-05] | 30/30 |
| reconstruction/pod_lasso | -0.088867 | [-0.088867, -0.088867] | [-0.183522, -0.0246249] | 30/30 |
| reconstruction/pod_ridge | -0.054608 | [-0.0546078, -0.0546078] | [-0.156656, 0.0197124] | 30/30 |
| deployment/conditional_entropy | -0.000025 | [-2.51912e-05, -2.51912e-05] | [-0.00021286, 0.000142779] | 30/30 |
| deployment/entropy_qr | 0.000078 | [7.84243e-05, 7.84243e-05] | [-7.02008e-05, 0.00023667] | 30/30 |
| deployment/maximin | -0.000080 | [-7.96359e-05, -7.96359e-05] | [-0.000324896, 0.000174168] | 30/30 |
| deployment/no_cost | -0.000029 | [-2.85683e-05, -2.85683e-05] | [-0.000181675, 0.000103722] | 30/30 |
| deployment/no_entropy | 0.000032 | [3.15747e-05, 3.15747e-05] | [-0.000224099, 0.000278378] | 30/30 |
| deployment/no_redundancy | 0.000000 | [0, 0] | [0, 0] | 30/30 |
| deployment/no_uncertainty | 0.000022 | [2.16643e-05, 2.16643e-05] | [-0.00010975, 0.000175831] | 30/30 |
| deployment/pod_qr | 0.000039 | [3.8878e-05, 3.8878e-05] | [-9.24961e-05, 0.000164587] | 30/30 |
| deployment/posterior_variance | 0.000012 | [1.22784e-05, 1.22784e-05] | [-0.000225939, 0.000244451] | 30/30 |
| deployment/random | -0.000037 | [-5.24187e-05, -2.17383e-05] | [-0.000182228, 0.000104908] | 30/30 |
| deployment/variance_reduction | 0.000012 | [1.22784e-05, 1.22784e-05] | [-0.000225939, 0.000244451] | 30/30 |

在 clean 最大预算的可用配对结果中，平均误差低于 rbmf 的基线为：gaussian_sbl_vb, pod_lasso, pod_ridge。这是当前固定环境下的描述，不是跨年份统计优越性声明。

在 clean 最大预算的可用配对结果中，平均误差低于 mc_eir 的基线为：conditional_entropy, maximin, random。这是当前固定环境下的描述，不是跨年份统计优越性声明。

### 保存后验的机制诊断

按固定规则选取 clean、最小种子、全部月份，Random/Maximin/MC-EIR 最后一步的观测/LF兼容精度迹比为 0.00127644–0.0156208，ARD/LF兼容精度迹比为 2.04684–2.05402，LF缩放 rho 为 -0.000191136–0.00132845；从初始节点到最终节点的预测场 RMS 变化为 0.000644321–0.00172566℃。这些诊断不计入算法计时。

精度矩阵的迹比较仅反映总体尺度，不能解释为每个方向相同的影响比例。兼容项大也不意味着预测紧跟原始 LF：它约束的是 rho 缩放后的 LF；rho 接近零时可与 ARD 共同将异常系数压向零，使物理预测接近训练气候态。应结合保存的两项右端向量、局部权重和后验有效自由度解释，不能只用选点不同或 LF 精度一个因素归因。详细逐步数据与节点 Jaccard 见 saved_diagnosis.json。

种子不是独立环境样本。空间按2°×5°分块、时间按相邻两个观测月份循环分块；只有四个相关月份，区间仅作探索性敏感性分析。独立环境 p 值和 Holm 校正 p 值留空；没有制造多方法显著性。

无污染场景的确定性重构与确定性部署在各种子上相同，因此种子条件区间可以退化为零宽度；这不代表环境不确定性为零。Random 的选点仍随种子变化。

## 失败、约束代价与消融

| 分支/方法/场景 | 预算 | 收敛同预算/计划 |
|---|---:|---:|
| 无 | — | 全部达到预算并收敛 |

原始状态、错误及未收敛诊断见 metrics.csv、failures.json、运行 ledger 和逐案例记录。POD-QR 残差为零时停止，不将数值舍入误差当作有效 pivot；超过所选 POD 秩的预算可能不可用。熵权 QR 使用论文负指示器信息权重乘残差范数，属于工程约束适配，未声称原论文严格复现。

| 约束代价方法 | clean RMSE ℃ | 可行率（按原约束审计） | 成本 | 能耗 J |
|---|---:|---:|---:|---:|
| pod_qr | 0.528830 | 1.000000 | 13.017000 | 6003.954130 |
| unconstrained_pod_qr | 0.528753 | 0.000000 | 13.801500 | 不可用 |
| mc_eir | 0.528791 | 1.000000 | 12.037500 | 7928.960717 |
| unconstrained_mc_eir | 0.528761 | 0.000000 | 12.060000 | 不可用 |

无约束版本单列，不纳入主比较。去多保真删除兼容因子及相应更新/ELBO 项；去鲁棒似然固定局部精度为1并删除局部 Gamma 项。四个部署消融删除对应评分项，仍保留全部硬约束。消融效果不预设为正。工程成本/能耗参数是场景假设，不是实测设备验证。

按原约束不可行时，原网络无法给出声明条件下的可执行任务能耗，保留不可用，不用放宽通信后的能耗代替。

## 不确定性

| 方法 | Gaussian 潜在场覆盖 | 潜在场宽度 ℃ | 未来观测覆盖 | 未来观测宽度 ℃ |
|---|---:|---:|---:|---:|
| rbmf | 0.103879 | 0.107265 | 0.963564 | 2.535018 |
| gaussian_sbl_vb | 0.328443 | 0.360273 | 0.879956 | 1.502995 |
| no_robust | 0.103879 | 0.107326 | 0.917742 | 1.945364 |
| no_mf | 0.288791 | 0.383048 | 0.401061 | 0.507289 |

验证选择的冗余惩罚系数 eta=0，完整方法的选点评分已不含冗余惩罚。因此去冗余消融与完整方法评分相同，本轮不能用这一相同结果判断非零冗余惩罚在其他环境中的贡献。

潜在场区间来自系数后验；未来观测区间另积分噪声精度和新局部精度，用独立预生成未来观测评价。覆盖率和宽度必须一起解释。遗漏基底学习、截断、模型差异和 mean-field 交叉协方差，不能凭较宽的未来观测区间声称潜在场校准良好。没有使用探索性真值校准区间。

## 验收、运行时间与复现

保存结果核验：passed=True，1558811 项检查，199200 份预测重新计算指标。固定种子重跑证据见选定参数 pilot 的 repeat_verification.json；单元参考测试见测试日志。正式并发数 6，批次墙钟 8253.20 秒，案例耗时累计 49453.90 秒。预处理、选点、重构时间分开记录；并发和审计 I/O 会影响总墙钟，不据此声称算法固有速度优势。

计时范围：部署表的 reconstruction_seconds 累计模型内部 VB 更新时间，selection_seconds 包含规划器准备、可行性筛选和评分；重构表的 reconstruction_seconds 是独立求解器外层计时，其观测节点作为共同输入，因此 selection_seconds 为0。不同分支的时间不作直接速度排名。区间抽样、落盘、额外机制诊断和绘图不在这些算法计时中，批次总墙钟另含审计 I/O。

运行目录：`outputs/comparison_formal_v1`；验证选择：`outputs/comparison_validation_v1`；图表：`outputs/comparison_formal_v1/figures`。源码、输入副本、版本、种子、逐步预测/后验/迭代/约束均归档，部署大日志为 `.json.gz`。

重绘不拟合模型，不改数据、不平滑、不强制单调。空间图按最早月份、最小种子、clean、最大计划预算固定选择，使用共同色标。

```powershell
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python scripts/tune_comparison_batch.py --config configs/comparison_2013_v1.json --output outputs/validation_new --workers 6
.venv\Scripts\python scripts/verify_comparison_tuning.py outputs/validation_new
.venv\Scripts\python scripts/run_comparison.py --mode pilot --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --full-pilot-budgets --workers 6 --output outputs/selected_pilot_new
.venv\Scripts\python scripts/verify_comparison.py outputs/selected_pilot_new
.venv\Scripts\python scripts/run_comparison.py --mode pilot --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --full-pilot-budgets --workers 6 --output outputs/selected_repeat_new
.venv\Scripts\python scripts/verify_comparison.py outputs/selected_pilot_new --repeat outputs/selected_repeat_new
.venv\Scripts\python scripts/freeze_comparison_scale.py outputs/selected_pilot_new --output outputs/scale_new.json
.venv\Scripts\python scripts/run_comparison.py --mode formal --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --pilot-verification outputs/selected_pilot_new/repeat_verification.json --workers 6 --output outputs/formal_new
.venv\Scripts\python scripts/verify_comparison.py outputs/formal_new
```

只核验或重绘本次已有结果，可单独使用以下命令：

```powershell
.venv\Scripts\python scripts/verify_comparison.py outputs/comparison_formal_v1
.venv\Scripts\python scripts/plot_comparison.py outputs/comparison_formal_v1 --output outputs/figures_new
.venv\Scripts\python scripts/report_comparison_formal.py outputs/comparison_formal_v1 --figures outputs/figures_new --tuning outputs/comparison_validation_v1 --output docs/report_new.md
```

以上重绘和报告命令可单独执行，无需重跑参数搜索或案例。输出目录必须不存在。复现固定数值允许绝对误差1e-12，运行时间不要求一致；归档 runtime 提供当次原始源码。历史诊断与未调参 pilot 见 comparison_report.md，不能与本次已验证参数结果混为一组。

图表阅读提示：部署曲线与部署鲁棒性热图放大了数值轴/色轴，差异在很小的℃尺度；应阅读刻度和配对区间，不能从视觉分离推断实用显著优势。稳健似然、LF兼容和评分各项的结果仅对应当前验证参数、固定污染强度、区域和月份。当前数据不支持普遍优越性或跨年份泛化结论。

## 最终图表与导出验收

导出与汇总独立核验 passed=True，18370 项检查；包含 132000 条预算指标、1100 行汇总和 900 项配对比较。九类 PDF 已逐张渲染并检查文字、图例、共同色标及节点标记；视觉检查 passed=True。PNG 为300dpi，PDF/SVG及CSV全部保留。验收记录为 artifact_verification.json、visual_review.json 和 acceptance.json。
