# RBMF-CS / MC-EIR 比较：诊断、实现与探索性 pilot

状态：**完成框架、小规模端到端验证和 pilot；正式验证调参与 30-seed 批量未执行。** 缺少真实 2013 验证 LF/海冰，不能以 2014 测试代替。用户已确认仅有 2014 四个月 LF。不存在可供本协议使用的未见配对测试环境。以下数字由保存结果生成，不是正式优越性证据。

## 历史结果诊断

历史三方法节点集合 Jaccard 范围 0.2000–0.2632，说明确实选择了不同位置。4→12 节点预测场 RMS 变化仅 0.00015789–0.00044339 ℃。最终观测精度项 / LF 兼容精度项的矩阵迹比为 0.000112064–0.00123535。这些是模型内各项相对大小的诊断，不能把 trace 比解释成逐方向相同的影响比例。

POD 名义秩 70，训练谱参与率有效秩 10.108，累计90%能量需 23 个模态，保留能量 99.0206%。局部鲁棒权重全程范围 0.2157–1.2498，部分观测被降权，但不是所有权重为零。约束排除事件累计（候选/轮次可重复且原因可重叠）：{'connectivity': 17934, 'energy_unavailable': 17934, 'ice_threshold': 5928}。历史所有方法最终都达到12节点，约束影响候选域但没有使三个节点集合相同。

结论：当前 LF 兼容项主导、稀疏观测响应很弱，是误差接近的直接数值证据；选点集合相同不是原因。不能从该现象推断所有部署策略等效，也不能凭测试诊断选择更有利的先验。本次未改变继承的 RBMF 先验来制造优势。细节（逐步精度、RHS、观测有效自由度、局部权重、后验参与率）见 `outputs/comparison_diagnosis_v2/diagnosis.json`。

## 本次范围与公平性

保持训练2000–2012、验证2013。2014全年保守标为探索性；现有四月已用于开发诊断。pilot：2 配对种子，月份 [201401]，预算 [4, 6]，10 个独立污染场景，共 20 个 case。固定 66 个评价格点、280 个候选格点，评价格点不参与选点；所有方法共享观测实现，重构比较共享 Maximin 节点。

Gaussian 与 t(3) 采用相同方差0.0004℃²（标准差0.02℃）；离群、缺测和 LF +0.2℃ 偏移分开；训练污染四类独立于在线污染。正向 LF 偏移只是预设敏感性方向，可能减小原有负偏差，不应预设为必然恶化。训练污染只改变用于 POD/covariance 的矩阵，原气候态/归一化不变，不能概括为所有预处理均被污染。

所有参数为未调优继承值（正则λ=0.1），图中的差异不能替代验证选择后的公平结论。部署统一四类工程约束；无约束附加试验独立标记，并以原约束审计。POD-QR 和熵权 POD-QR 都是适配版本；熵权方向和乘法已核对原论文式6-1至6-5、算法6-1及第118页。基线与变分 SBL 的公式/来源见 `docs/comparison_protocol.md`。

## 6节点结果（同一固定评价格点、两种子均值）

重构比较：

| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 可行率 |
|---|---:|---:|---:|
| lf_only | 0.69504894 | 0.69504894 | 100.0% |
| pod_ridge | 0.73607396 | 0.73597986 | 100.0% |
| pod_lasso | 0.87105674 | 0.87413513 | 100.0% |
| gaussian_sbl_vb | 0.66654728 | 0.74722148 | 100.0% |
| deterministic_mf | 0.69556533 | 0.69535531 | 100.0% |
| rbmf | 0.74874386 | 0.74874365 | 100.0% |
| no_robust | 0.74874660 | 0.74874675 | 100.0% |
| no_mf | 0.66658618 | 0.74728877 | 100.0% |

部署比较（固定 RBMF-CS）：

| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 可行率 |
|---|---:|---:|---:|
| random | 0.74871366 | 0.74871363 | 100.0% |
| maximin | 0.74875131 | 0.74875112 | 100.0% |
| pod_qr | 0.74871476 | 0.74871460 | 100.0% |
| entropy_qr | 0.74870638 | 0.74870631 | 100.0% |
| conditional_entropy | 0.74872558 | 0.74872520 | 100.0% |
| posterior_variance | 0.74871476 | 0.74871460 | 100.0% |
| variance_reduction | 0.74871476 | 0.74871460 | 100.0% |
| mc_eir | 0.74875997 | 0.74875968 | 100.0% |
| no_entropy | 0.74874839 | 0.74874807 | 100.0% |
| no_uncertainty | 0.74875885 | 0.74875849 | 100.0% |
| no_redundancy | 0.74874006 | 0.74873969 | 100.0% |
| no_cost | 0.74867921 | 0.74867965 | 100.0% |

本次 clean 的 RBMF RMSE 为 0.74874386 ℃，LF-only 为 0.69504894 ℃；RBMF 没有优于 LF-only。MC-EIR 为 0.74875997 ℃、Random 为 0.74871366 ℃、Maximin 为 0.74875131 ℃；未观察到 MC-EIR 的误差优势。这里没有任何跨年份统计优越性声明。去多保真必须删除 compatibility 因子，本实现及独立测试已验证，并非输入零 LF。

## 校准、统计与约束代价

Gaussian场景下 RBMF 的 latent 95% 覆盖为 6.13%，平均区间宽度 0.0766 ℃；future-observation 覆盖为 95.40%，宽度 3.5456 ℃。未来区间很宽不能证明潜在场不确定性校准良好。方差遗漏基底学习、POD截断、模型差异和 mean-field 交叉协方差；没有测试真值校准。

配对差值与 seed 条件 bootstrap CI 见 `paired_statistics.csv/json`。只有一个月份，时间/空间联合块区间和独立环境 p 值均留空；不能把两个噪声种子视为两个环境。实现了用于多月探索性敏感性的两观测月循环块 + 2°×5°空间块方法，并独立测试 Holm 校正函数，但当前不制造缺乏独立环境支持的 p 值。条件 CI 仅描述固定环境下随机扰动/Random变化，两个种子很少，不能外推。

无约束 QR 的 clean 原约束可行率为 0.0%；无约束 MC-EIR 为 0.0%。断连时原网络任务能耗为不可定义（null），不记作零。不同实际节点数不混入同预算误差均值，失败/缺项在 CSV 和失败 ledger 保留。

## 验收、耗时与保留的失败

本次保存 1160 份预测，落盘独立复核 9250 项，passed=True。单元/既有回归测试记录见 `outputs/comparison_tests_release_final.txt`。固定种子跨进程重复验证见 `outputs/comparison_repeat_v1/repeat_verification.json`（772数组及无计时的诊断/决策一致），后续全pilot重复结果另见当前运行的repeat_verification.json。重构方法、planner 的选点接口、共享扰动、约束、岭回归独立增广最小二乘、LASSO独立优化、QR独立SciPy pivot、去MF/去鲁棒均已验证。

pilot wall 202.85 s（包括归档与区间诊断I/O）。算法计时另存 preprocess/selection/reconstruction；不把区间抽样、绘图或验证计入选点/拟合。完整30种子×4月份×10场景、4/6/8/10/12预算的粗估 wall 为 3.38–13.52小时，不含验证网格搜索；只是按1–4倍每case扩展的估算，不能承诺。正式规模尚未冻结/执行，因为验证输入不完整。

首轮 `comparison_smoke_v1` 的无约束断连结果暴露 energy=null 保存异常；已修复，原异常JSON保留。历史 `initial_weak_prior_run` 未收敛结果、旧 `polar_deployment` 全部保留。当前 pilot case进程失败数 0，方法异常数 0；非收敛状态仍单独记录，未删除负面结果。

当前限制：单区域、单深度、一个pilot月份、两随机种子、未完成验证调参、无未见独立测试年；IAP网格传感器模拟不是独立现场观测，产品同化独立性未证实；工程功耗/成本是情景假设，不能声称海上部署验证。验证搜索代码因缺真实输入只测试了阻断路径，其真实2013端到端调参仍待运行。正式多月块统计同样尚无真实正式运行结果。

## 复现与重绘

在 `D:\project\RBMF-CS` 中执行，输出目录必须不存在，旧结果不会覆盖：

```powershell
.venv\Scripts\python scripts/diagnose_comparison.py --output outputs/diagnosis_new
.venv\Scripts\python scripts/run_comparison.py --mode pilot --output outputs/pilot_new
.venv\Scripts\python scripts/verify_comparison.py outputs/pilot_new
.venv\Scripts\python scripts/verify_comparison.py outputs/pilot_new --repeat outputs/comparison_pilot_v2
.venv\Scripts\python scripts/plot_comparison.py outputs/comparison_pilot_v2 --output outputs/figures_new
```

补充真实2013 LF/海冰后，写到新数据/配置目录，不覆盖旧原始数据。新 aligned_fields.npz 需保留原全部数组，增加与12个月2013日期对齐的 validation_lf_model 和 validation_ice_fraction，物理量转换/网格/训练尺度沿用既有合同，并重新审计来源与对齐。然后：

```powershell
.venv\Scripts\python scripts/tune_comparison.py --config configs/new_real_validation.json --output outputs/tuning_new
.venv\Scripts\python scripts/run_comparison.py --mode formal --config configs/new_real_validation.json --selection outputs/tuning_new/selection.json --pilot-verification outputs/pilot_new/verification.json --output outputs/formal_new
```

正式运行仍标记为2014探索性评估；不能因调参只使用验证年便恢复2014的未见测试身份。每次运行保留输入副本、输入hash、配置、源码zip、精确软件版本、训练矩阵、扰动、观测、逐轮预测/后验/诊断/可行性。源码快照可在新的目录中解包后用该脚本和保存config重跑；程序不要求 Git 仓库（当前工作区无.git）。

## 图表

所有图只读结果生成，CSV、SVG、PDF、300dpi PNG 位于 `outputs/comparison_pilot_v2/figures`。空间展示固定为最早月份、最小种子、clean、最大计划预算，共同色标。未平滑曲线、未强制误差随节点单调下降、未筛选有利seed。
