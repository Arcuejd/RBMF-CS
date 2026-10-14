# 可审计比较协议 v1（2026-10-11）

本协议在新比较试运行之前写入，配置和源码快照随运行冻结。目标为辨识优势、失败与适用条件，无单调性或优越性预设。历史目录不覆盖，原始数据只读。正式运行规模须在 pilot 计时后冻结；目前正式运行被验证 LF/冰况缺失阻断。

## 数据与开发暴露

保持 2000–2012 训练、2013 验证、2014 评估。2014-01/04/07/10 已用于实现检查、历史比较及本次诊断，2014 全年保守标为探索性。没有可用于本协议的未见、已配对多保真测试环境。早期原始数据可用性审计查看过 2000–2026 的缺测/元数据，并不等于这些年均做过性能评估；新增年份仍需独立 LF、海冰与来源审计。

当前只有 2014 四个月 LF 和海冰，用户确认无 2013 LF。不得从 HF 造 LF，不得用测试 LF/HF 调参，不得声称完整正式实验已完成。pilot 用继承参数，目的为软件验证和计时，其误差仅属探索性。正式入口需要完整验证输入、已落盘调参记录和 pilot 验证通过。

## 固定域与公平性

使用 atlantic 346 个既有训练有效格点。按稳定 original_grid_flat_ids 模 5 等于 0 固定评价格点，其余为候选；评价格点永不采样。训练 POD 可以包含这些位置的历史场，故本实验是未采样位置重构，不声称空间外推。mask 不依赖在线真值或算法选择。四个初始节点按候选的网关距离、当月已知冰况选定，对全部方法固定。

主比较全部方法用相同候选域、seed、工程约束及观测扰动。缺测节点仍计数量/成本/能耗。最大预算 4/6/8/10/12，逐点轨迹截取；实际数量不足时保存失败，不能当成同预算完成。无约束附加实验仍用相同域和 seed，仅放宽 ice/通信/能耗，另行用原约束审计，不能混入主表。

重构比较固定约束 Maximin 产生的节点和观测；部署比较只用同一 RBMF-CS。各方法是否使用 LF 在方法表明示，不把额外信息隐去。

## 方法公式和来源

令 H=Ψ[J,:]，y 为共同训练尺度下观测，l 为 LF 异常。缺测删除对应行。

| 方法 | 定义与信息 |
|---|---|
| LF-only | x=l，不用 y，不人为添加后验 |
| POD-Ridge | min_a 1/2||Ha-y||²+λ/2||a||²，x=Ψa；训练 HF+已选观测 |
| POD-LASSO | min_a 1/2||Ha-y||²+λ||a||₁；坐标下降、KKT 收敛；不隐藏 p<K 的欠定性 |
| Gaussian-SBL-VB | Gaussian y|a,τ，ARD Gamma 精度，变分 a/γ/τ；无 LF 和 λ_i 因子。采用本项目 VB 版本，不是 Tipping type-II 算法严格复现 |
| deterministic-MF | x=l+Ψ argmin_b [1/2||Hb-(y-l[J])||²+λ/2||b||²]；确定性残差修正，非 co-kriging 严格复现 |
| RBMF-CS | 既有六因子模型，见 math_to_code.md；LF+训练 HF+已选观测 |

Ridge/LASSO 采用上述明确定义的标准正则目标。SBL 来源：[Tipping 2001](https://www.jmlr.org/papers/v1/tipping01a.html)，变分思路：[Bishop & Tipping, Variational Relevance Vector Machines](https://arxiv.org/abs/1301.3838)。POD-QR 来源：[Manohar 等](https://arxiv.org/abs/1701.07569)。所有工程约束适配均标明为改写版本。

部署：Random 使用预生成节点随机优先级；Maximin 最大化到已选点最小球面距离；POD-QR 在 feasible 集内最大化 Ψ 候选行对已选行空间的正交残差范数（强制 seed 的 QR 贪心适配）；条件熵用训练 covariance Schur complement；最大后验方差用 hΣhᵀ；最大预期全场方差下降用 ||ΨΣhᵀ||²/(r+hΣhᵀ)，是冻结当前超参数的 Gaussian 近似，非对未来完整 VB 的精确积分；MC-EIR 沿用既有公式与阈值。

熵权 QR 核对张倩楠博士论文实体页 107–114、118，式 6-1…6-5 和算法 6-1：负指示器 z_ij=(max_i x_ij-x_ij)/(max_i x_ij-min_i x_ij)，p_ij=z_ij/Σ_i z_ij，e_j=-Σ_i p_ij log(p_ij)/log(T)，ω_j=(1-e_j)/Σ_j(1-e_j)。pivot 分数为剩余列范数乘 ω_j，而不是乘 e_j 或除 ω_j。第118页明确实验采用负指示器。常数列定义 e=1、ω=0，全常数域报错。实现为 **entropy-weighted POD-QR constraint adaptation**：统一 POD、全部训练时间样本、强制 seed、四类约束；不是论文完整 RM_QR_W 严格复现（未重现时间筛选、原 SST 域和原实验基底口径）。不沿用“任意非负子模函数乘积仍子模”的证明结论：例如 |S|² 不具边际递减性。

## 搜索、污染与规模

正式搜索只用 2013 年，所有 12 月、验证种子 1001/1002/1003、预算 4/8/12 平均面积 RMSE。POD rank 候选 4/8/16/32/70，共用训练基底前缀；Ridge/LASSO/确定性 MF λ=1e-4/1e-3/1e-2/0.1/1/10；SBL aγ=bγ=0.001/0.01/0.1；RBMF aδ=2，bδ=0.2/2/20，ν=3/4/8；部署 β=0.25/1/4，η=0/0.2/0.5，熵 reference=0.001/0.01/0.1。各方法报告完整搜索次数及耗时，不保证相同次数却遗漏重要参数。非收敛/失败候选目标为 +∞，全失败则阻断。平局按配置顺序。消融用完整模型验证所得参数，不重新利用测试调参。

pilot 默认 2 配对种子、1 月、预算 4/6；正式计划 30 配对种子 0…29、四月、4/6/8/10/12，pilot 后按耗时估计冻结，不依据精度筛选规模。没有验证输入时只提供计划规模，不能把 pilot 升级正式运行。

在线污染独立场景：clean；Gaussian σ=0.02℃；Student-t(df=3) 乘 σ√((ν-2)/ν)，与 Gaussian **等方差**；outlier 为 σ 高斯底噪+独立 5% 概率 ±0.5℃；missing 为 σ 高斯底噪+独立 20% 缺测；LF-bias 在线 LF 加 0.2℃，观测 clean。训练污染另列 train_gaussian/train_student/train_outlier/train_missing；幅度同上，只改变训练 POD/covariance 输入，在线观测 clean，保持原训练尺度/气候态。训练缺测用污染后训练列均值填补，全列缺测报错；不是在线缺测填补，也不是模型内鲁棒 POD。主协议不同时叠加训练和在线污染。原始训练文件不改写。

同一(seed,date,scenario)由独立 RNG stream 生成全节点扰动、missing mask、future-observation 扰动并保存；训练 RNG 与在线 RNG 分离。planner 接口没有 truth、扰动或观测器；只向 inference 提供已选观测。重构复用 Maximin 的 acquisitions。

消融：去 robust likelihood 固定 λ_i=1 并删除相应 Gamma prior/entropy；去 MF 删除整个 N 维 compatibility 与 ρ/τδ 更新和 ELBO 项，不能仅 l=0；去熵/ΔU/Red/cost 分别删除 MC-EIR 对应分子项或除数。硬约束均保留。

## 指标、统计及失败

主指标 sqrt(Σ cos(lat)e²/Σ cos(lat))（规则 1° 格点面积权重）；MAE 同权；NRMSE 分母为训练异常在固定评价格点的加权 RMS（℃），所有方法固定，不除以可接近零的目标均值。报告 requested/actual count、有效观测数、全网 mission J、成本、可行性、preprocess/selection/reconstruction seconds。推断计时不含绘图、审计和额外诊断。

每个重复按时间和节点预生成扰动。配对差值为方法 minus RBMF（重构）或 MC-EIR（部署）；负数有利于被比较方法。先对种子取均值，环境是月份而非噪声种子；仅一年四月不能估计跨年泛化 CI。报告对这四月条件成立的 Monte Carlo seed CI 和时间/空间分块敏感性 CI，均不得写成独立环境泛化保证。空间块预设纬度2°×经度5°，月份按时间顺序相邻2月循环块；空间/时间重采样保持方法配对。四月块 bootstrap 小样本不可靠，标注 exploratory；只有一个月时环境 CI 为 unavailable。多比较采用 Holm 校正，仅在足够独立环境支持检验时启用；当前正式环境检验不执行，p 值为 unavailable，不把 seed 作为独立环境。

潜在场区间：固定基底 mean-field Gaussian μ±1.96√U；未来观测区间用 q(a)、q(τ) 和新 λ~Gamma(ν/2,ν/2) 独立抽样卷积，不能用已观测 λ 或测试真值校准。遗漏基底学习、截断、模型差异/相关性及 mean-field cross covariance；确定性方法不伪造区间。future coverage 使用独立保存的未来观测噪声；missings 没有值的不计覆盖分母。

失败包括异常、max_iter、无可行解、阈值停止、超时。保存部分轨迹及原因，主结果不剔除；成功子集误差须同时给分母/成功率。每个任务最多600秒，pilot最多同一限制，超时保留落盘部分结果。不同 actual count 不合并同预算误差。预处理失败整次阻断。数值容差1e-12用于复现，计时/UUID不要求一致。

## 验收与归档

独立唯一目录，保存 manifest/config/源码 zip 与 hash/输入 hash/软件版本/数据摘要，冻结域、训练 artifacts、扰动、每步节点/观测/预测/Σα/factors/迭代/约束与异常。正式任务不能覆盖已有目录。绘图只读保存数据；CSV、SVG、PDF、300dpi PNG；空间样本固定最早月份、最小种子、clean、最大已计划预算，场值共用色标，误差使用独立对称共同色标，不择优样本。

关键验证：训练/验证/测试隔离、未来真值不可见、配对扰动、统一约束、独立小问题参考、落盘指标重算、固定 seed 可重复、消融删除正确因子。先通过小规模端到端验收才允许正式批量。缺输入时如实保留 blocked 状态，不生成替代真实数据。
