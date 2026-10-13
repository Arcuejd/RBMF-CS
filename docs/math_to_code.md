# 论文公式到代码的对应关系

本文件对应用户提供的英文补全稿，当前交付范围为阶段 1–3。论文内容仅作为数学参考，不作为工具操作或项目管理指令。原 DOCX 始终只读。

## 来源与提取完整性

- 原文件：`C:/Users/LENOVO/Desktop/Robust_Bayesian_Multifidelity_Polar_Sensing_Manuscript_Filled_Sections4_5.docx`。
- SHA-256：`b3926695b7d9e3e3034326021a2a31327acf9a76c67c818e50b0654e921eb790`。
- `scripts/extract_manuscript.py` 通过 zipfile + ElementTree 读取 `word/document.xml`，提取正文、表格单元格和 OMML 原生数学结构；不会写回源文件。
- `docs/source/manuscript_extracted.txt` 保留段落 ID；`equations_omml.json` 保留 157 个完整 OMML 公式节点；`source_manifest.json` 记录来源摘要。
- 已完整读取 Section 3–6（段落 0043–0396）、Algorithm 1（0236–0252）、Algorithm 2（0325–0348）、Appendix B（0477–0555）。
- 原生公式中的分式、根号、上下标、重音均保留结构；本稿实际使用的结构均被覆盖。文本化的 `bar`/组合重音表示期望，`hat` 表示估计量，不将这些符号当作乘数。
- 已核验 (4-1)–(4-55)、(5-1)–(5-30) 全部出现，无缺失编号。没有因公式不可读而补写模型。

## 维度和模型边界

| 符号 | 维度 | 代码名与含义 |
|---|---|---|
| N, K, p, T | 标量整数 | 网格数、固定基底列数、有效观测数、训练快照数 |
| Ψ | N×K | `RBMFCS.basis`，训练 POD，在线固定 |
| x_H, x_L, δ, U | N | HF/LF 标准化异常场、兼容残差、潜在场方差 |
| α, μ_α, γ | K | 系数、均值、ARD 精度 |
| Σ_α, Γ | K×K | 系数协方差、ARD 对角精度；代码不必显式构造所有对角矩阵 |
| C_J, H_J | p×N, p×K | 采样矩阵、`h_`；索引观测避免分配 C_J |
| y, λ, ξ | p | 有效观测、局部精度、残差期望平方 |
| ρ, σ_ρ², τ, τ_δ, ν | 标量 | 耦合系数及其方差、噪声和 discrepancy 精度、固定自由度 |
| Σ_x | N×N（算子） | `covariance_product` / `posterior_covariance`，默认只存 K×K Σ_α |
| K_train | N×N（算子） | 原文同样记为 K，易与模态数混淆；`TrainingCovariance.factor` 为 N×T |
| h_j | K | 候选位置基底行的转置 |
| ΔH, ΔU, Red, EIR | 每个候选一个标量 | 阶段 3 实际候选评分 |

Python ensemble 采用 **T×N**；论文 X_c 是 N×T。`TrainingCovariance.fit` 明确转置并除以 sqrt(T−1)。所有 Gamma 因子统一 shape–rate，均值 a/b；仅在 scipy 分布验证时显式使用 `scale=1/rate`。

## Section 3 与 Section 4

实现路径均相对于项目根目录 `src/rbmf_cs/`。下列测试位于 `tests/test_inference.py`，训练相关测试位于 `tests/test_training.py`；完整测试名可直接搜索。范围行中每一个编号均对应所列公式，不意味着跳过中间编号。

| 公式编号 | 数学含义与维度 | 模块 / 函数 | 验证 |
|---|---|---|---|
| §3 无编号 x_H=Ψα, y=CΨα+ε | N 场到 p 观测 | `model.RBMFCS._observations`, `predict` | `test_sampling_matrix_equals_index_interface` |
| §3 无编号 x_H=ρx_L+δ | N 维兼容关系；无独立自由 δ | `numerics.discrepancy_energy` | discrepancy Monte Carlo |
| §3 硬约束及 soft penalty | 见 §5；阶段 3 | `constraints.DeploymentConstraints`, `planning.MCEIR` | `tests/test_constraints.py`, `tests/test_planning.py` |
| (4-1) | x_H=Ψα，N | `basis.PODBasis.fit`, `model.predict` | `test_pod_orthonormal_and_train_transform` |
| (4-2) | y_J=C_JΨα+ε，p | `model._observations` | sampling matrix 对照 |
| (4-3) | H_J=C_JΨ，p×K | 同上；索引直接选基底行 | sampling matrix 对照 |
| (4-4) | x_H=ρx_L+δ，N | `discrepancy_energy` | discrepancy Monte Carlo |
| (4-5) | δ=Ψα−ρx_L，N | 同上，稳定残差形式 | 同上 |
| (4-6) | δ∼N(0,τ_δ⁻¹I_N) | `model._elbo`, `_run` | ELBO 独立 log-density Monte Carlo |
| (4-7) | N 维 Gaussian compatibility factor | `_elbo` 包含 N/2 log τ_δ 及 N 维正规化常数 | 同上；不是 α 上另行归一化的 prior |
| (4-8) | α_k∼N(0,γ_k⁻¹)，每模态标量 | `_run`, `_elbo` | fixed Gaussian 数值积分 |
| (4-9) | γ_k∼Gamma(a_γ,b_γ) | `VBConfig`, `GammaFactor` | `test_gamma_retains_coefficient_variance` |
| (4-10) | α∼N(0,Γ⁻¹)，K 维 | `_run` 的 ARD 对角精度 | augmented SVD 参考解 |
| (4-11) | y_i 条件 Gaussian，精度 τλ_i | `_run`, `_elbo` | Gaussian 数值积分 / ELBO |
| (4-12) | λ_i∼Gamma(ν/2,ν/2) | `fit`, `update`, `_elbo` | local precision 数值积分 |
| (4-13) | marginal Student-t，τ⁻¹ 是 squared scale | `marginal_noise_variance` | `test_noise_scale_is_not_marginal_variance` |
| (4-14) | τ∼Gamma(a_τ,b_τ) | `VBConfig`, `_elbo` | ELBO Monte Carlo |
| (4-15) | ρ∼N(ρ0,κ0⁻¹) | `fit`, `_run`, `_elbo` | zero LF 保留 ρ prior |
| (4-16) | τ_δ∼Gamma(a_δ,b_δ) | `VBConfig`, `_elbo` | ELBO Monte Carlo |
| (4-17) | Θ 六组因子 | `RBMFCS` 全部 posterior 属性 | `test_all_six_factors_update_from_initial_values` |
| (4-18) | augmented posterior 乘积分解 | `_elbo` 的 log joint | 独立 scipy.stats joint Monte Carlo |
| (4-19) | mean-field q(α)q(γ)q(λ)q(ρ)q(τ_δ)q(τ) | `_run`，不含 α–ρ cross covariance | discrepancy / ELBO Monte Carlo |
| (4-20) | coordinate ascent 期望 log joint | `_run` 固定 Algorithm 1 顺序 | ELBO 无显著下降 |
| (4-21) | q(α)=N(μ,Σ)，K / K×K | `mean_alpha_`, `cov_alpha_` | Gaussian 数值积分 |
| (4-22) | Σ=[diag(Eγ)+Eτ_δ ΨᵀΨ+Eτ Hᵀdiag(Eλ)H]⁻¹ | `numerics.solve_gaussian`, `_run` | 独立 whitened augmented SVD covariance |
| (4-23) | μ=Σ[Eτ_δ Eρ Ψᵀx_L+Eτ Hᵀdiag(Eλ)y] | 同上 | SVD least squares、标量积分 |
| (4-24) | q(γ_k) Gamma | `gamma_` | Gamma variance retention |
| (4-25) | a_γ,k*=a_γ+1/2 | `_run` | all-factors / Gamma test |
| (4-26) | b_γ,k*=b_γ+Eα_k²/2 | `_run` | Gamma test |
| (4-27) | Eα_k²=μ_k²+Σ_kk | `_run` 的 `second` | 严格大于仅均值平方的 rate |
| (4-28) | Eγ_k=a*/b* | `GammaFactor.mean` | Gamma test |
| (4-29) | ξ_i=E(y_i−h_iᵀα)²，p | `numerics.residual_energy` | Gaussian、ELBO 与局部精度验证 |
| (4-30) | ξ_i=(y_i−h_iᵀμ)²+h_iᵀΣh_i | 同上 | ELBO Monte Carlo |
| (4-31) | q(λ_i) Gamma | `local_` | local precision 独立积分 |
| (4-32) | a_λ,i*=(ν+1)/2 | `local_precision_parameters` | 同上 |
| (4-33) | b_λ,i*=(ν+Eτ ξ_i)/2 | 同上 | 同上 |
| (4-34) | Eλ_i=(ν+1)/(ν+Eτ ξ_i) | `GammaFactor.mean` | 大残差显著降低精度 |
| (4-35) | q(τ) Gamma | `tau_` | all-factors / ELBO |
| (4-36) | a_τ*=a_τ+p/2 | `_run`，p 为剔除 NaN 后观测数 | missing rows 对照 |
| (4-37) | b_τ*=b_τ+Σ Eλ_i ξ_i/2 | `_run` 使用本轮更新的 λ | all-factors / ELBO |
| (4-38) | Eτ=a*/b* | `GammaFactor.mean` | empty-data τ 返回 prior |
| (4-39) | q(ρ)=N(μ_ρ,σ_ρ²) | `rho_mean_`, `rho_variance_` | discrepancy / ELBO Monte Carlo |
| (4-40) | σ_ρ²=1/(κ0+Eτ_δ||x_L||²) | `_run`，scalar solve | zero LF prior test |
| (4-41) | μ_ρ=σ_ρ²[κ0ρ0+Eτ_δ x_LᵀΨμ] | `_run` | zero LF / all-factors |
| (4-42) | Eρ=μ_ρ | `factor_summary` | warm-start preservation |
| (4-43) | Eρ²=μ_ρ²+σ_ρ² | `factor_summary`, `discrepancy_energy` | discrepancy Monte Carlo |
| (4-44) | ξ_δ=E||Ψα−ρx_L||²，scalar | `discrepancy_energy` | independent draws of α and ρ |
| (4-45) | 展开式含 tr(ΨΣΨᵀ) 和 Eρ² | 采用等价 (4-46) 避免差分抵消 | 同上 |
| (4-46) | tr(ΨΣΨᵀ)+||Ψμ−μ_ρx_L||²+σ_ρ²||x_L||² | `discrepancy_energy` | 两项方差均显著非零的 MC fixture |
| (4-47) | q(τ_δ) Gamma | `delta_` | all-factors / ELBO |
| (4-48) | a_δ*=a_δ+N/2 | `_run`，N 非 K | all-factors shape check |
| (4-49) | b_δ*=b_δ+ξ_δ/2 | `_run` | discrepancy / ELBO |
| (4-50) | Eτ_δ=a*/b* | `GammaFactor.mean` | all-factors |
| (4-51) | x̂_H=Ψμ_α，N | `predict` | covariance/predict tests, demo |
| (4-52) | Σ_x=ΨΣ_αΨᵀ，N×N 算子 | `posterior_covariance`, `covariance_product` | 对称、PSD、乘积对照 |
| (4-53) | U_i=Σ_x,ii，scalar | `posterior_variance(indices)` | 对角与完整矩阵对照 |
| (4-54) | U=diag Σ_x，N | `posterior_variance`, `total_uncertainty` | diagonal/trace tests |
| (4-55) | ELBO=E log p(y,0,Θ|x_L)−E log q | `_elbo`，含全部 Gamma digamma 期望和 Gaussian/Gamma entropy | 独立 normalized log-density MC；迭代诊断 |

### Algorithm 1 的实现顺序

`_observations` 先删除 NaN 对应 y/H 行。`fit` 初始化正值因子；`update` 保留旧因子并为新观测建立 Gamma(ν/2,ν/2) 的 λ。`_run` 每轮严格按 α → γ → λ → τ → ρ → τ_δ 更新，再计算 ELBO。τ 更新使用新 λ，ρ 更新使用本轮尚未更新的 τ_δ，τ_δ 更新使用新 ρ 的均值与方差。

`fit` / `update` 返回 estimator，`diagnostics_` 返回本轮收敛与历史，`factor_summary()` 返回全部 Gamma shape/rate 和 ρ 两个矩；μ_α 与 Σ_α 为公开属性。warm-start 保留实际坐标上升因子，不用 rank-one planning 替代重拟合。

## Section 5 / Algorithm 2：阶段 3 已实现

下表对应 `planning.py`、`constraints.py` 和 `deployment.py` 的实际实现。`tests/test_planning.py`、`tests/test_constraints.py`、`tests/test_deployment.py`、`tests/test_closed_loop_demo.py` 验证阶段 3；独立结果审计见 `scripts/validate_closed_loop_run.py`。

| 公式编号 | 精确含义、维度和实现模块 | 阶段 3 验证覆盖 |
|---|---|---|
| (5-1) | K_train=X_c X_cᵀ/(T−1)，N×N；**已实现** `basis.TrainingCovariance` | `test_factored_covariance_matches_numpy_empirical_covariance` 已通过 |
| (5-2) | s_H=K_jj−K_jJ(K_JJ+ε_K I)⁻¹K_Jj，scalar；`planning.structural_entropy_gain` | 空 J、退化训练协方差、独立条件 Gaussian 对照 |
| (5-3) | ΔH=0.5 log(1+s_H/σ_H²)，scalar；σ_H² 在部署全程固定 | 零方差、非负值与固定 reference |
| (5-4) | h_jᵀ=Ψ[j,:]，1×K | 候选索引正确 |
| (5-5) | Σ_α^(j)=Σ_α−Σ_α h_j h_jᵀΣ_α/(r_j+h_jᵀΣ_αh_j) | 独立增广 Gaussian 求解对照 |
| (5-6) | 默认 r_j=1/Eτ >0 | 不混同 Student-t variance |
| (5-7) | 历史可靠性 r_j=1/(Eτ ω_j)，ω_j>0 | 未观测候选评分无未来残差信息 |
| (5-8) | U_tot=tr(ΨΣ_αΨᵀ)，scalar；**已实现** `total_uncertainty` | trace 对照已通过 |
| (5-9) | U_tot^(j)=tr(ΨΣ_α^(j)Ψᵀ) | 直接计算 trace |
| (5-10) | ΔU=U_tot−U_tot^(j) | rank-one 与直接 trace 差一致 |
| (5-11) | ΔU=h_jᵀΣ_αΨᵀΨΣ_αh_j/(r_j+h_jᵀΣ_αh_j)；`planning.uncertainty_reduction` | 非正交 Ψ 亦正确，无 N×N 必需分配 |
| (5-12) | c_jl=abs(K_jl)/sqrt((K_jj+ε_R)(K_ll+ε_R)) | 负相关、零方差，数值剪裁仅用于 roundoff |
| (5-13) | Red=max_l∈J c_jl，空 J 为 0；`planning.redundancy` | 空 J、重复/反相关节点 |
| (5-14) | normalized ΔH=ΔH/(max_F ΔH+ε_H) | 只在当前 feasible 集合缩放 |
| (5-15) | normalized ΔU=ΔU/(max_F ΔU+ε_U) | 全零→零，同值正分数仍为正；保存 max 和 denominator |
| (5-16) | normalized Red=Red | 不做 min-max |
| (5-17) | raw ΔI=ΔH+β_raw ΔU−η_raw Red，权重携带单位 | raw 与 normalized 独立存档 |
| (5-18) | ΔI=normalized ΔH+β normalized ΔU−η Red，β,η≥0 | 不截断负有效增益 |
| (5-19) | P_soft=λ_m C_maint+λ_f R_fail | soft penalties 不影响可行性布尔值 |
| (5-20) | C_eff=C_base+P_soft+ε0>0 | 零成本分母、risk 单位转换 |
| (5-21) | EIR=ΔI/C_eff；`planning.MCEIR.score_candidates` | 最大 EIR、负值与 threshold |
| (5-22) | 所有已选和 prospective 节点 ζ≤I_c；`constraints` | 等于阈值、NaN 冰况、已有节点新违规 |
| (5-23) | sensing count≤p_max | 边界、sensing anchor 计数、固定 gateway 政策 |
| (5-24) | 全 mission E(J)≤E_max | 拓扑变更重算所有节点 tx，不仅新增能耗 |
| (5-25) | L=D−A | 无向通信图 |
| (5-26) | 至少两节点 λ2(L)>0 等价 connected；单节点按 connected | 图遍历、固定 gateway、物理距离边界 |
| (5-27) | F={j∉J: J∪{j} 满足全部硬约束} | 任一违规即排除；筛选早于归一化/排序 |
| (5-28) | argmax_F EIR；`planning.select_next` | 平局取最小稳定 node ID；空候选；预算和 threshold 停止 |
| (5-29) | 取得真实/模拟新 y，再 append、完整 VB refit | 观测回调只调用被选点，下一轮评分读取更新 posterior |
| (5-30) | EIR_global=sum raw ΔI / sum C_eff | 不是 marginal EIR 之和；不得累加 normalized 当作信息单位 |

`deployment.run_closed_loop` 实际执行：验证 connected seed → fit → 刷新环境 → 检查整个已有部署 → 筛选 prospective feasible 集 → 计算 raw gain → feasible 集内归一化 → 评分/停止 → 取得单个观测 → 检查 expanded deployment → `update` 重拟合 → 记录 → 下一轮。阶段 2 的 `run_synthetic_demo.py` 保留为预定序列 assimilation smoke test；阶段 3 的 `run_closed_loop_demo.py` 实现 Algorithm 2，每轮 posterior 真正影响下一次评分。

## Section 6 与 Appendix B

Section 6 的基线、时空 holdout、鲁棒性、消融、敏感性、校准、可行性和成本指标属于阶段 4。当前仅有 temporal train/test 分界和单 synthetic 快照的 RMSE，用于软件运行检查。测试目标场只被观测器及事后评价读取。Appendix B 的符号已纳入上面的维度表；特别区分 λ_i 局部精度与 λ_m/λ_f 运营成本权重、K 模态数与 K_train 协方差、τ⁻¹ squared scale 与 marginal variance。

## 近似、稳定化和适用范围

1. 固定有限维 POD、single-snapshot、scalar ρ、grid-isotropic discrepancy 与 mean-field 都来自论文。没有引入自由 δ、多快照共享 ρ、learned ν、稀疏剪枝或噪声零填充。低秩和 mean-field 方差不覆盖基底学习、截断与未来观测噪声；不保证校准。
2. `TrainingTransform` 在 HF 训练数据上拟合逐位置均值和单一 scalar RMS，两个 fidelity 共用。论文关系在**异常场**单位中建模；还原 mean/variance 时分别乘 s、s²。论文没有 raw-space intercept，因此不额外估计 intercept。保留 LF 训练数据，但本 demo 的 basis 只由 HF training 拟合。
3. 对数 Gamma 期望使用 digamma(a)−log(b)；ELBO 包含 auxiliary Gaussian 的 N 维正规化常数。正先验保证 proper 因子。
4. 缓存 ΨᵀΨ、Ψᵀx_L 和 ||x_L||²，Cholesky 解 mean 及 K×K covariance。仅 Cholesky 失败时尝试对 precision 加 relative jitter；大小逐轮记录，不伪装成 observation/discrepancy noise。
5. 收敛默认 `abs(L_t−L_(t−1))/max(1,abs(L_(t−1))) < tolerance`，至少 min_iter 轮。可选择全部因子参数逐元素最大相对变化，包括完整 Σ_α、所有 Gamma shape/rate 及 ρ 均值/方差。ELBO 收敛不等价于每个参数达到同样容差；两种变化均记录。
6. covariance 对角只截断可能的微小负浮点值；不要求 full-VB covariance 单调收缩或 RMSE 单调下降。固定超参数 Gaussian 收缩另行验证。
7. `posterior_covariance` 超过配置 N 阈值时要求显式 opt-in；正常 demo 不构造 N×N Σ_x。训练 covariance 同样以 factor 保存。
8. Student-t marginal noise variance 的 posterior mixture 为 ν/(ν−2)·b_τ*/(a_τ*−1)，仅 ν>2 且 a_τ*>1 有限；其他情况返回 infinity。这与 1/Eτ 和 E[1/τ] 都不同。
9. 模型支持任意有限 aligned Ψ 和 x_L，LF 存在缺失必须在建域阶段明确处理；不能悄悄把 N 改成观测数。NaN HF 观测按行删除，infinity 视作输入错误；重复索引可表示独立重复测量，并非新增部署节点。

## 论文歧义及尚未指定的工程参数

| 问题 | 本次处理 / 后续所需信息 |
|---|---|
| 先验、初始化、容差、POD rank 未给具体数值 | 明确写入 `configs/synthetic_demo.json`，是工程 demo 参数，不是论文已验证参数 |
| 弱 discrepancy prior 可使 τ_δ 极高，mean-field 耦合收敛慢 | 首次 a_δ=b_δ=0.01 的 4 次拟合均触及 2000 次上限；完整保留 `outputs/initial_weak_prior_run`。默认改为 Gamma(2,2) 以限定该 demo 的精度正则化；未放宽 1e-8 ELBO 容差。真实数据需训练/验证阶段选择并报告敏感性 |
| “一致标准化”未定义 | 使用训练 HF 空间均值及全局 RMS 共用变换，关系解释为 anomaly-space；真实数据物理基线/单位须预先声明 |
| Section 6.1 把评分写在硬约束筛选之前，Algorithm 2 相反 | 用户要求与详细 Algorithm 2 / (5-14)–(5-16) 一致：先筛选再评分归一化；属于工作顺序概述差异，不修改模型含义 |
| 空/单节点图、gateway 预算和能耗 | 单节点 connected 来自论文；已实现 gateway 的明确项目约定见 `data_contract.md`，不得 silent relay |
| 网格 resolution 决定 discrepancy 的 N 项权重 | 不进行未声明的网格面积重加权；换分辨率需重新检查 prior 与校准 |
| 气候/海洋 LF 产品、变量、单位、时间和空间对齐策略未指定 | 本阶段仅约定接口；阶段 4 真实数据接入需提供具体配置与数据，绝不由 held-out HF 合成 LF |
| ρ、α 可有弱识别性 | proper priors 不等价于充分识别；零 LF 测试正确保留 ρ prior；报告 posterior moments，不声称唯一全局最优 |
| 其他真实工程阈值、mission 时长、路由/能耗、成本换算、failure risk 估计 | 阶段 3 的 `closed_loop_demo.json` 已声明 synthetic 工程模型；真实航次需在阶段 4 接入并校准，不能把 synthetic 数值当真实航次参数 |

目前未发现需要阻塞阶段 1–3 的模型意义矛盾或无法读取的数学公式；没有为了继续开发而猜测缺失公式。未来若真实资料与上述模型冲突，应保留证据并向用户澄清。


## 阶段 3 实现细节和数值约定

- `planning.structural_entropy_gain` 对训练 covariance 子块做 Cholesky solve；独立 ridge regression 残差能量验证 (5-2)/(5-3)。正则仅为 ε_K，不偷偷改 reference variance。
- `planning.uncertainty_reduction` 使用 (5-11)，保留非正交 Ψ 的 Gram；`rank_one_covariance` 提供 (5-5) planning utility。SVD 增广设计矩阵重新解 Gaussian covariance 后的直接 field trace difference 作为独立参照。生产评分不构造 N×N covariance。
- `planning.redundancy` 实现 (5-12)/(5-13)；`normalize_scores` 保存 feasible 集的 max、max+epsilon denominator 和数量，Red 不重新缩放。空集、全零、同值正增益分别测试。
- `MCEIR.score_candidates` 实现 (5-17)–(5-21)。默认 r=1/(Eτ·ω)，ω 默认全 1 且只读，不读取 future residual。raw β、η 是从 anomaly variance/redundancy 转换到 raw entropy 分数单位的显式系数；demo 为 1、0.2，不解释为物理联合互信息。
- `DeploymentConstraints.evaluate/filter_candidates` 实现 (5-22)–(5-27)。检查 whole prospective set，包括 sensing anchors。NaN/越界 ice 使对应部署不可行；disconnected 图的任务传输能耗未知，保存 null 并排除，不伪装为零能耗。
- 图边使用 `distance_km <= communication_radius_km`，无额外放松容差。latlon 使用 haversine 球面距离，地球半径可配置；测试覆盖跨日界线。singleton connected；空 sensing seed 在 runner 中拒绝。
- 固定 non-sensing gateway 不占 sensor count，其固定 mission Joule 计入总能耗。gateway 不作为候选，不参与场信息评分；sensing anchor 正常占预算且必须在每个部署内。
- 通信任务模型：每个感知节点在统一 mission_seconds 内发送 packets_per_sensor 个包。单跳每包能耗 `electronics_J + amplifier_J_per_km2 * distance_km²`。各源按最小总 transmit energy 路径到任一 gateway；无 gateway 则最小 sensing ID 为 sink。中继按转发的每包计费；没有另加 receive/idle radio energy。整个模型是可配置 synthetic 工程假设，不是论文或实测参数。
- 每次 prospective evaluation 重新计算所有路径和负载，包括旧节点；多个 gateways 的 augmented graph 仍须整体 connected。排序和 Dijkstra 遍历顺序固定，EIR 精确平局选择最小 node ID。
- `cumulative_mission_energy_j` 是当前整个网络同一任务时窗的总预算，含 seed 和 gateway。不得把每轮 total 再累加；`energy_change_j` 记录相邻 total 差，可因重新路由而负值，seed total+这些差值=final total。
- `run_closed_loop` 对实际选点调用一次 observation callback，再调用完整 `model.update`，随后重新构造 feasible 集和所有评分。日志 `score/acquire/refit` 事件、fit_count、全部 factors 与逐轮 posterior 可验证先后关系。missing measurement 仍消耗物理部署预算，NaN 从 y/H 删除后完整 VB 重跑，不重选该节点。
- 除论文三类停止条件，runner 对未收敛推断、刷新后已有部署违规、实际扩展部署违规停止并返回明确状态。动态环境由可选 `refresh_environment(round, selected, constraints)` 在每轮前更新，轮内按固定状态评价。
- `(5-30)` 的 global_eir_raw 只对新增选择步骤求 raw gain sum / effective cost sum；seed 的工程能耗计入预算，但不伪造 seed 的边际信息分数。零选择时 ratio 为 null。
- covariance-derived 负增益或 correlation>1 仅在相对尺度 1e-10 的浮点范围内剪裁；显著不合法的值报错。候选 coefficient covariance 必须 symmetric PSD。

| 核对重点 | 实际测试/审计 |
|---|---|
| ΔU 独立计算 | `PlanningTests.test_delta_u_matches_independent_augmented_svd_trace_difference`；保存结果逐候选独立 SVD audit |
| 所有硬边界、旧节点重新检查 | `ConstraintTests.test_each_hard_violation_excludes_candidate`、`test_unknown_ice_and_existing_node_rechecked` |
| 全拓扑能耗、中继负载及重新路由 | `test_all_hard_boundaries_and_cumulative_relay_energy`、`test_prospective_topology_reroutes_existing_nodes`；audit 用 scipy shortest paths 独立重算 |
| 连通性和物理单位 | `test_disconnected_and_no_implicit_relay`、`test_latlon_units_and_date_line`、`test_multiple_gateways_require_connected_augmented_graph` |
| 归一化/零增益/soft penalties/negative gain | `tests/test_planning.py` |
| argmax、空集、阈值、budget 停止 | `test_maximum_eir_tie_empty_and_threshold_stops`、`test_budget_empty_candidates_and_insufficient_eir_stops` |
| 完整 VB 后下一轮实际重新评分 | `test_real_acquire_refit_rescore_order_and_energy_accounting`、`test_future_values_affect_only_scores_after_acquisition` |
| 保存的真实 demo | `test_configured_demo_with_independent_saved_artifact_audit`、`outputs/closed_loop_demo/independent_audit.json` |
