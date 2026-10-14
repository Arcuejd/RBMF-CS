# 阶段 1、2 实际验证记录

执行日期：2026-10-08（用户时区 Asia/Shanghai）。这是软件/数值验证与 synthetic demo，不能作为论文真实极地数据结果或基线优越性证据。

## 已执行命令

在 `D:/project/RBMF-CS`：

```powershell
python scripts/run_tests.py
python scripts/run_synthetic_demo.py --config configs/synthetic_demo.json --output outputs/synthetic_demo
python scripts/check_reproducibility.py
```

实际环境 Python 3.13.7、NumPy 2.4.4、SciPy 1.17.1。测试入口以 src 导入包，不要求 pytest。安装元数据 `pyproject.toml` 已经 tomllib 实际读取验证；未下载依赖或修改系统 Python 安装。沙箱 Python 启动器会打印 `Failed to find real location of D:\python\python.exe`，但仍执行 Python；最终三条命令都成功完成。

## 结果

- **26 tests，0 failures，0 errors，0 skipped**；最终标准库测试套件实际用时约 0.801 s。机器可读结果 `outputs/test_results.json`。
- **两个独立新进程，35 项比对通过**：synthetic/train artifacts/observations/posterior 数组使用 1e-12 tolerance，全部变分 factors 与 ELBO history 精确一致；排除 timestamps/runtime。详见 `outputs/reproducibility/report.json`。
- 论文 SHA-256 在开发前后相同；`outputs/source_integrity.json` 记录只读校验，原稿未修改。
- 157 个 native OMML 已提取，85 个编号公式（4-1…4-55、5-1…5-30）全部找到。

| synthetic step | 本步新增节点 | 有效观测数 | VB iterations | converged | RMSE（arbitrary units） | 潜在场总方差 |
|---|---:|---:|---:|---|---:|---:|
| initial | 初始 12 次请求，1 个 missing | 11 | 73 | true | 0.05701613 | 0.23101500 |
| update 1 | 87 | 12 | 16 | true | 0.05679736 | 0.21842428 |
| update 2 | 113 | 13 | 30 | true | 0.06007052 | 0.22000994 |
| update 3 | 20 | 14 | 22 | true | 0.05776757 | 0.20881371 |

节点 ID 从 0 开始。每次 update 都调用完整 Algorithm 1；fit_count 分别为 1/2/3/4。四次拟合无显著 ELBO 下降，无 Cholesky jitter。第二次更新的误差和不确定性均上升，这是保留的实际结果，未施加不合理单调性要求。最终全流程耗时见 run.json，单次拟合耗时也按轮保存；这些小规模本地时间不构成性能 benchmark。

## 有意义的验证依据

1. 固定超参数 Gaussian：独立 whitened augmented least squares + SVD 给出 mean/covariance，与 Cholesky 实现一致；另有单系数 posterior 的数值积分。
2. Student-t 局部精度：对 prior×likelihood 在 λ 上积分求均值，核对 Gamma update；大残差明显降权。
3. Discrepancy energy：独立抽取 150000 组 α/ρ 样本，两类方差贡献都远大于 MC 误差；与期望 energy 一致。
4. ELBO：使用独立 scipy.stats normalized joint/q density 对 60000 个 posterior draws 积分，检查包括 N 维辅助观测常数及六组因子 entropy 的完整表达。
5. 对称/PSD：K×K 系数协方差和小规模 full Σ_x；diagonal、trace、matrix product 与直接矩阵一致；大型 full covariance 调用有保护。
6. Missing：NaN y/H 删除与手工删行得到完全相同结果，且与补零实现显著不同。
7. Warm-start：保留旧 factors，新 λ 初始期望为 1，加入观测后 mean/covariance 及 gamma/tau/delta 实际变化；检查实际 fit_count。
8. 收敛：ELBO 与全因子两种 stopping 模式、max_iter 未收敛状态、零 LF 保留 ρ prior、无观测时 τ 返回 prior、输入验证。
9. Training：POD orthonormality、重构、train-only transform、篡改 test 不能改变训练 artifacts、训练 covariance 与 numpy empirical covariance 一致。
10. Synthetic：固定 seed 可重现，邻近空间相关性，修改 HF discrepancy 不改变 LF，temporal 分界明确。
11. 配置 demo：直接调用 Python `run` 接口，四轮都收敛且每步输出存在；不构造 full field covariance。

## 实际发现和处理

- 初始 discrepancy prior Gamma(0.01,0.01) 导致该 synthetic LF 近基底子空间时 τ_delta 约 5187，首轮在 2000 iterations 时 relative ELBO change 约 1.59e-7，未满足 1e-8。四轮均如实标记未收敛。完整初始数据、配置和轨迹保留于 `outputs/initial_weak_prior_run`。
- 改用明确的 Gamma(2,2) discrepancy 正则化先验，保持原容差后四轮收敛。该数值由工程 conditioning 选择，不是论文指定值或真实数据优化结果。
- 新增端到端回归测试发现 Windows sandbox 默认 temp 目录不可写。获准沙箱外重跑确认算法测试正常后，测试改为只在项目 outputs 下创建与清理临时目录；最终沙箱内 26 项通过。
- 整理 Windows 文本编码，确保 pyproject TOML 可由标准解析器读取；修复 demo callable 对脚本全局启动时间的依赖，端到端测试覆盖直接调用。

## 尚未完成及适用限制

- 按本次范围，MC-EIR rank-one ΔU、候选 hard filtering、停止条件、取得观测后重新评分等 **阶段 3 测试尚未实施**；不得把阶段 2 assimilation smoke test 称为完整部署闭环。
- NetCDF adapter、所有基线、鲁棒性/消融/敏感性批量实验、正式 temporal/spatial holdout 和 calibration/feasibility/cost 指标属于阶段 4。
- 当前测试数据规模小；大网格避免 N×N 的设计已落实，但未进行大规模内存/吞吐基准。
- 共享 anomaly normalization、POD rank、先验、网格分辨率依赖、ρ 弱识别性与 mean-field 校准限制均在 math_to_code 中说明。没有需要阻塞阶段 1、2 的未读公式或模型矛盾。
