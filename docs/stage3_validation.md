# 阶段 3 实际验证记录

本记录只报告已运行的软件和 synthetic closed-loop 验证。没有真实极地实验、基线优越性、校准有效性或 greedy approximation guarantee 的结论。阶段 1、2 历史记录仍在 validation_report.md。

## 已实现

- constraints.py：whole-prospective-set ice/count/mission-energy/connectivity 硬筛选；connected seed、sensing anchor、固定非感知 gateway；km/haversine 距离。
- planning.py：条件 training entropy、固定超参数 Gaussian planning ΔU、绝对相关 redundancy、feasible-set max normalization、raw/normalized effective information、maintenance/failure soft costs、EIR 和选择停止。
- deployment.py：当前完整 posterior → feasible filter → score → acquire 一次 → 完整六组 VB warm-start → 下一轮重新评分。
- 保存所有候选的可行性及排除原因、raw/normalized score、归一化尺度、选点、任务能耗/路由、冰况/连通性、posterior、全部 inference factors/ELBO 与 runtime。

## 实际执行

```powershell
python scripts/run_tests.py
python scripts/run_closed_loop_demo.py --config configs/closed_loop_demo.json --output outputs/closed_loop_demo
python scripts/validate_closed_loop_run.py outputs/closed_loop_demo
python scripts/check_closed_loop_reproducibility.py
```

Python / NumPy / SciPy: 3.13.7 / 2.4.4 / 1.17.1。
测试 **53 项，0 failures，0 errors，0 skipped**。其中阶段 3 新增 27 项，原阶段 1、2 的 26 项也全部通过。完整 test runner 耗时含 discovery/import 为 2.898 s。

## 独立验证

- 单元测试用增广 whitened Gaussian 设计矩阵的 SVD 重新求 covariance，再直接计算 tr(ΨΣΨᵀ) 之差，核对 (5-11)；包含非正交 Ψ。
- 条件结构 entropy 用独立 ridge regression 的优化后残差能量核对，不只重复 Schur complement 公式。
- 独立输出审计脚本不 import rbmf_cs：用 scipy graph connected_components、shortest paths 重新核对 whole-network 可行性和源到 sink 的总路径能耗；用独立 SVD 对每个 scored candidate 核对 ΔU。
- 本次审计 **683 个可行性记录、69 个 ΔU**。最大 ΔU 绝对误差 **1.092e-16**；最大能耗绝对误差 **5.684e-14 J**。
- 单独 packet ledger fixture：一节点总 9 J，两节点有中继总 14 J；旧节点 tx 从 1 J 增为 2 J，证明只算新节点能耗会漏算。重新路由 fixture 的 total 从 17 J 降到 15 J，证明 prospective 全网重新计算有效。
- 四类硬边界、unknown ice、已有节点冰况改变、gateway/anchor 计数、singleton、多个 gateway 的 augmented connectivity、latitude/longitude 与跨日界线距离均覆盖。
- 测试最大 EIR、精确 tie、threshold equality、空候选、零增益、负有效信息、预算停止、已有部署失效、未收敛推断停止。
- 闭环用实际 model.update/_run 调用检查完整 VB；对每轮 posterior factor 的保留/变化、fit_count、后续 raw ΔU 与更新 covariance 逐项对照。改变取得的新观测，首轮评分不变，下一轮评分改变；不强制误差或 full-VB variance 单调降低。

## Synthetic 闭环运行结果

Connected seed `[0,1,12,13]`，sensing anchor `[0]`，non-sensing gateway `[[-5,0]]` km。ice threshold 0.56；通信半径 22 km；sensor budget 12；同一 3600 s mission energy budget 250 J。网关不占 sensor count，固定能耗 20 J 计入 mission。

最终 selected：`[0, 1, 12, 13, 3, 5, 7, 31, 18]`。
停止原因：`no_feasible_candidate`。
初始拟合及每次实际 acquisition 后完整重拟合，共 **6 次，全部收敛**。

| posterior step | 有效观测数 | VB iterations | converged | warm start |
|---|---:|---:|---|---|
| 0 | 4 | 68 | True | False |
| 1 | 5 | 10 | True | True |
| 2 | 6 | 13 | True | True |
| 3 | 7 | 13 | True | True |
| 4 | 8 | 12 | True | True |
| 5 | 8 | 3 | True | True |

| 新增节点 | 缺测 | EIR | 全网任务能耗 J | connected | posterior total variance（model units²） |
|---|---|---:|---:|---|---:|
| 3 | False | 1.32458877 | 104.300000 | True | 0.14114390 |
| 5 | False | 1.10864047 | 134.600000 | True | 0.14149405 |
| 7 | False | 1.02039657 | 172.100000 | True | 0.14092643 |
| 31 | False | 0.99004832 | 216.800000 | True | 0.14103873 |
| 18 | True | 0.99902305 | 247.100000 | True | 0.14103977 |

最终能耗 **247.1 / 250 J**，9 个 sensing 节点。最后一轮未选候选共有 111 个：86 个不连通、25 个连通但超过 energy budget；42 个还同时违反 ice threshold（这些原因可重叠）。没有用低分掩盖硬约束违规。

最后一次 acquisition 被 synthetic missing 配置置为 NaN：物理节点 18 仍计数、计费；该 y/H 行没有被零填充；Algorithm 1 在剩余有效观测上完整重跑。下一轮重新筛选得到空 feasible 集并停止。

本次全 demo wall runtime 约 0.216 s；只是小网格本地软件运行时间，不是可外推的性能结论。逐轮实际时间保存于 trajectory/fit JSON。

## 重现

两个独立新进程，**52 项数组、8 份 JSON** 比对通过。数组 tolerance 1e-12；全部候选/可行性/决策、acquisitions 和 inference JSON 去除 runtime 字段后精确一致。启动时间与 runtime 不要求相同。

## 边界处理与限制

- 实现时修正了空 feasible 集进入 covariance 子块索引的边界情况，确保返回空评分并正常停止；有独立空候选回归测试。
- 能耗日志明确为同一 mission 的当前全网 total，并另存相邻差值；不重复累加每轮 total。测试与审计验证 seed total + 差值之和 = final total。
- 缺测获取和 expanded-set 复核保留选点/观测事件，避免物理部署与有效统计观测数混淆。
- 最短 transmit-energy 路由、每包电子/距离平方能耗、mission 功率、成本和风险都是公开的 synthetic 工程假设；尚未包含 receive/idle radio 能耗或真实任务遥测。
- raw global EIR 只按新增步骤的显式 raw 权重累计，不能解释成精确 joint mutual information；归一化尺度逐轮变化。
- 真实数据接入、基线、消融、鲁棒性、敏感性、正式时空 holdout 和校准实验仍属阶段 4。原论文无修改。

## 输出位置

- `outputs/test_results.json`：测试摘要。
- `outputs/closed_loop_demo/`：全部配置、数据、training artifacts、deployment.json、trajectory.csv、acquisitions、逐轮 posterior 与 inference、run manifest、independent_audit.json。
- `outputs/closed_loop_reproducibility/`：两个独立运行与 report.json。
- `docs/math_to_code.md`：更新后的 Section 5/Algorithm 2 实际代码与测试对应。
