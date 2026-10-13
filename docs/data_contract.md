# 数据与工程接口约定

当前实施范围是阶段 1–3。RBMF-CS、MC-EIR、硬约束和 synthetic 闭环已实现；NetCDF 与真实部署数据实验属于阶段 4。

## 数组、拆分与单位

- `train_hf/train_lf`: `(T_train,N)`；`test_hf/test_lf`: `(T_test,N)`。模型 `fit` 每次输入一个 `(N,)` LF 与 `(p,)` HF 观测。
- 网格扁平化顺序为 C order，即 `(ny,nx)` 中 x 变化最快；固定 node ID 与训练、LF、HF、冰况、坐标完全一致。
- Synthetic `coordinates_km` 为平面局部笛卡尔 **km**，不假装经纬度。场值为 synthetic arbitrary units，无真实海洋物理含义；ice 为 `[0,1]` fraction。
- temporal split 在训练数据生成后取前 `n_train`，其余作为 holdout；整个训练过程禁止读 `test_hf`。synthetic AR 产生跨分界相关性，本 demo 不等价于严谨带时间间隔的 generalization 实验。
- HF 训练均值 `(N,)` 与全局 scalar RMS 对两个 fidelity 使用相同转换。训练 covariance 用训练 anomalies，不使用在线目标。future spatial holdout 必须在定义的训练域拟合，不能把空间 holdout 中的目标值用来补均值/基底。
- fixed POD basis `(N,K)` 在线只读；snapshot 变化可重新 `fit`，但本阶段 `update` 表示**同一目标快照**的新增独立观测，不能混入不同时间当作同一 latent state。
- HF 观测 NaN 按行丢弃；不能补零；inf 拒绝。LF、training 与基底要求 finite；陆地/永久缺测点须在训练前定义共同 valid-domain mask。暂不替 LF 缺测自动插值或变更 compatibility factor。

## 当前可运行接口

```python
from rbmf_cs import RBMFCS, VBConfig

model = RBMFCS(fixed_basis, VBConfig())
model.fit(low_fidelity_anomaly, observed_anomalies, indices=node_ids)
print(model.diagnostics_["converged"])
model.update(new_anomalies, indices=new_node_ids)
mean = model.predict()
variance = model.posterior_variance()
product = model.covariance_product(vector)
```

或者提供 `sampling_matrix=C`；它与 `indices` 二选一。一般线性观测必须已在同一 anomaly 单位中，physical y 转换为 `(y−C@training_mean)/scale`。重复索引代表独立重复观测，因此局部 λ 按 observation row 存储。部署集合按唯一 node ID 计数。

## 阶段 3 的部署约定

1. Feasibility 在评分前检查 entire prospective deployment；当前已有节点因环境变化违规时停止扩张并报告 operational reassessment。
2. 声明三类节点：候选 sensing node、已选 sensing anchor、固定 non-sensing gateway。sensing anchor 计入 sensor budget、ice 检查和 mission 能耗；non-sensing gateway **默认不计 sensor count，但计入单列固定 mission energy**，通信图包含所有固定 gateway。必须把政策和能耗值写入每次 run 的配置；当前实现固定采用此口径，gateway_energy_j 可配置为零，但不能静默漏计。
3. connected seed 由用户/实验配置给定。算法检查，不会自动加 relay 来使输入变成可行。无 gateway 时 singleton 按论文视为 connected；当前 runner 明确拒绝空 sensing seed，不自动选取启动点。
4. `coordinates_km` 使用 Euclidean 距离，阈值为 km。若输入 `latitude_deg/longitude_deg`，用声明的球面 great-circle 距离或指定投影转换，处理跨日界线；绝不能直接把 degree 差当 km。
5. mission energy 为同一 duration 内的 Joule；sense/process/tx 需要模型与参数。拓扑依赖 tx 时每次重算整个 prospective set 的路径及所有节点 tx。
6. 任一冰况/count/energy/connectivity 违规即排除；缺失关键可行性数据不能默认安全。恰好等于 ice/count/energy 阈值可行，通信半径边界策略采用 `distance <= radius` 并记录数值容差。
7. Candidate planning 默认 `r=1/E[tau]`，可显式加入事先已知 `omega>0`；不能用尚未取得的 y/residual。完整 robust refit 改变全部因子。两者须独立的函数和日志字段。
8. 保存每步 selected node、全 feasible 候选 raw/normalized ΔH/ΔU/Red、缩放 max 与实际 denominator、ΔI、C_eff、EIR、各约束判定及排除原因、累计能耗、冰况、连通性、方差、fit convergence 与 runtime。归一化只覆盖当前 feasible 集；全零 gain 仍为零。
9. 贪心平局按最小稳定 node ID；停止原因至少包括 count budget、无 feasible candidate、max EIR≤非负 threshold、已有部署失去可行性。阶段 2 demo 的预定随机顺序保留；阶段 3 closed-loop demo 完整实现这些规则。

## 阶段 4 的 NetCDF 接入配置合同

后续 adapter 应接受独立的 HF/LF 文件列表与以下显式字段（本次不创建虚假加载器）：

| 字段 | 必须声明的内容 |
|---|---|
| `hf.paths`, `lf.paths` | 各自实际 NetCDF 文件；禁止用 HF 测试真值生成 LF |
| `variables.target/ice` | 目标和冰况变量名，允许两产品名称不同 |
| `coordinates.time/latitude/longitude/depth` | 坐标变量名、维度布局、选定深度或聚合政策 |
| `time.selection`, `calendar`, `alignment` | 时间范围、日历、nearest/exact/aggregation 策略及最大允许偏差 |
| `spatial.target_grid`, `regrid_method`, `crs` | 公共网格、投影/插值算法；经度 0–360 与 −180–180 的处理 |
| `units.target`, `units.ice` | 温度等物理单位转换，ice percent→fraction；禁止无声明转换 |
| `missing.fill_values`, `valid_range`, `mask_policy` | CF fill values、NaN、QC flags、陆地/缺测域；禁止补零 |
| `split.train/validation/test`, `holdout_gap` | 时间与空间分界、时间隔离间隔；参数选择仅用 train/validation |
| `alignment.audit` | 有效节点 ID 映射、插值来源、时间偏差、丢弃计数及原因 |

推荐实际 adapter 在配置不充分时明确失败；不得悄悄把 test 当 train 或把未来 HF 用于校准 LF。现已实现特定产品的50 m对齐流程 `scripts/align_polar_sample.py`（IAP温度异常、ORAS5位温与盐度、NSIDC海冰、ITP/Argo），其固定科学政策和结果见 `real_data_alignment_report.md`；其他产品仍需配置与适配。

该流程输出为时间×格点，POD为格点×模态；物理场为℃，模型数组无量纲。2014全年不得用于POD、逐月气候态、中心化、尺度或协方差拟合。点观测的 `node_ids` 与 `weights` 构成稀疏观测算子C，不可直接当成最近格点的点采样；必须对气候态和训练均值应用相同的C。现有推断器支持sampling_matrix，但闭环acquire(node_id)仍是格点接口，不能自动把瞬时漂流剖面当成可部署的月平均节点观测。

## 后续实验比较合同

所有重构/部署基线复用相同 split、固定 training artifact、HF 观测预算、LF 访问权限和 observation realization；分别报告其是否满足工程约束。无约束 QR/Random 的准确率不等于工程可部署性。调参只在训练/验证段；runtime 包括对应模型实际训练/在线推断成本，记录 hardware。NLPD/CRPS/coverage 须明确评估 latent field 还是带 measurement noise 的 predictive distribution。


## 已实现的闭环接口

```python
from rbmf_cs import run_closed_loop
result = run_closed_loop(model, low_fidelity, seed, seed_observations,
                         constraints, planner, acquire)
```

`constraints` 是 `DeploymentConstraints`，`planner` 是 `MCEIR`；`acquire(node_id)` 返回同一快照的一次观测。`result` 包含 selected、stop_reason、decisions、trajectory、snapshots、events。完整配置示例见 `configs/closed_loop_demo.json`，运行 `python scripts/run_closed_loop_demo.py`。

保存的 observation null 代表 NaN 缺测；节点仍留在物理部署中，节点数和能耗照常计算，RBMF-CS 按既有合同删除 NaN 观测行并重拟合。

任务能耗是统一时间窗的 whole-network forecast。每轮保存 energy_change，而非重复累加 whole-network totals。路径重新规划可改变旧节点能耗。当前射频模型不含 receive/idle 能耗，这一限制随结果清单保存。具体公式与最短路径口径见 `math_to_code.md` 的阶段 3 细节。
