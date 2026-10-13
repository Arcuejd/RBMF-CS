# 真实数据准备与接入方案

核对日期：2026-10-08。本文件是准备与实施规范，不代表已经下载配套产品或实现真实数据 runner。现有 IAP 文件与有效域检查见 `local_dataset_assessment.md`。

## 先固定任务和实验层次

两条可并行准备的路线：

1. 1 m 月度温度异常：以 IAP 作历史训练/网格参考场，准备独立来源 LF；从 IAP 参考场抽点可运行离线观测模拟，必须明确这种观测不是原位实测。
2. 冰下实测验证：优先筛选 ITP 的 10 m 或 50 m 数据，先统计观测轨迹、年份和月份，再决定区域。IAP 在这些深度有精确层且通过了完整有效域检查。不能承诺此前 70–80°N、20–60°E 窗口一定有足够 ITP 数据。

先做重构和实测对照，再加工程闭环。历史实测数据不能为任意算法选点生成新观测；回放闭环只能在已知有观测的候选节点中选择。可通过当前 `run_closed_loop(..., candidates=available_node_ids)` 限制候选，评分阶段不能读取候选观测值。若观测点使用一般线性采样矩阵，当前单独推断器已支持，部署闭环接口仍按网格 node ID，需要相应扩展或事先声明网格内聚合政策。

## LF：独立来源背景场

优先筛选 [ECMWF ORAS5 月度再分析](https://cds.climate.copernicus.eu/datasets/reanalysis-oras5?tab=overview)。官方产品为全球月度数据，约 0.25° 三极网格，75 个垂向模型层，提供位温、盐度及海冰等变量。CDS 中 1958–2014 为 consolidated，2015 起为 operational，跨段需记录版本和一致性。下载时保留原生经纬度、深度、海洋掩膜和产品说明；若接口不支持小区域裁切，下载后在本地裁切，不承诺请求时一定可按经纬度和深度子集。

准备方法：

- 先下载一个年份或若干月份作格式试验，再扩展到与 IAP 匹配的时段。验证产品的实际最新月份，不把网页“至今”当成文件已经存在。
- 下载温度任务需要的温度字段；若要把位温转换为原位温度，还需盐度、深度/压力及位置。盐度任务同样须区分实用盐度与绝对盐度。
- 读取真实的二维曲线经纬度，不把原生三极网格当规则 lat/lon。按产品网格及海岸掩膜选择重网格方法，保留插值权重与来源，不跨陆地插值，不外推到没有支持的数据域。
- LF 在训练/验证/测试月份都需要，但核心训练专用 POD、均值和协方差目前使用历史 HF。训练/验证期 LF 用于检查关系、基线表现和任何允许的 LF 校准。
- ORAS5 是独立来源的背景候选，不因其网格细或模型来源就自动是 LF。用相同变量定义，在验证段比较 LF-only 与参考产品、实测观测的表现，说明保真角色依据。
- 再分析同化了观测，产品来源独立不等于统计独立；要声明与验证观测的同化关系。若要求严格不含验证观测的背景，需要未同化这些观测的模型试验或具有可核查截止时刻的预测。

## 温度和异常单位先对齐

IAP 本地温度包是异常，ORAS5 位温通常是绝对数值。不能直接融合，也不能给两者分别减各自均值后假定物理定义相同。

优先获取对应版本 IAP 的气候态、异常基准说明和温度物理定义。随后选一种明确路线：

1. 将 IAP 异常用匹配气候态恢复为绝对温度，统一 HF/LF/观测的温度定义，再按训练段重新定义异常。
2. 保留发布的 IAP 异常，将 LF 和实测温度转换为同一温度定义，并减去同一位置、深度、月份的匹配气候态。气候态必须提前固定，不能借测试真值拟合转换。

物理单位对齐之后，模型层再使用 HF 训练均值和训练 scalar RMS 统一标准化：`z=(x-training_mean)/scale`。对于一般采样矩阵 C，观测为 `(y-C@training_mean)/scale`。位温转原位温度的标准函数参考 [TEOS-10 gsw_t_from_pt0](https://www.teos-10.org/pubs/gsw/html/gsw_t_from_pt0.html)。如果温度定义或异常基准尚未确认，adapter 应明确拒绝物理融合。

## HF 点观测：下载、质控、配准

冰下优先 [WHOI ITP 数据](https://www2.whoi.edu/site/itp/data/data-products/)，选择任务完成后的 Level 3 最终数据并下载轨迹。各仪器深度范围需要逐任务核对；例如 [ITP 73](https://www2.whoi.edu/site/itp/data/completed-missions/itp73-overview/itp73-data-description/) 的标准采样范围约 7–750 m，不能用这些剖面外推成 1 m 观测。

也可使用 [Argo GDAC](https://argo.ucsd.edu/data/data-from-gdacs/) 的区域剖面；优先 delayed-mode 调整字段，遵守 [Argo 变量和 QC 规则](https://argo.ucsd.edu/data/how-to-use-argo-files/)。对于 core 文件依据 DATA_MODE 决定 raw/adjusted；对于 BGC 则按参数模式判断。优先科学变量 QC=1，并检查压力、时间和位置 QC。冰下位置可能为估计位置，不能静默当作精确定位。

建议保留以下观测长表，不在原始文件上改写：

| 字段 | 内容 |
|---|---|
| observation_id, platform_id, profile_id | 唯一观测、设备、剖面标识，用于去重和独立分组 |
| time_utc, latitude_deg, longitude_deg | 原始时间和位置 |
| pressure_dbar, depth_m | 原压力、换算深度及换算来源 |
| variable, temperature_definition, value, unit | 原位/位温等定义、观测值和单位 |
| value_qc, time_qc, position_qc, pressure_qc | 质量标记 |
| measurement_uncertainty, position_uncertainty | 已知测量和定位不确定性；未知明确标记 |
| source_path, product_version, data_mode | 来源和处理版本 |
| target_month, target_depth_m, node_id 或 sampling_weights | 接入后的目标和空间对应关系 |
| time_offset, depth_bracket, distance_km | 时间/垂向/空间配准证据 |
| role, assimilation_status | assimilation 或 validation；是否已被参考/背景产品同化或未知 |

处理政策：

- 垂向只能在合格观测夹住目标深度时插值，声明允许的垂向间隔；缺少浅层数据则丢弃，不默认外推。
- 先按设备轨迹和目标网格单元分组，再按月聚合。漂流设备不能把整月沿轨迹的观测都挂在单一坐标上。
- 月度稀疏样本均值仍不等于精确月均值。记录观测日期数量、时间覆盖和代表性误差；不足以支撑月度 assimilation 的样本可用于单独的配准验证。
- 采样矩阵 C 可表示空间插值观测；若把点观测聚合为网格节点值，须保留距离与聚合统计，并限制空间偏差。不能静默最近邻匹配任意远的节点。
- 按 platform/cruise/time 分组划分 assimilation 与 validation，不能让同一相关轨迹两侧造成虚假的独立性。
- 观测不确定性应先保存和用于 QC/误差分析；当前推断器不直接接收逐观测已知误差数组，若需异方差观测模型，应另行扩展。
- 将参与更新的观测和独立评分观测分开。无法排除被 IAP/ORAS5 使用过的观测时，标为同化关系未知或产品一致性验证，不能称严格独立验证。

## 冰况

使用 [NOAA/NSIDC G02202 海冰浓度 CDR](https://nsidc.org/data/g02202/versions/6)，目前页面为 Version 6。产品提供南北极日/月海冰浓度，25 km 极地投影网格，时间自 1978 年起。下载数值 NetCDF，不使用冰范围总量或图片替代节点处浓度；保存版本、来源、实际覆盖月份和引用信息。

- 月度重构实验先配月度浓度；真实任务的安全约束依据任务日期/窗口数据及可用预测，不只用月平均。
- 按文件 CRS 转换到目标节点，处理陆地、海岸、缺测、极区观测空洞等标记。只将已识别的物理浓度转为 `[0,1]`，不能将编码标记裁剪成合法浓度。
- 生成每目标时段的 `ice_fraction`，形状 `(N,)` 或全时段 `(T,N)`，保留 mask 和来源。
- 冰况未知的节点排除；实时规划不能使用决策时尚未得到的未来冰况。历史完整任务窗口统计只能作为 retrospective 场景。
- 当前项目以浓度高于阈值为不可行。冰系设备的可部署政策可能不同，不能把这一规则直接视为所有冰下系统的安全模型。

## 工程数据

气候数据库无法提供真实设备网络参数。需要设备规格、任务计划、链路测量、预算和维护记录；尚无实物系统时可以做明确标为 scenario 的参数化实验，并进行敏感性分析。

| 参数/表 | 准备依据和单位 |
|---|---|
| 候选节点、初始 seed、anchor、gateway | 位置和角色；经纬度顺序 `[lat,lon]`，经度转至 [-180,180] |
| max_sensors | 实际可部署设备数 |
| ice_threshold | 所用平台在声明任务条件下的允许政策 |
| communication_radius_km | 对应通信介质、链路预算或实测；不能随意放大来获得连通 |
| mission_seconds | 与能耗估算一致的整个任务窗口 |
| sense_power_w, process_power_w | 规格与 duty cycle 后的等效功率，W |
| packets_per_sensor | 同一任务窗的消息数，应关联载荷大小和采样频率 |
| tx_electronics_j_per_packet | 每包发射基础能耗，J/packet |
| tx_amplifier_j_per_packet_km2 | 当前代码的距离平方近似参数；从数据拟合并记录有效范围 |
| energy_budget_j, gateway_energy_j | 网络总任务能耗预算和固定网关能耗，J |
| base_cost, maintenance_cost, failure_risk, reliability | 每节点成本、故障风险和历史可靠性；成本统一货币/尺度，风险定义与时间窗一致 |
| parameter_source, source_type, uncertainty | 每个参数来源、measured/specification/assumed、误差或取值范围 |

初始 seed 必须按实际几何位置、冰况和参数通过可行性检查。当前能耗模型未包含 receive/idle，单一半径与距离平方传输也不代表任意水声/卫星网络；若硬件与模型不符，需要扩展模型后才能作真实部署结论。1° 网格南北相邻约百公里，原 demo 的 22 km 不能直接照搬；把场插值到细网格并不会增加物理信息或自动解决链路问题。

## 加载流程的实现顺序

建议使用 xarray/netCDF4 读取，按产品需要引入重网格、投影和 TEOS-10 工具。以下是待实现流程；现有 `audit_local_dataset.py` 仅负责检查。

1. **输入 manifest**：HF/LF/obs/ice 文件清单、URL/DOI、版本、下载日期、hash、变量名、温度/盐度定义、坐标/深度、年月日历、单位、掩膜、配准和 split 政策。凭据留在下载客户端配置，不能写入 manifest。
2. **标准化解码**：IAP YYYYMM、其他产品 CF/calendar、填充值和 QC、纬经度/CRS/深度；保留未改写原始文件。
3. **定义 split 与有效域**：区域及静态海洋域提前声明；HF 训练缺测和 LF 训练支持共同确定固定域，禁止用测试真值优化域、基底或参数。
4. **物理对齐**：温度定义、气候基准、单位；垂向插值和共同网格；禁止跨陆地和无依据外推。
5. **按月配准与观测处理**：生成一份节点映射，HF/LF/ice 共用；生成独立观测长表和 sampling mapping。
6. **训练 artifact**：仅 HF 训练段拟合 mean、scale、POD basis、covariance factor；LF 校准仅使用允许的训练/验证数据。
7. **保存输入和审计**：HF/LF `(T,N)`，坐标 `(N,2)`，冰况 `(T,N)`，观测长表，split，训练 artifact，插值权重、缺测/丢弃计数和原因。
8. **先重构验收**：有限值/维度/节点一致性、随机原始值回查、配准误差图；在独立点观测上比较 LF-only、POD/单保真与多保真表现。失败时检查保真关系、单位和先验。
9. **后运行闭环**：验证 seed、候选观测可用性和网络可行性；冻结每次评分时的环境，单点 acquire 后完整 refit，记录日志。

首批试运行应先准备一个小区域、一个年份或少数目标月份的 LF/冰况及对应观测。先确认格式和观测覆盖，再扩展全部 316 个月。CDS/其他平台如需账户授权，应由用户完成登录并接受数据条款，程序使用当地客户端凭据配置。
