# 2013年极地验证数据补齐与审计

用途：只用于2013年验证、选择参数；训练固定2000–2012，2014禁止调参。原2014结果属于探索性测试，已查看过；严格确认性评价需另保留未查看时段。

已下载24个ORAS5月文件（12温度+12配套实用盐度），12个G02202 V6 F17北极月海冰文件，另含tmask和海冰辅助坐标/掩膜。URL、下载时间、字节和SHA256见downloads.json，逐文件变量/时间/空间检查见audit_report.json。

ORAS5采用ICDC r1x1/opa0，并非原生0.25°网格或完整集合；位温℃、实用盐度psu，先掩膜再做TEOS-10原位温度转换，插值至50 m，周期双线性映射至固定IAP格点。海冰EPSG:3411、25 km，CF自动解包0.01缩放得到0–1浓度，按辅助经纬度球面最近邻映射，最大40 km；F17极区观测空洞、陆地、海岸及缺测保持NaN。原QA保留。

时间按日历月份配对，不要求ORAS代表日与海冰时间戳相等；海冰起止覆盖逐月核验，ORAS的月平均产品定义与CF代表时间共同核验。无ORAS原生时间界限变量时，不把推定日历月起止声称为文件内实测时间界限。

异常、中心化、尺度、POD及协方差全部复用原训练产物，副本字节一致；未重新训练，也未根据验证或测试修改共同域。输出仅包含2013验证字段和2000–2012训练产物，不导出2014场。

| 区域 | 验证HF/LF矩阵（月份×格点） | 每月海冰已知格点 |
|---|---|---|
| atlantic | [12, 346] | [346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346, 346] |
| central_arctic | [12, 3305] | [2942, 2942, 2942, 2942, 2942, 2942, 2942, 2942, 2942, 2942, 2942, 2942] |

文件：aligned_50m/各区域/validation_fields.npz保存HF、LF、海冰、异常、模型数组、固定坐标和源网格权重；training_artifacts.npz为冻结训练副本。

尚未进行参数搜索或部署训练。盐度用于物理转换，不是新增的盐度重构目标。此包未补齐2013独立ITP/Argo观测。

来源：[ICDC ORAS5目录](https://icdc.cen.uni-hamburg.de/thredds/catalog/ftpthredds/EASYInit/oras5/catalog.html)、[ORAS5掩膜修正说明](https://icdc.cen.uni-hamburg.de/thredds/fileServer/ftpthredds/EASYInit/oras5/r1x1/Correction_to_ORAS5_r1x1_files.pdf)、[NSIDC G02202 V6](https://nsidc.org/data/g02202/versions/6)。
