# 用户执行说明：先测试和小试，后确认完整批量

当前只完成实现、协议和测试代码，尚未运行本轮测试或性能实验；不宣称测试通过或算法改善。按用户要求，最终中文交付报告在收到实际结果后生成。使用项目既有`.venv`，不要升级环境。所有命令从`D:\project\RBMF-CS`执行；不要运行旧的`run_tests.py`，它会覆盖旧全局测试结果。

## 1. 独立参考、合成控制和原项目回归测试

```powershell
Set-Location D:\project\RBMF-CS
.venv\Scripts\python scripts/run_clbr_p1_tests.py --output outputs/clbr_p1_tests_v1
```

阅读`outputs/clbr_p1_tests_v1/tests.json`中的`passed`。仅当为true时继续下一步。`tests.log`是完整日志；失败时把`tests.json`及失败日志给我，保留目录，修复后用v2新目录。测试包括新模型独立增广SVD/归一化密度/一维积分、场与未来观测抽样、单位、缺测/重复观测、折污染和I/O门禁；也运行原全部回归测试。Windows临时目录权限问题请保留原报错，不以跳过测试取得通过。

## 2. 一个外层季度的有限小试

```powershell
.venv\Scripts\python scripts/run_clbr_p1.py --phase pilot --tests outputs/clbr_p1_tests_v1/tests.json --output outputs/clbr_p1_pilot_v1
```

小试外层留出1–3月；内层仅4–12月，三个季度轮流留出。双场景、seed1001、4/12节点。新模型3个尺度候选、严格折外Ridge6候选，外层10种点预测和3种区间方法。不依据小试精度追加候选。300秒、250MiB按案例间检查点停止，允许最后一个案例及保存开销；捕获到的失败保留status/failures和已有预测，手动强制终止可能只留下部分NPZ。源码改变会使旧测试证明失效，必须重跑测试。

## 3. 独立核验落盘指标、选参及折隔离

```powershell
.venv\Scripts\python scripts/verify_clbr_p1.py --run outputs/clbr_p1_pilot_v1
```

核验不重跑候选模型；从保存预测独立计算RMSE/MAE、覆盖、宽度、区间评分、WIS，重算内层选参及经验分位数，并检查节点/观测/未来噪声/折/源码/原算法/训练与封存状态。`verification.json`不可覆盖；失败后保留结果并使用新版本实验目录。

## 4. 回传这些文件后再决定全年批量

- `outputs/clbr_p1_tests_v1/tests.json`
- `outputs/clbr_p1_pilot_v1/resources.json`
- `outputs/clbr_p1_pilot_v1/verification.json`
- 若失败：`status.json`、`failures.json`和终端报错/测试日志。

如果文件仍在同一项目目录，你直接告诉我“测试和小试已完成，读取结果”即可，不必复制全部CSV。届时先核查候选数、实测时间/内存/磁盘及保守全年估算，再询问全年资源上限。计划全年4折×12月合计×双场景×3种子×5预算，尚未获本轮资源确认；不要现在运行full。

`resources.json`的全年倍数30是按4折、3种子、5/2预算算的工作量上界，再乘2作时间/磁盘保守系数；内存估计3倍另列。实际先验拟合等固定开销不会线性增长，此估计只用于资源安排，不能当实测。

完整入口已实现，但要求后续批准JSON（user_confirmed、source_sha256、与configs一致的plan、pilot_verification路径及limits.wall_limit_seconds/disk_limit_mib）。该文件须在收到用户确认后才创建；独立测试从不在授权范围内。后续运行命令会连同批准文件一起给出。

## 输出与边界

每次run包含source.zip、freeze.json、fold_manifest.json、inner_metrics.csv、point_metrics.csv、interval_metrics.csv、responses.json、inner/outer预测NPZ、resources/status/failures。逐案例表保留全部月份/预算/场景/种子，后续汇总先种子后月/预算。折内校准参数、残差统计、候选得分、区间尺度均可追溯。

后验与确定性残差点均值在固定未采样评分域应相同，贝叶斯不会被计为第二份点收益；去LF均值消融仍继承完整模型的残差统计，因此不是独立调优无LF基线。严格折外Ridge另列以供公平比较。原冻结方法具有2013全年选参暴露，且整条新路线也是反复开发所得，所有结果仍是开发证据。

没有运行独立测试、下载数据或更改封存。完整模型是否值得冻结、点精度与不确定性是否分别通过，均等待实际结果后判断。
