"""Generate a candid Chinese report from saved artifacts, without refitting."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(root, figures, diagnosis, output):
    config, manifest = load(root/'config.json'), load(root/'manifest.json')
    verify, scale = load(root/'verification.json'), load(root/'scale_estimate.json')
    historical = load(diagnosis/'diagnosis.json')
    rows = list(csv.DictReader((figures/'summary.csv').open(encoding='utf-8-sig')))
    ledger = load(root/'ledger.json')
    realized = np.load(next((root/'cases').glob('*/realization.npz')))
    budget = max(config['comparison']['budgets'])
    def row(branch, method, scenario='clean'):
        return next(r for r in rows if r['branch'] == branch and r['method'] == method
                    and r['scenario'] == scenario and int(r['requested_nodes']) == budget)
    def value(branch, method, key, scenario='clean'):
        return float(row(branch, method, scenario)[key])
    pairs = [p for month in historical['months'].values() for p in month['pairs']]
    histories = [r for month in historical['months'].values() for r in month['methods'].values()]
    trace_ratio = [r['steps'][-1]['observation_to_lf_precision_trace'] for r in histories]
    delta = [r['seed_to_final_prediction_rms_c'] for r in histories]
    lm = [s['lambda_min'] for r in histories for s in r['steps']]
    lx = [s['lambda_max'] for r in histories for s in r['steps']]
    reason_counts = {}
    for r in histories:
        for key, count in r['constraint_reasons'].items():
            reason_counts[key] = reason_counts.get(key, 0)+count
    tables = []
    for branch, methods in [('reconstruction', config['comparison']['reconstruction']),
                            ('deployment', config['comparison']['deployment'])]:
        table = ['| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 可行率 |', '|---|---:|---:|---:|']
        for method in methods:
            clean, gaussian = row(branch, method), row(branch, method, 'gaussian')
            table.append(f"| {method} | {float(clean['rmse_c']):.8f} | {float(gaussian['rmse_c']):.8f} | {float(clean['feasible_rate_of_planned']):.1%} |")
        tables.append('\n'.join(table))
    text = f'''# RBMF-CS / MC-EIR 比较：诊断、实现与探索性 pilot

状态：**完成框架、小规模端到端验证和 pilot；正式验证调参与 30-seed 批量未执行。** 缺少真实 2013 验证 LF/海冰，不能以 2014 测试代替。用户已确认仅有 2014 四个月 LF。不存在可供本协议使用的未见配对测试环境。以下数字由保存结果生成，不是正式优越性证据。

## 历史结果诊断

历史三方法节点集合 Jaccard 范围 {min(p['jaccard'] for p in pairs):.4f}–{max(p['jaccard'] for p in pairs):.4f}，说明确实选择了不同位置。4→12 节点预测场 RMS 变化仅 {min(delta):.8f}–{max(delta):.8f} ℃。最终观测精度项 / LF 兼容精度项的矩阵迹比为 {min(trace_ratio):.6g}–{max(trace_ratio):.6g}。这些是模型内各项相对大小的诊断，不能把 trace 比解释成逐方向相同的影响比例。

POD 名义秩 {historical['pod']['rank']}，训练谱参与率有效秩 {historical['pod']['participation_rank']:.3f}，累计90%能量需 {historical['pod']['modes_90_percent']} 个模态，保留能量 {historical['pod']['energy_retained']:.4%}。局部鲁棒权重全程范围 {min(lm):.4f}–{max(lx):.4f}，部分观测被降权，但不是所有权重为零。约束排除事件累计（候选/轮次可重复且原因可重叠）：{reason_counts}。历史所有方法最终都达到12节点，约束影响候选域但没有使三个节点集合相同。

结论：当前 LF 兼容项主导、稀疏观测响应很弱，是误差接近的直接数值证据；选点集合相同不是原因。不能从该现象推断所有部署策略等效，也不能凭测试诊断选择更有利的先验。本次未改变继承的 RBMF 先验来制造优势。细节（逐步精度、RHS、观测有效自由度、局部权重、后验参与率）见 `{diagnosis.as_posix()}/diagnosis.json`。

## 本次范围与公平性

保持训练2000–2012、验证2013。2014全年保守标为探索性；现有四月已用于开发诊断。pilot：{len(config['comparison']['seeds'])} 配对种子，月份 {config['comparison']['dates']}，预算 {config['comparison']['budgets']}，{len(config['comparison']['scenarios'])} 个独立污染场景，共 {len(ledger)} 个 case。固定 {len(realized['evaluation'])} 个评价格点、{len(realized['candidates'])} 个候选格点，评价格点不参与选点；所有方法共享观测实现，重构比较共享 Maximin 节点。

Gaussian 与 t(3) 采用相同方差0.0004℃²（标准差0.02℃）；离群、缺测和 LF +0.2℃ 偏移分开；训练污染四类独立于在线污染。正向 LF 偏移只是预设敏感性方向，可能减小原有负偏差，不应预设为必然恶化。训练污染只改变用于 POD/covariance 的矩阵，原气候态/归一化不变，不能概括为所有预处理均被污染。

所有参数为未调优继承值（正则λ=0.1），图中的差异不能替代验证选择后的公平结论。部署统一四类工程约束；无约束附加试验独立标记，并以原约束审计。POD-QR 和熵权 POD-QR 都是适配版本；熵权方向和乘法已核对原论文式6-1至6-5、算法6-1及第118页。基线与变分 SBL 的公式/来源见 `docs/comparison_protocol.md`。

## {budget}节点结果（同一固定评价格点、两种子均值）

重构比较：

{tables[0]}

部署比较（固定 RBMF-CS）：

{tables[1]}

本次 clean 的 RBMF RMSE 为 {value('reconstruction','rbmf','rmse_c'):.8f} ℃，LF-only 为 {value('reconstruction','lf_only','rmse_c'):.8f} ℃；RBMF 没有优于 LF-only。MC-EIR 为 {value('deployment','mc_eir','rmse_c'):.8f} ℃、Random 为 {value('deployment','random','rmse_c'):.8f} ℃、Maximin 为 {value('deployment','maximin','rmse_c'):.8f} ℃；未观察到 MC-EIR 的误差优势。这里没有任何跨年份统计优越性声明。去多保真必须删除 compatibility 因子，本实现及独立测试已验证，并非输入零 LF。

## 校准、统计与约束代价

Gaussian场景下 RBMF 的 latent 95% 覆盖为 {value('reconstruction','rbmf','latent_coverage95','gaussian'):.2%}，平均区间宽度 {value('reconstruction','rbmf','latent_width95_c','gaussian'):.4f} ℃；future-observation 覆盖为 {value('reconstruction','rbmf','future_coverage95','gaussian'):.2%}，宽度 {value('reconstruction','rbmf','future_width95_c','gaussian'):.4f} ℃。未来区间很宽不能证明潜在场不确定性校准良好。方差遗漏基底学习、POD截断、模型差异和 mean-field 交叉协方差；没有测试真值校准。

配对差值与 seed 条件 bootstrap CI 见 `paired_statistics.csv/json`。只有一个月份，时间/空间联合块区间和独立环境 p 值均留空；不能把两个噪声种子视为两个环境。实现了用于多月探索性敏感性的两观测月循环块 + 2°×5°空间块方法，并独立测试 Holm 校正函数，但当前不制造缺乏独立环境支持的 p 值。条件 CI 仅描述固定环境下随机扰动/Random变化，两个种子很少，不能外推。

无约束 QR 的 clean 原约束可行率为 {value('deployment','unconstrained_pod_qr','feasible_rate_of_planned'):.1%}；无约束 MC-EIR 为 {value('deployment','unconstrained_mc_eir','feasible_rate_of_planned'):.1%}。断连时原网络任务能耗为不可定义（null），不记作零。不同实际节点数不混入同预算误差均值，失败/缺项在 CSV 和失败 ledger 保留。

## 验收、耗时与保留的失败

本次保存 {verify['recomputed_predictions']} 份预测，落盘独立复核 {verify['checks']} 项，passed={verify['passed']}。单元/既有回归测试记录见 `outputs/comparison_tests_release_final.txt`。固定种子跨进程重复验证见 `outputs/comparison_repeat_v1/repeat_verification.json`（772数组及无计时的诊断/决策一致），后续全pilot重复结果另见当前运行的repeat_verification.json。重构方法、planner 的选点接口、共享扰动、约束、岭回归独立增广最小二乘、LASSO独立优化、QR独立SciPy pivot、去MF/去鲁棒均已验证。

pilot wall {manifest['wall_seconds']:.2f} s（包括归档与区间诊断I/O）。算法计时另存 preprocess/selection/reconstruction；不把区间抽样、绘图或验证计入选点/拟合。完整30种子×4月份×10场景、4/6/8/10/12预算的粗估 wall 为 {scale['rough_formal_seconds_lower']/3600:.2f}–{scale['rough_formal_seconds_upper']/3600:.2f}小时，不含验证网格搜索；只是按1–4倍每case扩展的估算，不能承诺。正式规模尚未冻结/执行，因为验证输入不完整。

首轮 `comparison_smoke_v1` 的无约束断连结果暴露 energy=null 保存异常；已修复，原异常JSON保留。历史 `initial_weak_prior_run` 未收敛结果、旧 `polar_deployment` 全部保留。当前 pilot case进程失败数 {sum(e['status'] != 'completed' for e in ledger)}，方法异常数 {sum(e.get('method_failures',0) for e in ledger)}；非收敛状态仍单独记录，未删除负面结果。

当前限制：单区域、单深度、一个pilot月份、两随机种子、未完成验证调参、无未见独立测试年；IAP网格传感器模拟不是独立现场观测，产品同化独立性未证实；工程功耗/成本是情景假设，不能声称海上部署验证。验证搜索代码因缺真实输入只测试了阻断路径，其真实2013端到端调参仍待运行。正式多月块统计同样尚无真实正式运行结果。

## 复现与重绘

在 `D:\\project\\RBMF-CS` 中执行，输出目录必须不存在，旧结果不会覆盖：

```powershell
.venv\\Scripts\\python scripts/diagnose_comparison.py --output outputs/diagnosis_new
.venv\\Scripts\\python scripts/run_comparison.py --mode pilot --output outputs/pilot_new
.venv\\Scripts\\python scripts/verify_comparison.py outputs/pilot_new
.venv\\Scripts\\python scripts/verify_comparison.py outputs/pilot_new --repeat {root.as_posix()}
.venv\\Scripts\\python scripts/plot_comparison.py {root.as_posix()} --output outputs/figures_new
```

补充真实2013 LF/海冰后，写到新数据/配置目录，不覆盖旧原始数据。新 aligned_fields.npz 需保留原全部数组，增加与12个月2013日期对齐的 validation_lf_model 和 validation_ice_fraction，物理量转换/网格/训练尺度沿用既有合同，并重新审计来源与对齐。然后：

```powershell
.venv\\Scripts\\python scripts/tune_comparison.py --config configs/new_real_validation.json --output outputs/tuning_new
.venv\\Scripts\\python scripts/run_comparison.py --mode formal --config configs/new_real_validation.json --selection outputs/tuning_new/selection.json --pilot-verification outputs/pilot_new/verification.json --output outputs/formal_new
```

正式运行仍标记为2014探索性评估；不能因调参只使用验证年便恢复2014的未见测试身份。每次运行保留输入副本、输入hash、配置、源码zip、精确软件版本、训练矩阵、扰动、观测、逐轮预测/后验/诊断/可行性。源码快照可在新的目录中解包后用该脚本和保存config重跑；程序不要求 Git 仓库（当前工作区无.git）。

## 图表

所有图只读结果生成，CSV、SVG、PDF、300dpi PNG 位于 `{figures.as_posix()}`。空间展示固定为最早月份、最小种子、clean、最大计划预算，共同色标。未平滑曲线、未强制误差随节点单调下降、未筛选有利seed。
'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding='utf-8')
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--diagnosis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.figures, args.diagnosis, args.output)

