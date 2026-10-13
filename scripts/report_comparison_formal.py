"""Render the completed, validation-selected comparison from saved records only."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(root, figures, tuning, output):
    cfg, manifest = load(root/'config.json'), load(root/'manifest.json')
    selection = load(root/'selection.json')
    verification = load(root/'verification.json')
    ledger = load(root/'ledger.json')
    rows = list(csv.DictReader((figures/'summary.csv').open(encoding='utf-8-sig')))
    pairs = load(figures/'paired_statistics.json')
    scores, planning = load(tuning/'scores.json'), load(tuning/'planning_scores.json')
    settings = cfg['comparison']
    budget = max(settings['budgets'])
    def fmt(x):
        return '不可用' if x is None or x == '' else f'{float(x):.6f}'
    def get(branch, method, scenario='clean'):
        return next(r for r in rows if r['branch'] == branch and r['method'] == method and r['scenario'] == scenario and int(r['requested_nodes']) == budget)
    def interval(value):
        return '不可用' if value is None else '['+', '.join(f'{v:.6g}' for v in value)+']'
    clean_rbmf, clean_lf = get('reconstruction', 'rbmf'), get('reconstruction', 'lf_only')
    clean_eir = get('deployment', 'mc_eir')
    lines = ['# 验证集选择后的 RBMF-CS / MC-EIR 探索性比较', '',
        f"运行状态：{manifest['status']}。完成预注册批量执行不等于获得独立测试证据。2014 年已用于诊断和开发，本轮仍为探索性评估；没有未见的独立配对测试年份。", '',
        f"核心结果：无污染、{budget}节点时 RBMF-CS 面积 RMSE={fmt(clean_rbmf['rmse_c'])}℃，优于 LF-only 的 {fmt(clean_lf['rmse_c'])}℃，但没有优于同节点和观测下的全部 POD 基线。MC-EIR RMSE={fmt(clean_eir['rmse_c'])}℃；各部署方法的误差差异很小，本轮未支持其误差优势。优势、失败和适用边界须按下述配对结果、污染场景及约束共同解释。", '',
        '## 数据、规模与隔离', '',
        '2000–2012 训练、2013 全年验证、2014 四个已有月份探索性评估。2013 补充 LF/海冰通过独立输入审计；训练工件、HF 和网格保持不变。调参工作进程输入副本不含 2014 参考场和 LF。原始数据及历史结果均保留。', '',
        '本实验以 IAP 格点温度作为 HF 参考场、ORAS5 作为 LF 候选，从参考格点模拟传感器采样并加入预先生成的污染；不是独立 Argo/ITP 点观测的野外验证。HF/LF 是建模约定，尚未证明 IAP 在全部目标位置精度高于 ORAS5；产品内部同化相关性也不能由项目时间划分消除。误差与覆盖率均相对于 IAP 参考场解释。', '',
        f"种子 {settings['seeds']}；月份 {settings['dates']}；预算 {settings['budgets']}；场景 {settings['scenarios']}。共 {len(ledger)} 个案例。状态计数：{dict(Counter(e['status'] for e in ledger))}。进程异常记录数 {sum(e.get('method_failures', 0) for e in ledger)}；非收敛、提前停止另在逐预算结果中保留。", '',
        '同一案例全方法共享按节点预生成的扰动，观测器只公开已选节点。重构比较使用相同约束 Maximin 节点；部署比较固定 RBMF-CS。固定评价格点不采样，面积权重 cos(latitude)，NRMSE 分母为该域训练异常的加权 RMS。数据准备、参数空间、公式和来源见 comparison_protocol.md 及 2013 补充协议。', '',
        '高斯标准差 0.02℃；t(3) 按等方差缩放；离群、缺测、LF 偏移和训练污染独立。仅考察协议固定污染尺度，没有依据结果增加强度以制造优势。LF +0.2℃ 是固定方向的敏感性分析，可能抵消原有负偏差。训练污染改变 POD/covariance 输入，保持原气候态和尺度，不能推广为所有预处理均污染。部署的方差下降采用冻结超参数的高斯近似，没有对未来完整 VB 重拟合做精确积分。', '',
        '## 验证选择', '',
        '| 方法 | 搜索数 | 可用候选 | 选定参数 |', '|---|---:|---:|---|']
    for method in sorted({s['method'] for s in scores}):
        group = [s for s in scores if s['method'] == method]
        selected = {'rank': selection['comparison']['method_rank'][method],
                    'inference': selection['comparison']['method_inference'][method]}
        if method in ('pod_ridge', 'pod_lasso', 'deterministic_mf'):
            selected['lambda'] = selection['comparison']['regularization'][method]
        lines.append(f"| {method} | {len(group)} | {sum(s['score'] is not None for s in group)} | `{json.dumps(selected)}` |")
    lines += [f"| MC-EIR | {len(planning)} | {sum(s['score'] is not None for s in planning)} | `{json.dumps(selection['planning'])}` |", '',
        '全部候选记录均保留。验证目标为 12 月 × 3 配对种子 × 3 节点预算的平均面积 RMSE，任何未收敛/失败使候选不可选。消融继承完整方法的验证参数，不利用探索性评估调参。', '',
        f'## {budget} 节点比较', '',
        '下表误差仅汇总收敛且实际达到该预算的案例；成功数/计划数必须一起阅读。未达到预算的结果没有并入同预算误差。CSV 另含 MAE、NRMSE、实际节点、能耗、成本、可行率和分项运行时间。']
    for branch in ['reconstruction', 'deployment']:
        lines += ['', '### '+branch, '', '| 方法 | clean RMSE ℃ | Gaussian RMSE ℃ | clean 收敛同预算/计划 | clean 可行率 |', '|---|---:|---:|---:|---:|']
        for method in settings[branch]:
            a, b = get(branch, method), get(branch, method, 'gaussian')
            lines.append(f"| {method} | {fmt(a['rmse_c'])} | {fmt(b['rmse_c'])} | {a['converged_same_count_cases']}/{a['planned_cases']} | {fmt(a['feasible_rate_of_planned'])} |")
    lines += ['', '### 固定污染尺度下的比较', '',
        f'以下为 {budget} 节点的面积 RMSE（℃）；训练污染与在线污染独立，没有叠加。完整方法、基线和消融均用保存结果汇总。', '',
        '| 场景 | RBMF | 去鲁棒 | 去多保真 | POD-LASSO | Gaussian-SBL | MC-EIR |', '|---|---:|---:|---:|---:|---:|---:|']
    for scenario in settings['scenarios']:
        values = [fmt(get('reconstruction', m, scenario)['rmse_c']) for m in ['rbmf', 'no_robust', 'no_mf', 'pod_lasso', 'gaussian_sbl_vb']]
        values += [fmt(get('deployment', 'mc_eir', scenario)['rmse_c'])]
        lines.append('| '+scenario+' | '+' | '.join(values)+' |')
    lines += ['', '## 配对差值与适用边界', '',
        '差值为表中方法减完整参考方法（重构 RBMF、部署 MC-EIR），负数有利于表中方法。以下是 clean 最大预算；全部场景和预算见 paired_statistics.csv/json。只使用四个月均有可用配对的种子，同时报告分母。', '',
        '| 分支/方法 | 配对平均差 ℃ | 固定环境种子条件95%区间 | 时间/空间分块95%敏感性区间 | 配对种子/计划 |', '|---|---:|---|---|---:|']
    for p in pairs:
        if p['scenario'] == 'clean' and p['budget'] == budget:
            lines.append(f"| {p['branch']}/{p['method']} | {fmt(p['paired_mean_difference_c'])} | {interval(p.get('seed_conditional_ci95_c'))} | {interval(p.get('time_space_block_sensitivity95_c'))} | {len(p['paired_seeds'])}/{p['planned_seeds']} |")
    for branch, reference in [('reconstruction', 'rbmf'), ('deployment', 'mc_eir')]:
        valid = [p for p in pairs if p['branch'] == branch and p['scenario'] == 'clean' and p['budget'] == budget and p['paired_mean_difference_c'] is not None and not p['method'].startswith('no_')]
        better = [p['method'] for p in valid if p['paired_mean_difference_c'] < 0]
        lines += ['', f"在 clean 最大预算的可用配对结果中，平均误差低于 {reference} 的基线为：{', '.join(better) if better else '未观察到'}。这是当前固定环境下的描述，不是跨年份统计优越性声明。"]
    diagnosis_path = root/'saved_diagnosis.json'
    if diagnosis_path.exists():
        diagnosis = load(diagnosis_path)
        records = [r for month in diagnosis['months'].values() for r in month.get('methods', {}).values() if 'steps' in r]
        endpoints = [r['steps'][-1] for r in records]
        if endpoints and all('lf_scale_rho' in e for e in endpoints):
            ratios = [e['observation_to_lf_precision_trace'] for e in endpoints]
            rho = [e['lf_scale_rho'] for e in endpoints]
            ard = [e['ard_precision_trace']/e['lf_precision_trace'] for e in endpoints]
            changes = [r['seed_to_final_prediction_rms_c'] for r in records]
            lines += ['', '### 保存后验的机制诊断', '',
                f"按固定规则选取 clean、最小种子、全部月份，Random/Maximin/MC-EIR 最后一步的观测/LF兼容精度迹比为 {min(ratios):.6g}–{max(ratios):.6g}，ARD/LF兼容精度迹比为 {min(ard):.6g}–{max(ard):.6g}，LF缩放 rho 为 {min(rho):.6g}–{max(rho):.6g}；从初始节点到最终节点的预测场 RMS 变化为 {min(changes):.6g}–{max(changes):.6g}℃。这些诊断不计入算法计时。", '',
                '精度矩阵的迹比较仅反映总体尺度，不能解释为每个方向相同的影响比例。兼容项大也不意味着预测紧跟原始 LF：它约束的是 rho 缩放后的 LF；rho 接近零时可与 ARD 共同将异常系数压向零，使物理预测接近训练气候态。应结合保存的两项右端向量、局部权重和后验有效自由度解释，不能只用选点不同或 LF 精度一个因素归因。详细逐步数据与节点 Jaccard 见 saved_diagnosis.json。']
    lines += ['', '种子不是独立环境样本。空间按2°×5°分块、时间按相邻两个观测月份循环分块；只有四个相关月份，区间仅作探索性敏感性分析。独立环境 p 值和 Holm 校正 p 值留空；没有制造多方法显著性。', '',
        '无污染场景的确定性重构与确定性部署在各种子上相同，因此种子条件区间可以退化为零宽度；这不代表环境不确定性为零。Random 的选点仍随种子变化。', '',
        '## 失败、约束代价与消融', '',
        '| 分支/方法/场景 | 预算 | 收敛同预算/计划 |', '|---|---:|---:|']
    incomplete = [r for r in rows if int(r['converged_same_count_cases']) < int(r['planned_cases'])]
    for r in incomplete:
        lines.append(f"| {r['branch']}/{r['method']}/{r['scenario']} | {r['requested_nodes']} | {r['converged_same_count_cases']}/{r['planned_cases']} |")
    if not incomplete:
        lines.append('| 无 | — | 全部达到预算并收敛 |')
    lines += ['', '原始状态、错误及未收敛诊断见 metrics.csv、failures.json、运行 ledger 和逐案例记录。POD-QR 残差为零时停止，不将数值舍入误差当作有效 pivot；超过所选 POD 秩的预算可能不可用。熵权 QR 使用论文负指示器信息权重乘残差范数，属于工程约束适配，未声称原论文严格复现。', '',
        '| 约束代价方法 | clean RMSE ℃ | 可行率（按原约束审计） | 成本 | 能耗 J |', '|---|---:|---:|---:|---:|']
    for method in ['pod_qr', 'unconstrained_pod_qr', 'mc_eir', 'unconstrained_mc_eir']:
        r = get('deployment', method)
        lines.append(f"| {method} | {fmt(r['rmse_c'])} | {fmt(r['feasible_rate_of_planned'])} | {fmt(r['cost'])} | {fmt(r['mission_energy_j'])} |")
    lines += ['', '无约束版本单列，不纳入主比较。去多保真删除兼容因子及相应更新/ELBO 项；去鲁棒似然固定局部精度为1并删除局部 Gamma 项。四个部署消融删除对应评分项，仍保留全部硬约束。消融效果不预设为正。工程成本/能耗参数是场景假设，不是实测设备验证。', '',
        '按原约束不可行时，原网络无法给出声明条件下的可执行任务能耗，保留不可用，不用放宽通信后的能耗代替。', '',
        '## 不确定性', '', '| 方法 | Gaussian 潜在场覆盖 | 潜在场宽度 ℃ | 未来观测覆盖 | 未来观测宽度 ℃ |', '|---|---:|---:|---:|---:|']
    for method in ['rbmf', 'gaussian_sbl_vb', 'no_robust', 'no_mf']:
        r = get('reconstruction', method, 'gaussian')
        lines.append(f"| {method} | {fmt(r['latent_coverage95'])} | {fmt(r['latent_width95_c'])} | {fmt(r['future_coverage95'])} | {fmt(r['future_width95_c'])} |")
    if selection['planning']['eta'] == 0:
        lines += ['', '验证选择的冗余惩罚系数 eta=0，完整方法的选点评分已不含冗余惩罚。因此去冗余消融与完整方法评分相同，本轮不能用这一相同结果判断非零冗余惩罚在其他环境中的贡献。']
    lines += ['', '潜在场区间来自系数后验；未来观测区间另积分噪声精度和新局部精度，用独立预生成未来观测评价。覆盖率和宽度必须一起解释。遗漏基底学习、截断、模型差异和 mean-field 交叉协方差，不能凭较宽的未来观测区间声称潜在场校准良好。没有使用探索性真值校准区间。', '',
        '## 验收、运行时间与复现', '',
        f"保存结果核验：passed={verification['passed']}，{verification['checks']} 项检查，{verification['recomputed_predictions']} 份预测重新计算指标。固定种子重跑证据见选定参数 pilot 的 repeat_verification.json；单元参考测试见测试日志。正式并发数 {manifest['workers']}，批次墙钟 {manifest['wall_seconds']:.2f} 秒，案例耗时累计 {manifest.get('sum_case_seconds', manifest['wall_seconds']):.2f} 秒。预处理、选点、重构时间分开记录；并发和审计 I/O 会影响总墙钟，不据此声称算法固有速度优势。", '',
        '计时范围：部署表的 reconstruction_seconds 累计模型内部 VB 更新时间，selection_seconds 包含规划器准备、可行性筛选和评分；重构表的 reconstruction_seconds 是独立求解器外层计时，其观测节点作为共同输入，因此 selection_seconds 为0。不同分支的时间不作直接速度排名。区间抽样、落盘、额外机制诊断和绘图不在这些算法计时中，批次总墙钟另含审计 I/O。', '',
        f"运行目录：`{root.as_posix()}`；验证选择：`{tuning.as_posix()}`；图表：`{figures.as_posix()}`。源码、输入副本、版本、种子、逐步预测/后验/迭代/约束均归档，部署大日志为 `.json.gz`。", '',
        '重绘不拟合模型，不改数据、不平滑、不强制单调。空间图按最早月份、最小种子、clean、最大计划预算固定选择，使用共同色标。', '', '```powershell',
        '.venv\\Scripts\\python -m unittest discover -s tests -v',
        '.venv\\Scripts\\python scripts/tune_comparison_batch.py --config configs/comparison_2013_v1.json --output outputs/validation_new --workers 6',
        '.venv\\Scripts\\python scripts/verify_comparison_tuning.py outputs/validation_new',
        '.venv\\Scripts\\python scripts/run_comparison.py --mode pilot --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --full-pilot-budgets --workers 6 --output outputs/selected_pilot_new',
        '.venv\\Scripts\\python scripts/verify_comparison.py outputs/selected_pilot_new',
        '.venv\\Scripts\\python scripts/run_comparison.py --mode pilot --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --full-pilot-budgets --workers 6 --output outputs/selected_repeat_new',
        '.venv\\Scripts\\python scripts/verify_comparison.py outputs/selected_pilot_new --repeat outputs/selected_repeat_new',
        '.venv\\Scripts\\python scripts/freeze_comparison_scale.py outputs/selected_pilot_new --output outputs/scale_new.json',
        '.venv\\Scripts\\python scripts/run_comparison.py --mode formal --config configs/comparison_2013_v1.json --selection outputs/validation_new/selection.json --pilot-verification outputs/selected_pilot_new/repeat_verification.json --workers 6 --output outputs/formal_new',
        '.venv\\Scripts\\python scripts/verify_comparison.py outputs/formal_new', '```', '',
        '只核验或重绘本次已有结果，可单独使用以下命令：', '', '```powershell',
        f'.venv\\Scripts\\python scripts/verify_comparison.py {root.as_posix()}',
        f'.venv\\Scripts\\python scripts/plot_comparison.py {root.as_posix()} --output outputs/figures_new',
        f'.venv\\Scripts\\python scripts/report_comparison_formal.py {root.as_posix()} --figures outputs/figures_new --tuning {tuning.as_posix()} --output docs/report_new.md', '```', '',
        '以上重绘和报告命令可单独执行，无需重跑参数搜索或案例。输出目录必须不存在。复现固定数值允许绝对误差1e-12，运行时间不要求一致；归档 runtime 提供当次原始源码。历史诊断与未调参 pilot 见 comparison_report.md，不能与本次已验证参数结果混为一组。', '',
        '图表阅读提示：部署曲线与部署鲁棒性热图放大了数值轴/色轴，差异在很小的℃尺度；应阅读刻度和配对区间，不能从视觉分离推断实用显著优势。稳健似然、LF兼容和评分各项的结果仅对应当前验证参数、固定污染强度、区域和月份。当前数据不支持普遍优越性或跨年份泛化结论。']
    artifacts = root/'artifact_verification.json'
    visual = root/'visual_review.json'
    if artifacts.exists() and visual.exists():
        check, review = load(artifacts), load(visual)
        lines += ['', '## 最终图表与导出验收', '',
            f"导出与汇总独立核验 passed={check['passed']}，{check['checks']} 项检查；包含 {check['requested_budget_metric_rows']} 条预算指标、{check['summary_rows']} 行汇总和 {check['paired_comparisons']} 项配对比较。九类 PDF 已逐张渲染并检查文字、图例、共同色标及节点标记；视觉检查 passed={review['passed']}。PNG 为300dpi，PDF/SVG及CSV全部保留。验收记录为 artifact_verification.json、visual_review.json 和 acceptance.json。"]
    output.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--figures', required=True, type=Path)
    p.add_argument('--tuning', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    run(a.root, a.figures, a.tuning, a.output)
