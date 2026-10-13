"""Real-field deployment experiment; grid observations and hardware are scenarios."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rbmf_cs import (RBMFCS, VBConfig, TrainingCovariance, ConstraintConfig,
                    DeploymentConstraints, PlanningConfig, MCEIR, run_closed_loop)
from rbmf_cs.constraints import distances_km


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


class ComparisonPlanner(MCEIR):
    """Reuse feasibility and inference, changing only the selection criterion."""
    def __init__(self,*args,method,coordinates,random_priority,**kwargs):
        super().__init__(*args,**kwargs)
        self.method,self.coordinates,self.random_priority=method,coordinates,random_priority

    def score_candidates(self,model,selected,feasible_candidates):
        scores=super().score_candidates(model,selected,feasible_candidates)
        if self.method=='mc_eir':
            return scores
        candidates=[r['node'] for r in scores['candidates']]
        if self.method=='random':
            priorities=self.random_priority[candidates]+1.
        elif self.method=='space_filling':
            joined=np.vstack((self.coordinates[candidates],self.coordinates[selected]))
            priorities=distances_km(joined,'latlon_degrees')[:len(candidates),len(candidates):].min(axis=1)+1e-12 if candidates else []
        else:
            raise ValueError('Unknown method')
        for row,priority in zip(scores['candidates'],priorities):
            row['mc_eir_diagnostic']=row['eir']
            row['eir']=float(priority)
            row['selection_rule']=self.method
        scores['selection_rule']=self.method
        return scores


def metrics(prediction,truth,selected,weights):
    unseen=np.ones(len(truth),bool); unseen[selected]=False
    error=prediction-truth
    def rmse(mask):
        return float(np.sqrt(np.average(error[mask]**2,weights=weights[mask]))) if mask.any() else None
    return {'rmse_all_c':rmse(np.ones(len(truth),bool)), 'rmse_unobserved_c':rmse(unseen),
            'mae_all_c':float(np.average(abs(error),weights=weights)),
            'bias_c':float(np.average(error,weights=weights))}


def svg_map(coordinates,values,selected,title):
    # Coordinate map, not a coast-line or navigation chart.
    lat,lon=coordinates.T
    vmin,vmax=float(np.min(values)),float(np.max(values))
    circles=[]
    for i,(la,lo,v) in enumerate(zip(lat,lon,values)):
        q=(v-vmin)/max(vmax-vmin,1e-12)
        color=f'rgb({int(255*q)},80,{int(255*(1-q))})'
        x,y=40+(lo+20)/40*520,280-(la-70)/10*240
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"><title>node {i}; {la:.2f}N {lo:.2f}E; {v:.3f} C</title></circle>')
    for j,i in enumerate(selected):
        x,y=40+(lon[i]+20)/40*520,280-(lat[i]-70)/10*240
        circles.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="none" stroke="black" stroke-width="2"/><text x="{x+6:.2f}" y="{y-6:.2f}" font-size="10">{j+1}</text>')
    return f'<h3>{title}</h3><svg viewBox="0 0 600 320"><rect x="40" y="40" width="520" height="240" fill="#eee"/>'+''.join(circles)+f'<text x="40" y="305">20W — 20E; 70N — 80N | {vmin:.2f} to {vmax:.2f} °C | circles: sensors</text></svg>'


def dashboard(output,summary):
    sections=[]
    labels={'mc_eir':'MC-EIR','random':'随机选点','space_filling':'均匀覆盖（最远点）'}
    for month in summary['months']:
        blocks=[]
        rows=summary['months'][month]
        for method,row in rows.items():
            folder=output/month/method
            if not (folder/'final.npz').exists():
                continue
            with np.load(folder/'final.npz') as z:
                blocks.append(svg_map(z['coordinates_latlon'],z['prediction_insitu_c'],z['selected'],f"{labels[method]}：{len(z['selected'])}个传感器"))
            comparison=f"共同未观测区域RMSE：{row['common_unobserved_rmse_c']:.4f} ℃；同域ORAS5：{row['lf_common_unobserved_rmse_c']:.4f} ℃。" if 'common_unobserved_rmse_c' in row else ''
            blocks.append(f"<p>{comparison}全域RMSE：{row['final']['rmse_all_c']:.4f} ℃；停止原因：{row['stop_reason']}；全部推断收敛：{row['all_fits_converged']}</p>")
        curves=[]
        colors={'mc_eir':'#e45756','random':'#4c78a8','space_filling':'#54a24b'}
        all_y=[p['rmse_unobserved_c'] for r in rows.values() for p in r.get('trajectory',[]) if p['rmse_unobserved_c'] is not None]
        ymax=max(all_y,default=1)*1.1
        for method,row in rows.items():
            pts=' '.join(f"{50+(p['sensors']-4)/8*480:.1f},{260-p['rmse_unobserved_c']/ymax*220:.1f}" for p in row.get('trajectory',[]) if p['rmse_unobserved_c'] is not None)
            curves.append(f'<polyline points="{pts}" fill="none" stroke="{colors[method]}" stroke-width="3"/>')
        chart='<svg viewBox="0 0 600 310"><path d="M50 40V260H530" fill="none" stroke="black"/>'+''.join(curves)+f'<text x="50" y="25">未观测格点面积加权RMSE (0–{ymax:.3f} ℃)</text><text x="50" y="290">4 sensors</text><text x="450" y="290">12 sensors</text></svg>'
        sections.append(f'<section><h2>{month}</h2><p>红：MC-EIR；蓝：随机；绿：均匀覆盖。曲线在各方法自身未观测区域计算，区域随选点变化；方法比较应使用下面的共同未观测区域结果。</p>{chart}<div class="maps">'+''.join(blocks)+'</div></section>')
    html='<!doctype html><meta charset="utf-8"><title>真实极地场部署实验</title><style>body{font:16px system-ui;margin:30px;max-width:1300px}svg{width:100%;max-width:600px}.maps{display:grid;grid-template-columns:1fr 1fr;gap:12px}section{border-top:1px solid #ddd;margin-top:25px}p{line-height:1.6}</style><h1>真实极地场：传感器部署数值实验</h1><p>北大西洋侧70–80°N、20°W–20°E，50 m月平均原位温度。训练2000–2012、验证2013、测试2014。POD及预处理未使用2014。地图为经纬度格点示意，未绘制海岸线。</p><p>观测由IAP格点模拟，加入0.02 ℃高斯噪声。ORAS5提供独立来源LF输入。硬件、通信、成本均为情景假设；结果不等同于现场验证。各方法同月使用相同初始点、噪声、预算和约束。随机仅运行一次，不能据此判断统计优势。</p>'+''.join(sections)
    (output/'index.html').write_text(html,encoding='utf-8')


def final_comparison(output,summary):
    """All methods evaluated on the same nodes, excluding their sensor union."""
    for month,methods in summary['months'].items():
        union=set(j for row in methods.values() for j in row['selected'])
        for method,row in methods.items():
            with np.load(output/month/method/'final.npz') as z:
                weights=np.cos(np.deg2rad(z['coordinates_latlon'][:,0]))
                row['common_unobserved_rmse_c']=metrics(z['prediction_insitu_c'],z['truth_insitu_c'],list(union),weights)['rmse_unobserved_c']
                row['lf_common_unobserved_rmse_c']=metrics(z['lf_insitu_c'],z['truth_insitu_c'],list(union),weights)['rmse_unobserved_c']
                row['common_unobserved_nodes']=len(weights)-len(union)


def run(config_path):
    config=json.loads(config_path.read_text(encoding='utf-8'))
    data_dir,output=Path(config['data_directory']),Path(config['output_directory'])
    output.mkdir(parents=True,exist_ok=True)
    log_path=output/'progress.log'
    def log(message):
        print(message,flush=True)
        with log_path.open('a',encoding='utf-8') as stream: stream.write(message+'\n')
    write_json(output/'config.json',config)
    with np.load(data_dir/'aligned_fields.npz') as z: fields={k:z[k] for k in z.files}
    with np.load(data_dir/'training_artifacts.npz') as z: artifact={k:z[k] for k in z.files}
    assert np.max(artifact['train_dates'])==201212 and not np.any(artifact['train_dates']//100==2014)
    coordinates=fields['coordinates_latlon']; n=len(coordinates)
    weights=np.cos(np.deg2rad(coordinates[:,0]))
    gateway=np.array([config['gateway_latlon']])
    distance=distances_km(np.vstack((coordinates,gateway)),'latlon_degrees')[:-1,-1]
    covariance=TrainingCovariance(artifact['covariance_factor'])
    summary={'label':config['label'],'engineering_parameters':'scenario assumptions; not measured hardware',
             'training_end':int(artifact['train_dates'].max()),'pod_modes':artifact['basis'].shape[1],
             'months':{},'metric':'cos(latitude) area-weighted RMSE; unobserved nodes exclude acquired sensors',
             'random_repeats':1,'parameter_tuning':'none; inherited priors, fixed configuration before test evaluation',
             'limitations':['IAP grid observations are simulated, not independent in-situ validation.',
                            'Original retrospective IAP/ORAS5 assimilation independence is unverified.',
                            'Engineering, costs and observation noise are declared scenarios.',
                            'Results compare one random realization; no statistical superiority claim.']}
    start=perf_counter()
    for month in config['months']:
        idx=int(np.flatnonzero(fields['test_dates']==month)[0])
        ice=fields['ice_fraction'][idx]
        cfg=ConstraintConfig(**config['constraints'])
        eligible=np.flatnonzero(np.isfinite(ice)&(ice<=cfg.ice_threshold)&(distance<=cfg.communication_radius_km))
        seed=eligible[np.argsort(distance[eligible],kind='stable')[:config['seed_sensors']]].tolist()
        if len(seed)!=config['seed_sensors']: raise ValueError('Not enough gateway-connected ice-valid seed points')
        truth=fields['test_hf_insitu_c'][idx]
        baseline=artifact['monthly_climatology_c'][month%100-1]+artifact['mean_c']
        scale=float(artifact['scale_c'])
        # One node-indexed realization per month, shared by all methods.
        noise=np.random.default_rng(config['seed']+month).normal(0,config['observation_noise_std_c'],n)
        priorities=np.random.default_rng(config['seed']+month+1).random(n)
        low=fields['test_lf_model'][idx]
        lf_metrics=metrics(fields['test_lf_insitu_c'][idx],truth,[],weights)
        summary['months'][str(month)]={}
        for method in config['methods']:
            log(f'{month} {method}: fitting seed and selecting up to {cfg.max_sensors} sensors')
            constraints=DeploymentConstraints(coordinates,ice,cfg,gateways=gateway)
            if not constraints.evaluate(seed)['feasible']: raise ValueError('Scenario seed infeasible')
            # Unknown ice remains unknown to constraints; finite placeholders only for costs.
            safe_ice=np.where(np.isfinite(ice),ice,0.)
            costs=config['costs']
            planner=ComparisonPlanner(covariance,np.full(n,costs['base']),costs['maintenance_per_ice']*safe_ice,
                costs['failure_per_ice']*safe_ice,PlanningConfig(**config['planning']),
                method=method,coordinates=coordinates,random_priority=priorities)
            model=RBMFCS(artifact['basis'],VBConfig(**config['inference']))
            acquired=[]
            def acquire(j):
                # Only chosen nodes are revealed to inference; planner never sees truth.
                value=(truth[j]+noise[j]-baseline[j])/scale
                acquired.append({'node':int(j),'value_model':float(value)})
                return float(value)
            result=run_closed_loop(model,low,seed,[acquire(j) for j in seed],constraints,planner,acquire)
            snapshots=result.pop('snapshots')
            folder=output/str(month)/method; folder.mkdir(parents=True,exist_ok=True)
            trajectory=[]
            for step,snapshot in enumerate(snapshots):
                selected=seed if step==0 else result['trajectory'][step-1]['selected']
                prediction=baseline+scale*snapshot['mean']
                metric=metrics(prediction,truth,selected,weights)
                metric.update(sensors=len(selected),converged=bool(snapshot['diagnostics']['converged']),
                              iterations=int(snapshot['diagnostics']['iterations']))
                trajectory.append(metric)
                np.savez_compressed(folder/f'posterior_{step:03d}.npz',prediction_insitu_c=prediction,
                    variance_c2=snapshot['variance']*scale**2,selected=selected)
                write_json(folder/f'fit_{step:03d}.json',snapshot['diagnostics'])
            final_prediction=baseline+scale*snapshots[-1]['mean']
            np.savez_compressed(folder/'final.npz',coordinates_latlon=coordinates,truth_insitu_c=truth,
                lf_insitu_c=fields['test_lf_insitu_c'][idx],prediction_insitu_c=final_prediction,
                selected=result['selected'],ice_fraction=ice,variance_c2=snapshots[-1]['variance']*scale**2)
            write_json(folder/'deployment.json',result)
            write_json(folder/'acquisitions.json',acquired)
            row={'stop_reason':result['stop_reason'],'all_fits_converged':all(p['converged'] for p in trajectory),
                 'final':trajectory[-1],'trajectory':trajectory,'selected':result['selected'],
                 'lf_only':lf_metrics,'final_feasible':result['terminal_feasibility']['feasible']}
            summary['months'][str(month)][method]=row
            write_json(output/'summary.json',summary)
            dashboard(output,summary)
            log(f"{month} {method}: {len(result['selected'])} sensors; stop={result['stop_reason']}; unseen RMSE={trajectory[-1]['rmse_unobserved_c']:.4f} C; converged={row['all_fits_converged']}")
    summary['runtime_seconds']=perf_counter()-start
    final_comparison(output,summary)
    summary['input_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [config_path,data_dir/'aligned_fields.npz',data_dir/'training_artifacts.npz',Path(__file__)]}
    write_json(output/'summary.json',summary)
    dashboard(output,summary)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['month','method','sensors','rmse_all_c','rmse_unobserved_c','mae_all_c','bias_c','converged','iterations'])
        writer.writeheader()
        for month,methods in summary['months'].items():
            for method,row in methods.items():
                for point in row['trajectory']: writer.writerow(dict(month=month,method=method,**point))
    complete=all(row['all_fits_converged'] and row['final_feasible'] for methods in summary['months'].values() for row in methods.values())
    log(f"Finished. All fits converged and deployments feasible: {complete}; dashboard: {output/'index.html'}")
    return complete


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/polar_deployment.json')
    args=parser.parse_args()
    sys.exit(0 if run(args.config) else 2)
