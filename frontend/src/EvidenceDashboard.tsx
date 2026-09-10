import { useEffect, useMemo, useState } from 'react'
import type { Telemetry } from './types'

type RegistryEntry={label:string;category:string;unit:string;scenarios:string[]}
type WaterfallRow={error:string;stage:string;value:number;unit:string;change_from_previous:number|null;relative_change_percent:number|null;status:string|null}

const sweepValues:Record<string,number[]>={
  'disturbance.vibration.stochastic_sigma_rad':[0,5e-6,15e-6,30e-6,50e-6],
  'camera.noise_sigma':[0,5,10,20,35],
  'disturbance.beacon.nominal_intensity':[40,80,140,200,255],
  'disturbance.atmosphere.beam_wander_sigma_rad':[0,5e-6,15e-6,30e-6,50e-6],
  'disturbance.beacon.random_dropout_probability':[0,.02,.05,.1,.2],
  'disturbance.gimbal.command_latency_s':[0,.02,.05,.1,.2],
  'platform.speed_mps':[0,10,25,50,80],
  'platform.altitude_m':[50,100,250,500,1000],
  'platform.angular_rate_rad_s':[0,.005,.01,.03,.06],
}

const format=(value:number|null|undefined,unit='')=>value==null||!Number.isFinite(value)?'—':`${Math.abs(value)<.001&&value!==0?value.toExponential(2):value.toFixed(2)} ${unit}`

function changeTone(value:number|null|undefined){return value==null?'':value>0?'degraded':value<0?'improved':'neutral'}

export default function EvidenceDashboard({telemetry}:{telemetry:Telemetry|null}){
  const [registry,setRegistry]=useState<Record<string,RegistryEntry>>({})
  const [config,setConfig]=useState<any>(null)
  const [level,setLevel]=useState('NOMINAL')
  const [parameter,setParameter]=useState('disturbance.vibration.stochastic_sigma_rad')
  const [busy,setBusy]=useState('')
  const [comparison,setComparison]=useState<any>(null)
  const [sweep,setSweep]=useState<any>(null)
  const [history,setHistory]=useState<any>(null)
  const [failures,setFailures]=useState<any>(null)
  const [error,setError]=useState('')
  async function json(path:string,options?:RequestInit){const response=await fetch(path,options);const body=await response.json();if(!response.ok)throw Error(body.detail||'Request failed');return body}
  useEffect(()=>{Promise.all([json('/api/experiment'),json('/api/evidence/registry'),json('/api/evidence/cumulative'),json('/api/evidence/failures')]).then(([experiment,meta,cumulative,failureData])=>{setConfig(experiment.config);setRegistry(meta.variables);setHistory(cumulative);setFailures(failureData);const first=Object.keys(meta.variables)[0];if(first)setParameter(first)}).catch(e=>setError(String(e)))},[])
  async function post(path:string,body:any,label:string){setBusy(label);setError('');try{return await json(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})}catch(e){setError(String(e));return null}finally{setBusy('')}}
  async function applyPreset(next:string){if(!config)return;setLevel(next);const result=await post('/api/evidence/preset',{config,level:next},'preset')
    if(result)setConfig(result.config)}
  async function compare(){if(!config)return;const baseline=structuredClone(config);baseline.pipeline_preset='CUSTOM';baseline.vision={...baseline.vision,algorithm:'centroid',correction:'none',cnn_correction:false};baseline.cnn={...baseline.cnn,enabled:false};baseline.estimator='kalman';baseline.predictor='none';baseline.controller='pid';baseline.reacquisition='basic';const result=await post('/api/evidence/before-after',{config_a:baseline,config_b:config,runs:3,duration_s:Math.min(config.duration_s,2),base_seed:config.seed},'compare');if(result)setComparison(result)}
  async function runSweep(){if(!config)return;const values=sweepValues[parameter]??[0,.25,.5,.75,1];const result=await post('/api/evidence/sweep',{config,parameter,values,runs_per_point:3,duration_s:Math.min(config.duration_s,1.5),base_seed:config.seed,distributions:{[parameter]:{distribution:'fixed',value:values[0]}}},'sweep');if(result)setSweep(result)}
  async function refreshHistory(){setBusy('history');try{setHistory(await json(`/api/evidence/cumulative?scenario=${config?.scenario??''}`))}catch(e){setError(String(e))}finally{setBusy('')}}
  async function replay(runId:string){const result=await post('/api/experiments/replay',{run_id:runId},'replay');if(result){setConfig(result.config);setError(`Replaying stored run ${runId.slice(0,8)} with original provenance.`)}}
  const waterfall=(telemetry?.error_waterfall??[]) as WaterfallRow[]
  const pxMax=Math.max(1,...waterfall.filter(x=>x.unit==='px').map(x=>x.value))
  const exportPayload=useMemo(()=>({exported_at:new Date().toISOString(),scenario:telemetry?.scenario,seed:telemetry?.seed,resolved_error_config:telemetry?.resolved_error_config,live_waterfall:waterfall,running_metrics:telemetry?.metrics,before_after:comparison,parameter_sweep:sweep,cumulative:history}),[telemetry,waterfall,comparison,sweep,history])
  function download(){const blob=new Blob([JSON.stringify(exportPayload,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download=`pat-evidence-${telemetry?.scenario??'run'}-${telemetry?.seed??42}.json`;a.click();URL.revokeObjectURL(url)}
  return <section className="evidence-dashboard">
    <div className="section-intro"><div><p className="panel-kicker">EVIDENCE / RESULTS</p><h2>What changed, where it propagated, and how certain we are</h2></div><div className="evidence-actions"><button className="quiet-button" onClick={download}>Export JSON</button><button className="quiet-button" onClick={()=>window.open('/api/evidence/report','_blank')}>Print-ready report</button></div></div>
    {error&&<p className="evidence-alert" role="alert">{error}</p>}
    <article className="panel evidence-control-panel">
      <div><h3>Scenario-aware physical controls</h3><p>Simple levels resolve to exact values. Advanced values remain available in Experiment.</p></div>
      <div className="level-buttons">{['NOMINAL','LOW','MEDIUM','HIGH','STRESS'].map(x=><button key={x} className={level===x?'active':''} disabled={!!busy} onClick={()=>void applyPreset(x)}>{x}</button>)}</div>
      <details><summary>View resolved error configuration</summary><pre>{JSON.stringify(telemetry?.resolved_error_config??{},null,2)}</pre></details>
    </article>
    <div className="evidence-three">
      <article className="panel waterfall-card"><div className="evidence-heading"><div><p className="panel-kicker">LIVE ERROR FLOW</p><h3>Measured stage errors</h3></div><span>TRUTH USED ONLY FOR EVALUATION</span></div>
        <div className="waterfall-list">{waterfall.length?waterfall.map(row=><div className="waterfall-row" key={row.error}><div><strong>{row.error.replace('e_','').replace('_',' ')}</strong><small>{row.stage}</small></div><div className="error-track"><i style={{width:row.unit==='px'?`${Math.max(2,row.value/pxMax*100)}%`:'35%'}}/></div><b>{format(row.value,row.unit)}</b><em className={changeTone(row.change_from_previous)}>{row.change_from_previous==null?'—':`${row.status} ${format(row.relative_change_percent,'%')}`}</em></div>):<p>Waiting for evaluated telemetry.</p>}</div>
        <div className="live-stats"><span>Pointing <b>{format(telemetry?.pointing_error_rad==null?null:telemetry.pointing_error_rad*1e6,'µrad')}</b></span><span>Link margin <b>{format(telemetry?.link_margin_db,'dB')}</b></span><span>Lock <b>{telemetry?.lock_state??'—'}</b></span></div>
      </article>
      <article className="panel before-card"><div className="evidence-heading"><div><p className="panel-kicker">BEFORE / AFTER</p><h3>Paired physical conditions</h3></div><span>SAME SEEDS</span></div><p>Baseline centroid → KF → PID versus the currently selected candidate. Degradation remains visible.</p><button disabled={!!busy||!config} onClick={()=>void compare()}>{busy==='compare'?'Running paired seeds…':'Run paired comparison'}</button>
        {comparison&&<div className="change-list">{Object.entries(comparison.changes).slice(0,7).map(([name,item]:[string,any])=><div key={name}><span>{name.replaceAll('_',' ')}</span><b>{format(item.baseline)} → {format(item.candidate)}</b><em className={item.status==='IMPROVED'?'improved':item.status==='DEGRADED'?'degraded':''}>{item.status} · {format(item.improvement_percent,'%')}</em></div>)}</div>}
      </article>
      <article className="panel monte-card"><div className="evidence-heading"><div><p className="panel-kicker">MONTE CARLO / CUMULATIVE</p><h3>One-variable sensitivity</h3></div><span>SEEDED CI</span></div><label>Controlled variable<select value={parameter} onChange={e=>setParameter(e.target.value)}>{Object.entries(registry).map(([path,meta])=><option value={path} key={path}>{meta.label} · {meta.unit}</option>)}</select></label><button disabled={!!busy||!config} onClick={()=>void runSweep()}>{busy==='sweep'?'Running sweep…':'Run 5-point sweep'}</button>
        {sweep&&<div className="sweep-mini">{sweep.points.map((point:any)=><div key={point.value}><span>{format(point.value,registry[parameter]?.unit)}</span><i style={{height:`${Math.max(4,(point.tracking_rmse.mean??0)/Math.max(...sweep.points.map((p:any)=>p.tracking_rmse.mean??0),1)*100)}%`}}/><b>{format(point.tracking_rmse.mean,'px')}</b></div>)}</div>}
        <div className="history-strip"><span>Compatible history</span><strong>{history?.runs??0} runs</strong><button className="quiet-button" onClick={()=>void refreshHistory()}>Refresh</button></div>
      </article>
    </div>
    <article className="panel failure-panel"><div className="evidence-heading"><div><p className="panel-kicker">FAILURE / WORST-CASE EVIDENCE</p><h3>Failed runs remain in the evidence</h3></div><span>REPLAYABLE SEEDS</span></div><div className="failure-content"><div>{(failures?.distribution??[]).map((row:any)=><p key={row.cause}><span>{row.cause.replaceAll('_',' ')}</span><b>{row.count}</b><small>{row.percent.toFixed(1)}%</small></p>)}</div><div>{(failures?.worst_runs??[]).slice(0,5).map((run:any)=><p key={run.run_id}><span>{run.scenario} · seed {run.seed}</span><b>{run.metrics.rmse_tracking_error_px?.toFixed(2)??'—'} px</b><button onClick={()=>void replay(run.run_id)}>Replay</button></p>)}</div></div></article>
  </section>
}
