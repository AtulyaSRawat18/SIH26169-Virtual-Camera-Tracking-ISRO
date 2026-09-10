import { useEffect, useState } from 'react'
import type { Telemetry } from './types'

export default function EstimatorLab({telemetry}:{telemetry:Telemetry|null}){
  const [estimators,setEstimators]=useState<string[]>([]),[selected,setSelected]=useState('kf_cv')
  const [result,setResult]=useState<any>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  useEffect(()=>{fetch('/api/experiment').then(r=>r.json()).then(d=>setEstimators(d.algorithms.estimator||[])).catch(e=>setMessage(String(e)))},[])
  async function post(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed');setResult(d)}catch(e){setMessage(String(e))}finally{setBusy(false)}}
  const compare=estimators.filter(x=>x!=='kalman')
  const ellipse=telemetry?.estimator_uncertainty_ellipse
  return <section className="panel lab-panel">
    <div className="panel-heading"><div><p className="panel-kicker">STATE ESTIMATION LAB</p><h2>Open-loop replay · closed-loop PAT · uncertainty</h2></div><span>{telemetry?.estimator_prediction_only?'PREDICTION ONLY':'MEASUREMENT UPDATE'}</span></div>
    <div className="lab-controls">
      <label>Estimator<select value={selected} onChange={e=>setSelected(e.target.value)}>{estimators.map(x=><option key={x}>{x}</option>)}</select></label>
      <button disabled={busy} onClick={()=>void post('/api/estimation/replay',{estimators:[selected],duration_s:2})}>Replay one</button>
      <button disabled={busy} onClick={()=>void post('/api/estimation/replay',{estimators:compare,duration_s:2})}>Compare estimators</button>
      <button disabled={busy} onClick={()=>void post('/api/estimation/benchmark',{estimators:compare,seeds:[41,42]})}>Scenario matrix</button>
      <button disabled={busy} onClick={()=>void post('/api/estimation/sweep',{parameter:'estimation.measurement_noise_px2',values:[1,4,9,16],estimators:['kf_cv','ekf_angular','ukf_angular'],duration_s:2})}>R sweep</button>
    </div>
    <div className="estimator-live">
      <span>NIS <b>{telemetry?.estimator_nis?.toFixed(3)??'—'}</b></span><span>NEES₂ <b>{telemetry?.estimator_nees_position?.toFixed(3)??'—'}</b></span>
      <span>R scale <b>{telemetry?.estimator_r_scale?.toFixed(2)??'—'}</b></span><span>Latency <b>{telemetry?.estimator_latency_ms?.toFixed(3)??'—'} ms</b></span>
      <span>95% ellipse <b>{ellipse?`${ellipse.major_axis_px.toFixed(1)} × ${ellipse.minor_axis_px.toFixed(1)} px`:'—'}</b></span>
    </div>
    {message&&<p role="alert">{message}</p>}
    {result&&<details open><summary>Measured Estimator Lab result</summary><pre>{JSON.stringify(result.estimators?Object.fromEntries(Object.entries(result.estimators).map(([k,v]:any)=>[k,v.summary])):result.summary??result.points??result,null,2)}</pre></details>}
  </section>
}
