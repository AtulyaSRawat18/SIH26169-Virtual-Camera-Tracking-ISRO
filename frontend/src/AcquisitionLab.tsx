import { useEffect, useState } from 'react'
import type { Telemetry } from './types'

export default function AcquisitionLab({telemetry}:{telemetry:Telemetry|null}){
  const [strategies,setStrategies]=useState<string[]>([]),[selected,setSelected]=useState('hybrid')
  const [result,setResult]=useState<any>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  useEffect(()=>{fetch('/api/experiment').then(r=>r.json()).then(d=>setStrategies((d.algorithms.reacquisition||[]).filter((x:string)=>!['basic','none'].includes(x)))).catch(e=>setMessage(String(e)))},[])
  async function post(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed');setResult(d)}catch(e){setMessage(String(e))}finally{setBusy(false)}}
  return <section className="panel lab-panel">
    <div className="panel-heading"><div><p className="panel-kicker">ACQUISITION / REACQUISITION LAB</p><h2>Search only from estimates, predictions, covariance, and time</h2></div><span>{telemetry?.search_active?telemetry.search_mode:'TRACK HANDOFF'}</span></div>
    <div className="lab-controls">
      <label>Strategy<select value={selected} onChange={e=>setSelected(e.target.value)}>{strategies.map(x=><option key={x}>{x}</option>)}</select></label>
      <button disabled={busy} onClick={()=>void post('/api/acquisition/replay',{strategies:[selected],duration_s:2})}>Replay search</button>
      <button disabled={busy} onClick={()=>void post('/api/acquisition/benchmark',{strategies,seeds:[41,42],duration_s:3})}>Search matrix</button>
      <button disabled={busy} onClick={()=>void post('/api/acquisition/sweep',{strategy:selected,parameter:'acquisition.scan_rate_rad_s',values:[.03,.05,.08,.11],seeds:[41]})}>Scan-rate sweep</button>
    </div>
    <div className="estimator-live">
      <span>PAT state <b>{telemetry?.lock_state??'—'}</b></span><span>Search <b>{telemetry?.search_active?'ACTIVE':'IDLE'}</b></span>
      <span>Mode <b>{telemetry?.search_mode??'—'}</b></span><span>Confirmation <b>{telemetry?.confirmation_valid?'VALID':'WAIT'}</b></span>
      <span>Strategy <b>{telemetry?.search_strategy??'—'}</b></span>
    </div>
    {message&&<p className="lab-message" role="alert">{message}</p>}
    {result&&<details open><summary>Measured acquisition result</summary><pre>{JSON.stringify(result.summary??result.strategies??result,null,2)}</pre></details>}
  </section>
}
