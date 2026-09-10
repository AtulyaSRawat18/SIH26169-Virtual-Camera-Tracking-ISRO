import { useEffect, useState } from 'react'
import type { Telemetry } from './types'

export default function ControllerLab({telemetry}:{telemetry:Telemetry|null}){
  const [controllers,setControllers]=useState<string[]>([]),[selected,setSelected]=useState('pid')
  const [result,setResult]=useState<any>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  useEffect(()=>{fetch('/api/experiment').then(r=>r.json()).then(d=>setControllers(d.algorithms.controller||[])).catch(e=>setMessage(String(e)))},[])
  async function post(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed');setResult(d)}catch(e){setMessage(String(e))}finally{setBusy(false)}}
  return <section className="panel lab-panel">
    <div className="panel-heading"><div><p className="panel-kicker">CONTROLLER LAB</p><h2>Same input · same gimbal · measured trade-offs</h2></div><span>{telemetry?.controller_fallback?'FALLBACK':'ACTIVE'}</span></div>
    <div className="lab-controls">
      <label>Controller<select value={selected} onChange={e=>setSelected(e.target.value)}>{controllers.map(x=><option key={x}>{x}</option>)}</select></label>
      <button disabled={busy} onClick={()=>void post('/api/control/replay',{controllers:[selected],duration_s:3})}>Replay one</button>
      <button disabled={busy} onClick={()=>void post('/api/control/step-response',{controllers,step_rad:.025,duration_s:3})}>Step response</button>
      <button disabled={busy} onClick={()=>void post('/api/control/benchmark',{controllers,seeds:[41,42],duration_s:1.5})}>Scenario matrix</button>
      <button disabled={busy} onClick={()=>void post('/api/control/sweep',{controller:'pid',parameter:'pid.kp',values:[.8,1.2,1.6,2.0,2.4],seeds:[41]})}>PID Kp sweep</button>
    </div>
    <div className="estimator-live">
      <span>Controller <b>{telemetry?.controller_name??'—'}</b></span>
      <span>Latency <b>{telemetry?.controller_latency_ms?.toFixed(3)??'—'} ms</b></span>
      <span>Deadline <b>{telemetry?.controller_deadline_missed?'MISS':'OK'}</b></span>
      <span>Command <b>{telemetry?.controller_output?.map(x=>x.toFixed(3)).join(' / ')??'—'} rad/s</b></span>
      <span>Saturation <b>{telemetry?.controller_command_saturated?'YES':'NO'}</b></span>
    </div>
    {message&&<p role="alert">{message}</p>}
    {result&&<details open><summary>Measured Controller Lab result</summary><pre>{JSON.stringify(result.summary??result.controllers??result,null,2)}</pre></details>}
  </section>
}
