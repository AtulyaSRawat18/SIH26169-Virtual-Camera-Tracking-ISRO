import { useState } from 'react'
import type { Telemetry } from './types'

export default function AICorrectionLab({telemetry}:{telemetry:Telemetry|null}){
  const [result,setResult]=useState<any>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  async function post(path:string,body:any={}){
    setBusy(true);setMessage('')
    try{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      const data=await response.json();if(!response.ok)throw Error(data.detail||'Request failed');setResult(data)
    }catch(error){setMessage(String(error))}finally{setBusy(false)}
  }
  const sigma=telemetry?.measurement_quality?.cnn_predicted_sigma_px
  return <section className="panel lab-panel">
    <div className="panel-heading"><div><p className="panel-kicker">AI MEASUREMENT CORRECTION · EXPERIMENTAL</p><h2>CNN residual · learned measurement uncertainty</h2></div><span className="evaluation-label">NOT PROMOTED · TRUTH USED FOR METRICS ONLY</span></div>
    <div className="lab-controls">
      <button disabled={busy} onClick={()=>void post('/api/ai/compare-frame')}>Compare current frame</button>
      <button disabled={busy} onClick={()=>void post('/api/ai/dataset/generate',{trajectories_per_scenario:3,frames_per_trajectory:20})}>Generate dataset</button>
      <button disabled={busy} onClick={()=>void post('/api/ai/train',{epochs:12})}>Train CNN</button>
    </div>
    <div className="estimator-live">
      <span>Correction <b>{telemetry?.correction_algorithm??'none'}</b></span>
      <span>Vector <b>{telemetry?.correction_vector_px?.map(value=>value.toFixed(3)).join(', ')??'—'} px</b></span>
      <span>Learned σ <b>{sigma?.map(value=>value.toFixed(2)).join(' × ')??'—'} px</b></span>
      <span>Latency <b>{telemetry?.correction_latency_ms?.toFixed(3)??'—'} ms</b></span>
      <span>Fallback <b>{telemetry?.correction_fallback?telemetry.correction_failure_reason:'NO'}</b></span>
    </div>
    {message&&<p className="lab-message" role="alert">{message}</p>}
    {result?.overlay_image&&<div className="ai-overlay"><img src={result.overlay_image} alt="Classical and CNN correction evaluation overlay"/></div>}
    {result&&<details open><summary>Measured AI correction result</summary><pre>{JSON.stringify({...result,overlay_image:result.overlay_image?'[image shown above]':undefined},null,2)}</pre></details>}
  </section>
}
