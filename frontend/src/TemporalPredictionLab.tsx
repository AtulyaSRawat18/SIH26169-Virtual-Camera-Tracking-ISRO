import { useState } from 'react'
import type { Telemetry } from './types'

export default function TemporalPredictionLab({telemetry}:{telemetry:Telemetry|null}){
  const [predictor,setPredictor]=useState('cv'),[history,setHistory]=useState(12),[horizon,setHorizon]=useState(0.05)
  const [result,setResult]=useState<any>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  async function post(path:string,body:any){
    setBusy(true);setMessage('')
    try{const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      const data=await response.json();if(!response.ok)throw Error(data.detail||'Request failed');setResult(data)
    }catch(error){setMessage(String(error))}finally{setBusy(false)}
  }
  const predictors=['none','cv','ca','gru','lstm','gru_residual_cv']
  return <section className="panel lab-panel">
    <div className="panel-heading"><div><p className="panel-kicker">TEMPORAL PREDICTION LAB · EXPERIMENTAL</p><h2>Classical extrapolation · GRU/LSTM · FF-PID</h2></div><span>PHYSICAL-TIME HORIZON · NOT PROMOTED</span></div>
    <div className="lab-controls">
      <label>Predictor<select value={predictor} onChange={event=>setPredictor(event.target.value)}>{predictors.map(name=><option key={name}>{name}</option>)}</select></label>
      <label>History<select value={history} onChange={event=>setHistory(Number(event.target.value))}>{[5,10,12,20,30].map(value=><option key={value} value={value}>{value} frames</option>)}</select></label>
      <label>Horizon<select value={horizon} onChange={event=>setHorizon(Number(event.target.value))}>{[0.02,0.05,0.1,0.2].map(value=><option key={value} value={value}>{value*1000} ms</option>)}</select></label>
      <button disabled={busy} onClick={()=>void post('/api/prediction/replay',{predictors:[predictor],history_frames:history,horizon_s:horizon,duration_s:3})}>Replay one</button>
      <button disabled={busy} onClick={()=>void post('/api/prediction/replay',{predictors,history_frames:history,horizon_s:horizon,duration_s:3})}>Compare predictors</button>
      <button disabled={busy} onClick={()=>void post('/api/prediction/horizon-sweep',{predictors,duration_s:3})}>Horizon sweep</button>
      <button disabled={busy} onClick={()=>void post('/api/prediction/closed-loop',{predictors:predictors.slice(0,5),horizon_s:horizon,duration_s:2,seeds:[41,42]})}>Closed-loop paired</button>
    </div>
    <div className="estimator-live">
      <span>Live predictor <b>{telemetry?.predictor_name??telemetry?.predictor??'none'}</b></span>
      <span>Future point <b>{telemetry?.predicted_pixel?.map(value=>value.toFixed(1)).join(', ')??'—'}</b></span>
      <span>Horizon <b>{((telemetry?.prediction_horizon_s??0)*1000).toFixed(0)} ms</b></span>
      <span>Latency <b>{telemetry?.predictor_latency_ms?.toFixed(3)??'—'} ms</b></span>
      <span>Fallback <b>{telemetry?.predictor_fallback?telemetry.predictor_failure_reason:'NO'}</b></span>
    </div>
    {message&&<p className="lab-message" role="alert">{message}</p>}
    {result&&<details open><summary>Measured prediction result</summary><pre>{JSON.stringify(result.summaries??result.results??result.points??result,null,2)}</pre></details>}
  </section>
}
