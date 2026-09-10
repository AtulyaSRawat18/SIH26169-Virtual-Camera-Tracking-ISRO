import { useEffect, useState } from 'react'

type Result = {algorithm:string;localization_error_px:number|null;latency_ms:number;measurement:{valid:boolean;confidence:number|null;image_snr_estimate:number|null;failure_reason:string|null;quality:Record<string,unknown>};mask_image:string|null;roi_image:string|null;overlay_image:string|null}
type Comparison = {benchmark_mode:string;raw_image:string|null;truth_evaluation_only:number[]|null;results:Result[]}

export default function VisionLab(){
  const [algorithms,setAlgorithms]=useState<string[]>([]),[selected,setSelected]=useState('weighted_centroid')
  const [mode,setMode]=useState('FULL_FRAME'),[result,setResult]=useState<Comparison|null>(null)
  const [report,setReport]=useState<any>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  useEffect(()=>{fetch('/api/experiment').then(r=>r.json()).then(d=>setAlgorithms(d.algorithms.vision||[])).catch(e=>setMessage(String(e)))},[])
  async function post(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed');return d}catch(e){setMessage(String(e));return null}finally{setBusy(false)}}
  async function compare(names:string[]){const d=await post('/api/vision/compare-frame',{algorithms:names,benchmark_mode:mode});if(d){setResult(d);setReport(null)}}
  async function benchmark(){const d=await post('/api/vision/benchmark',{algorithms:compareNames,seeds:[41,42],frames_per_seed:4,benchmark_mode:mode});if(d){setReport(d);setResult(null)}}
  async function sweep(){const d=await post('/api/vision/sweep',{parameter:'camera.noise_sigma',values:[0,5,12,22],algorithms:compareNames,seeds:[41,42],frames_per_seed:3});if(d){setReport(d);setResult(null)}}
  const compareNames=algorithms.filter(x=>!['centroid','legacy_centroid'].includes(x))
  return <section className="panel lab-panel">
    <div className="panel-heading"><div><p className="panel-kicker">CLASSICAL SPOT LOCALIZATION</p><h2>Vision Lab · identical pixels, measured outcomes</h2></div><span className="evaluation-label">TRUTH OVERLAY · EVALUATION ONLY</span></div>
    <div className="lab-controls">
      <label>Algorithm<select value={selected} onChange={e=>setSelected(e.target.value)}>{algorithms.map(x=><option key={x}>{x}</option>)}</select></label>
      <label>Benchmark mode<select value={mode} onChange={e=>setMode(e.target.value)}><option>FULL_FRAME</option><option>ORACLE_ROI</option></select></label>
      <button disabled={busy} onClick={()=>void compare([selected])}>Run single</button>
      <button disabled={busy} onClick={()=>void compare(compareNames)}>Compare classical</button>
      <button disabled={busy} onClick={()=>void benchmark()}>Paired benchmark</button>
      <button disabled={busy} onClick={()=>void sweep()}>Noise sweep</button>
    </div>
    {message&&<p role="alert">{message}</p>}
    {result&&<div className="vision-grid">
      {result.results.map(item=><article key={item.algorithm}>
        <h3>{item.algorithm}</h3>{item.overlay_image&&<img src={item.overlay_image} alt={`${item.algorithm} evaluation overlay`}/>}<div className="debug-thumbs">{item.mask_image&&<img src={item.mask_image} alt="candidate mask"/>}{item.roi_image&&<img src={item.roi_image} alt="selected ROI"/>}</div>
        <p>Error <b>{item.localization_error_px?.toFixed(3)??'invalid'} px</b> · {item.latency_ms.toFixed(2)} ms</p>
        <p>Confidence {item.measurement.confidence?.toFixed(3)??'—'} · image SNR {item.measurement.image_snr_estimate?.toFixed(2)??'—'}</p>
        {item.measurement.failure_reason&&<p className="warning">{item.measurement.failure_reason}</p>}
      </article>)}
    </div>}
    {report&&<details open><summary>Measured Vision Lab result</summary><pre>{JSON.stringify(report.summary??report.points??report,null,2)}</pre></details>}
  </section>
}
