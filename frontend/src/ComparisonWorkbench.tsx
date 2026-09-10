import { useEffect, useMemo, useState } from 'react'
import type { Telemetry } from './types'

type Sample = { frame:number; cv:number|null; estimate:number|null; tracking:number|null }
type SeriesKey = 'cv'|'estimate'|'tracking'

const distance=(a:[number,number]|null,b:[number,number]|null)=>a&&b?Math.hypot(a[0]-b[0],a[1]-b[1]):null
const colors:Record<SeriesKey,string>={cv:'#53d3b0',estimate:'#65a7ff',tracking:'#f2b765'}
const labels:Record<SeriesKey,string>={cv:'OpenCV localization',estimate:'Estimator residual',tracking:'Closed-loop tracking'}

function stats(values:(number|null)[]){
  const clean=values.filter((v):v is number=>v!==null&&Number.isFinite(v)).sort((a,b)=>a-b)
  if(!clean.length)return {latest:null,rmse:null,mean:null,p95:null,samples:0}
  const rmse=Math.sqrt(clean.reduce((sum,v)=>sum+v*v,0)/clean.length)
  return {latest:clean.at(-1)??null,rmse,mean:clean.reduce((sum,v)=>sum+v,0)/clean.length,p95:clean[Math.min(clean.length-1,Math.floor(clean.length*.95))],samples:clean.length}
}

function EvidenceBars({title,subtitle,values}:{title:string;subtitle:string;values:Record<string,number>}){
  const entries=Object.entries(values).filter(([,v])=>Number.isFinite(v))
  const max=Math.max(...entries.map(([,v])=>v),1)
  return <article className="evidence-card">
    <div className="evidence-card-title"><div><h3>{title}</h3><p>{subtitle}</p></div><span>RMSE · LOWER IS BETTER</span></div>
    <div className="bar-list">{entries.map(([name,v])=><div className="bar-row" key={name}>
      <strong>{name.replaceAll('_',' ')}</strong><div className="bar-track"><i style={{width:`${Math.max(2,v/max*100)}%`}}/></div><b>{v.toFixed(2)} px</b>
    </div>)}</div>
  </article>
}

export default function ComparisonWorkbench({telemetry}:{telemetry:Telemetry|null}){
  const [samples,setSamples]=useState<Sample[]>([])
  const [evidence,setEvidence]=useState<any>(null)
  const [visible,setVisible]=useState<Record<SeriesKey,boolean>>({cv:true,estimate:true,tracking:true})
  useEffect(()=>{fetch('/api/evidence/summaries').then(r=>r.json()).then(setEvidence).catch(()=>setEvidence({}))},[])
  useEffect(()=>{
    if(!telemetry)return
    setSamples(current=>{
      if(current.at(-1)?.frame===telemetry.frame)return current
      const next:Sample={frame:telemetry.frame,cv:distance(telemetry.opencv_pixel,telemetry.ground_truth_pixel),estimate:distance(telemetry.kalman_pixel,telemetry.ground_truth_pixel),tracking:telemetry.tracking_error_px}
      return [...current,next].slice(-180)
    })
  },[telemetry])
  const summaries=useMemo(()=>({
    cv:stats(samples.map(s=>s.cv)),estimate:stats(samples.map(s=>s.estimate)),tracking:stats(samples.map(s=>s.tracking))
  }),[samples])
  const width=900,height=260,pad=34
  const plotted=(Object.keys(visible) as SeriesKey[]).filter(key=>visible[key])
  const ymax=Math.max(1,...samples.flatMap(s=>plotted.map(key=>s[key]??0)))
  const path=(key:SeriesKey)=>samples.map((sample,index)=>{
    const v=sample[key];if(v===null)return null
    const x=pad+(index/Math.max(1,samples.length-1))*(width-pad*2),y=height-pad-(v/ymax)*(height-pad*2)
    return `${index===0?'M':'L'} ${x.toFixed(1)} ${y.toFixed(1)}`
  }).filter(Boolean).join(' ')
  const cnn=evidence?.cnn
  const temporal=evidence?.temporal
  return <section className="compare-workbench">
    <div className="section-intro"><div><p className="panel-kicker">MEASURED COMPARISON</p><h2>One view for live behavior and saved evidence</h2></div><button className="quiet-button" onClick={()=>setSamples([])}>Clear live window</button></div>
    <article className="panel comparison-panel">
      <div className="comparison-toolbar">
        <div><h3>Rolling image-plane error</h3><p>Last {samples.length} frames · hidden truth appears only in evaluation metrics</p></div>
        <div className="series-toggles">{(Object.keys(visible) as SeriesKey[]).map(key=><button className={visible[key]?'active':''} key={key} onClick={()=>setVisible(v=>({...v,[key]:!v[key]}))}><i style={{background:colors[key]}}/>{labels[key]}</button>)}</div>
      </div>
      <div className="chart-wrap">
        <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Rolling error comparison chart">
          {[0,.25,.5,.75,1].map(f=><g key={f}><line x1={pad} x2={width-pad} y1={pad+f*(height-pad*2)} y2={pad+f*(height-pad*2)}/><text x="2" y={pad+f*(height-pad*2)+4}>{(ymax*(1-f)).toFixed(0)}</text></g>)}
          {plotted.map(key=><path key={key} d={path(key)} stroke={colors[key]} fill="none" strokeWidth="2.2" vectorEffect="non-scaling-stroke"/>) }
        </svg>
        {!samples.length&&<div className="chart-empty">Waiting for telemetry</div>}
      </div>
      <div className="numeric-grid">{(Object.keys(summaries) as SeriesKey[]).map(key=>{const s=summaries[key];return <article key={key} style={{'--series':colors[key]} as React.CSSProperties}>
        <p>{labels[key]}</p><strong>{s.rmse===null?'—':s.rmse.toFixed(2)} <small>px RMSE</small></strong>
        <dl><div><dt>Mean</dt><dd>{s.mean?.toFixed(2)??'—'}</dd></div><div><dt>P95</dt><dd>{s.p95?.toFixed(2)??'—'}</dd></div><div><dt>Frames</dt><dd>{s.samples}</dd></div></dl>
      </article>})}</div>
    </article>
    <div className="evidence-grid">
      {cnn&&<EvidenceBars title="CNN correction" subtitle="Held-out image localization" values={{classical:cnn.held_out.classical_rmse_px,cnn:cnn.held_out.cnn_rmse_px}}/>}
      {temporal&&<EvidenceBars title="Temporal prediction" subtitle="Out-of-distribution future position" values={temporal.ood_rmse_px}/>}
    </div>
    <div className="decision-strip"><span>Current decision</span><strong>Classical pipeline stays the default</strong><p>CNN and temporal models remain visible as experiments because their gains do not yet transfer reliably to the complete closed loop.</p></div>
  </section>
}
