import type { Telemetry } from './types'

const format=(value:number,unit:string)=>unit==='rad'?`${(value*1e6).toFixed(1)} µrad`:unit==='rad/s'?`${(value*1e3).toFixed(2)} mrad/s`:`${value.toFixed(2)} ${unit}`

export default function LiveErrorFlow({telemetry}:{telemetry:Telemetry|null}){
  const rows=telemetry?.error_waterfall??[]
  return <article className="panel live-flow-panel">
    <div className="panel-heading"><div><p className="panel-kicker">LIVE ERROR WATERFALL</p><h2>Actual stage output · not a scripted improvement</h2></div><span>EVALUATION TRUTH ISOLATED</span></div>
    <div className="live-flow-body">
      <div className="live-flow-stages">{rows.map((row,index)=><div key={row.error}><span>{row.error.replace('e_','').replace('_',' ')}</span><strong>{format(row.value,row.unit)}</strong>{row.status&&<small className={row.status==='DEGRADED'?'degraded':row.status==='IMPROVED'?'improved':''}>{row.status}{row.relative_change_percent!=null?` ${Math.abs(row.relative_change_percent).toFixed(1)}%`:''}</small>}{index<rows.length-1&&<b>→</b>}</div>)}</div>
      <div className="pat-now"><span>PAT STATE</span><strong>{telemetry?.lock_state??'CONNECTING'}</strong><small>{(telemetry?.time_in_pat_state_s??0).toFixed(1)} s in state</small></div>
      <div className="event-timeline"><span>RECENT EVENTS</span>{(telemetry?.pat_events??[]).slice(-5).reverse().map((event,index)=><p key={`${event.timestamp_s}-${index}`}><time>{event.timestamp_s.toFixed(1)}s</time><b>{event.event}</b><small>{event.reason}</small></p>)}</div>
    </div>
  </article>
}
