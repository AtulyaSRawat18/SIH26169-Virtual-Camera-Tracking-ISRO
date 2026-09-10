import { useEffect, useState } from 'react'
import type { Telemetry } from './types'

const show=(value:number|null|undefined,digits=2)=>value==null?'—':value.toFixed(digits)

export default function OpticalLinkPanel({telemetry}:{telemetry:Telemetry|null}){
  const [config,setConfig]=useState<any>(null),[advanced,setAdvanced]=useState(false),[busy,setBusy]=useState(false),[message,setMessage]=useState(''),[result,setResult]=useState<any>(null)
  useEffect(()=>{fetch('/api/experiment').then(r=>r.json()).then(d=>setConfig(d.config)).catch(e=>setMessage(String(e)))},[])
  async function apply(preset?:string){if(!config)return;setBusy(true);setMessage('');const next=structuredClone(config),link=next.optical_link;link.enabled=true
    if(preset==='NOMINAL')Object.assign(link,{preset:'NOMINAL',transmit_power_w:1,beam_divergence_urad:25,transmitter_efficiency:.78,receiver_efficiency:.72})
    if(preset==='CONSERVATIVE')Object.assign(link,{preset:'CONSERVATIVE',transmit_power_w:.5,beam_divergence_urad:18,transmitter_efficiency:.62,receiver_efficiency:.58})
    try{const r=await fetch('/api/experiment/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(next)}),d=await r.json();if(!r.ok)throw Error(d.detail||'Apply failed');setConfig(d.config)}catch(e){setMessage(String(e))}finally{setBusy(false)}}
  async function run(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}),d=await r.json();if(!r.ok)throw Error(d.detail||'Experiment failed');setResult(d)}catch(e){setMessage(String(e))}finally{setBusy(false)}}
  const link=config?.optical_link
  function numberField(key:string,label:string,unit:string){return <label>{label}<span><input type="number" value={link?.[key]??''} onChange={e=>setConfig({...config,optical_link:{...link,[key]:Number(e.target.value),preset:'CUSTOM'}})}/><em>{unit}</em></span></label>}
  return <section className="panel optical-panel">
    <div className="panel-heading"><div><p className="panel-kicker">OPTICAL LINK CONSEQUENCE</p><h2>Actual pointing → Gaussian loss → received power</h2></div><span className={`link-state ${telemetry?.link_available?'available':''}`}>{telemetry?.link_state??'DISABLED'}</span></div>
    <div className="optical-live"><span>Pointing error<b>{telemetry?.pointing_error_rad==null?'—':show(telemetry.pointing_error_rad*1e6,1)} µrad</b></span><span>Pointing loss<b>{show(telemetry?.pointing_loss_db)} dB</b></span><span>Atmospheric loss<b>{telemetry?.optical_link?.atmospheric_loss_db==null?'—':show(telemetry.optical_link.atmospheric_loss_db)} dB</b></span><span>Received power<b>{show(telemetry?.received_power_dbm)} dBm</b></span><span>Link margin<b>{show(telemetry?.link_margin_db)} dB</b></span></div>
    <div className="optical-controls"><div><button disabled={busy} onClick={()=>void apply('NOMINAL')}>Nominal</button><button disabled={busy} onClick={()=>void apply('CONSERVATIVE')}>Conservative</button><button onClick={()=>setAdvanced(!advanced)}>Advanced {advanced?'−':'+'}</button><button disabled={busy} onClick={()=>void run('/api/optical/pointing-sweep',{config,values_urad:[0,10,25,50,100]})}>Pointing sweep</button><button disabled={busy} onClick={()=>void run('/api/optical/compare',{config,seeds:[41,42],duration_s:2})}>Baseline / AI</button></div><small>Range: {telemetry?.optical_link?.range_mode??'AUTO_FROM_SCENARIO'} · PAT lock and link availability are evaluated separately.</small></div>
    {advanced&&link&&<div className="optical-fields">{numberField('transmit_power_w','Transmit power','W')}{numberField('wavelength_nm','Wavelength','nm')}{numberField('beam_divergence_urad','Beam divergence','µrad')}{numberField('receiver_aperture_m','Receiver aperture','m')}{numberField('receiver_sensitivity_dbm','Receiver sensitivity','dBm')}<button disabled={busy} onClick={()=>void apply()}>Apply custom</button></div>}
    {message&&<p className="lab-message" role="alert">{message}</p>}
    {result&&<details className="optical-result" open><summary>Measured optical result</summary><pre>{JSON.stringify(result.summary??result.rows??result,null,2)}</pre></details>}
  </section>
}
