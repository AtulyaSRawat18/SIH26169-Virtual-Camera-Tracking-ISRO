import { useEffect, useState } from 'react'
import type { Telemetry } from './types'

type Config=Record<string,any>

const stableArchitecture='Centroid → Kalman filter → PID → constrained gimbal → basic reacquisition'
const caseNotes:Record<string,{title:string;why:string}>={
  SAT_SAT_NOMINAL:{title:'Nominal satellite tracking',why:'Smooth relative motion does not justify extra compute; the tested classical loop is sufficient.'},
  DEMO_1_NOMINAL_SAT_SAT:{title:'Nominal satellite tracking',why:'This is the clean reference case, so the smallest validated loop makes the comparison easy to understand.'},
  DEMO_2_HIGH_VIBRATION:{title:'High vibration',why:'Vibration exposes estimator and gimbal limits. The classical chain stays the honest reference until an advanced estimator wins a paired test.'},
  DEMO_3_WEAK_BEACON_GROUND_SAT:{title:'Weak ground-to-satellite beacon',why:'The stable detector is kept as fallback because CNN correction has not yet passed its promotion gate.'},
  DEMO_4_DROPOUT_REACQUISITION:{title:'Dropout and reacquisition',why:'The case begins locked, deliberately hides the beacon, then shows loss and recovery through the same PAT loop.'},
  DEMO_5_AGGRESSIVE_UAV:{title:'Aggressive UAV motion',why:'This case stresses lag and angular rate; predictive control remains available for comparison but is not presented as validated.'},
  DEMO_6_COMBINED_STRESS:{title:'Combined stress',why:'The 100-pair acceptance gate retained this chain because Gradient + UKF increased localization error and latency.'},
}

export default function LiveControlPanel({telemetry}:{telemetry:Telemetry|null}){
  const [config,setConfig]=useState<Config|null>(null),[presets,setPresets]=useState<{name:string;scenario:string}[]>([])
  const [advanced,setAdvanced]=useState(false),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  async function request(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(d.detail||'Request failed');return d}catch(e){setMessage(String(e));return null}finally{setBusy(false)}}
  useEffect(()=>{fetch('/api/experiment').then(r=>r.json()).then(d=>{setConfig(d.config);setPresets(d.catalogue.presets)}).catch(e=>setMessage(String(e)))},[])
  async function choose(name:string,pipeline='DEFAULT_STABLE'){const resolved=await request('/api/scenarios/resolve',{scenario_preset:name,pipeline_preset:pipeline});if(!resolved)return;const reset=await request('/api/experiment/reset',resolved.config);if(reset)setConfig(reset.config)}
  async function level(name:string){if(!config)return;const result=await request('/api/evidence/preset',{config,level:name});if(result)setConfig(result.config)}
  async function change(path:string,value:number){const result=await request('/api/evidence/change',{path,value});if(result)setConfig(result.config)}
  if(!config)return <article className="panel live-controls"><p>{message||'Loading controls…'}</p></article>
  const demos=presets.filter(x=>x.name.startsWith('DEMO_'))
  const relevant=telemetry?.scenario??config.scenario
  const note=caseNotes[config.scenario_preset]??{title:config.scenario_preset.replaceAll('_',' '),why:'Use the stable reference first; compare advanced stages only in a controlled experiment.'}
  const activeArchitecture=`${config.vision.algorithm} → ${config.estimator} → ${config.controller} → constrained gimbal → ${config.reacquisition}`
  return <article className="panel live-controls">
    <div className="live-control-head"><div><p className="panel-kicker">LIVE PHYSICAL CONTROLS</p><h2>Turn a real error up, then watch the loop respond</h2></div><button className="quiet-button" onClick={()=>setAdvanced(v=>!v)}>{advanced?'Presentation mode':'Expert controls'}</button></div>
    <div className="live-control-grid">
      <label>Demo preset<select disabled={busy} value={config.scenario_preset} onChange={e=>void choose(e.target.value)}>{demos.map(x=><option key={x.name}>{x.name}</option>)}{!demos.some(x=>x.name===config.scenario_preset)&&<option>{config.scenario_preset}</option>}</select></label>
      <label>Pipeline<select disabled={busy} value={config.pipeline_preset} onChange={e=>void choose(config.scenario_preset,e.target.value)}><option>DEFAULT_STABLE</option><option>REFERENCE_BASELINE</option><option>CURRENT_BEST_VALIDATED</option></select></label>
      <div className="simple-levels"><span>Error preset</span>{['NOMINAL','LOW','MEDIUM','HIGH','STRESS'].map(x=><button disabled={busy} className={config.disturbance.preset_level===x?'active':''} key={x} onClick={()=>void level(x)}>{x}</button>)}</div>
      {advanced&&<>
        <label>Vibration σ (µrad)<input type="number" min="0" value={(config.disturbance.vibration.stochastic_sigma_rad*1e6).toFixed(1)} onChange={e=>void change('disturbance.vibration.stochastic_sigma_rad',e.target.valueAsNumber*1e-6)}/></label>
        <label>Beacon intensity (DN)<input type="number" min="0" max="255" value={config.disturbance.beacon.nominal_intensity} onChange={e=>void change('disturbance.beacon.nominal_intensity',e.target.valueAsNumber)}/></label>
        <label>Image noise σ (DN)<input type="number" min="0" max="255" value={config.camera.noise_sigma} onChange={e=>void change('camera.noise_sigma',e.target.valueAsNumber)}/></label>
        <label>Random dropout<input type="number" min="0" max="1" step=".01" value={config.disturbance.beacon.random_dropout_probability} onChange={e=>void change('disturbance.beacon.random_dropout_probability',e.target.valueAsNumber)}/></label>
        {relevant!=='satellite_satellite'&&<label>Cloud transmission<input type="number" min="0" max="1" step=".05" value={config.disturbance.atmosphere.cloud_attenuation} onChange={e=>void change('disturbance.atmosphere.cloud_attenuation',e.target.valueAsNumber)}/></label>}
      </>}
    </div>
    <aside className="case-note">
      <div><span>CASE NOTE</span><strong>{note.title}</strong><p>{note.why}</p></div>
      <div><span>RECOMMENDED · VALIDATED</span><strong>{stableArchitecture}</strong><p>Active now: {activeArchitecture}</p></div>
    </aside>
    <div className="live-control-foot"><span>Seed {config.seed}</span><span>{config.disturbance.preset_level} resolves to stored physical values</span><span>{message||'Changes restart the deterministic realization and are timestamped.'}</span></div>
  </article>
}
