import { useEffect, useState } from 'react'

type AnyConfig = Record<string, any>
type Catalogue = { scenario_version: string; presets: {name:string;scenario:string}[]; pipelines: Record<string,{status:string;required?:string[]}> }

export default function ExperimentPanel() {
  const [config,setConfig]=useState<AnyConfig|null>(null),[algorithms,setAlgorithms]=useState<Record<string,string[]>>({})
  const [catalogue,setCatalogue]=useState<Catalogue|null>(null),[busy,setBusy]=useState(false),[message,setMessage]=useState('')
  const [result,setResult]=useState<any>(null),[runs,setRuns]=useState(100),[presetName,setPresetName]=useState('My configuration')
  const [summaryOnly,setSummaryOnly]=useState(true),[distributionMode,setDistributionMode]=useState<'fixed'|'uniform'|'normal'>('fixed')
  const [distributionPath,setDistributionPath]=useState('disturbance.vibration.stochastic_sigma_rad'),[distributionA,setDistributionA]=useState(0),[distributionB,setDistributionB]=useState(.00005)
  const [saved,setSaved]=useState<Record<string,AnyConfig>>(()=>JSON.parse(localStorage.getItem('pat-presets')||'{}'))
  const [selectedSaved,setSelectedSaved]=useState('')
  const saveAll=(next:Record<string,AnyConfig>)=>{setSaved(next);localStorage.setItem('pat-presets',JSON.stringify(next))}
  async function load(){try{const r=await fetch('/api/experiment');const d=await r.json();if(!r.ok)throw Error(d.detail||'Cannot load configuration');setConfig(d.config);setAlgorithms(d.algorithms);setCatalogue(d.catalogue)}catch(e){setMessage(String(e))}}
  useEffect(()=>{void load()},[])
  async function post(path:string,body:any){setBusy(true);setMessage('');try{const r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();if(!r.ok)throw Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail));return d}catch(e){setMessage(String(e));return null}finally{setBusy(false)}}
  async function chooseProfile(name:string,pipeline='DEFAULT_STABLE'){const d=await post('/api/scenarios/resolve',{scenario_preset:name,pipeline_preset:pipeline});if(d)setConfig(d.config)}
  async function live(){const d=await post('/api/experiment/reset',config);if(d){setConfig(d.config);setResult({mode:'LIVE',summary:'Experiment restarted; live metrics cleared.'})}}
  const distribution=distributionMode==='fixed'?{distribution:'fixed',value:distributionA}:distributionMode==='uniform'?{distribution:'uniform',low:distributionA,high:distributionB}:{distribution:'normal',mean:distributionA,std:distributionB}
  async function batch(count=runs){const d=await post('/api/experiments/monte-carlo',{config,runs:count,base_seed:config?.seed??42,duration_s:config?.duration_s??10,summary_only:summaryOnly,distributions:{[distributionPath]:distribution}});if(d)setResult({mode:count===1?'SINGLE HEADLESS':'MONTE CARLO',...d})}
  async function compare(){if(!config)return;const alternate={...config,estimator:config.estimator==='kalman'?'none':'kalman',pipeline_preset:'CUSTOM'};const d=await post('/api/experiments/compare',{config_a:config,config_b:alternate,runs,base_seed:config.seed,duration_s:config.duration_s,summary_only:true});if(d)setResult({mode:`COMPARE ${config.estimator} vs ${alternate.estimator}`,...d})}
  async function history(){try{const r=await fetch(`/api/experiments/history?scenario=${config?.scenario}`);const d=await r.json();setResult({mode:'COMPATIBLE HISTORY',...d})}catch(e){setMessage(String(e))}}
  const patch=(path:string,value:any)=>setConfig(current=>{if(!current)return current;const next=structuredClone(current);const parts=path.split('.');let target=next;parts.slice(0,-1).forEach(p=>target=target[p]);target[parts.at(-1)!]=value;next.pipeline_preset=path.startsWith('vision.')||['estimator','predictor','controller','reacquisition'].includes(path)?'CUSTOM':next.pipeline_preset;return next})
  const download=(name:string,payload:any)=>{const url=URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url)}
  if(!config||!catalogue)return <section className="panel experiment"><h2>PAT experiment lab</h2><button onClick={()=>void load()}>Load</button><p role="alert">{message}</p></section>
  const profiles=catalogue.presets.filter(p=>p.scenario===config.scenario)
  const isUav=config.scenario.includes('uav'),hasAtmosphere=config.scenario!=='satellite_satellite'
  return <section className="panel experiment">
    <div className="experiment-title"><div><p className="panel-kicker">PAT EXPERIMENT LAB</p><h2>Scenario · Environment · Pipeline · Results</h2></div><span>Schema {config.simulation_schema_version}</span></div>
    <div className="config-grid">
      <label>Scenario<select value={config.scenario} onChange={e=>{const first=catalogue.presets.find(p=>p.scenario===e.target.value);if(first)void chooseProfile(first.name)}}>{[...new Set(catalogue.presets.map(p=>p.scenario))].map(x=><option key={x}>{x}</option>)}</select></label>
      <label>Environment profile<select value={config.scenario_preset} onChange={e=>void chooseProfile(e.target.value)}>{profiles.map(p=><option key={p.name}>{p.name}</option>)}</select></label>
      <label>Architecture<select value={config.pipeline_preset==='CUSTOM'?'CUSTOM':config.pipeline_preset} onChange={e=>{if(e.target.value!=='CUSTOM')void chooseProfile(config.scenario_preset,e.target.value)}}>{Object.entries(catalogue.pipelines).map(([name,value])=><option key={name} value={name} disabled={value.status!=='runnable'}>{name}{value.status!=='runnable'?` · ${value.status}`:''}</option>)}{config.pipeline_preset==='CUSTOM'&&<option>CUSTOM</option>}</select></label>
      <label>Master seed<input type="number" min="0" max="4294967295" value={config.seed} onChange={e=>patch('seed',e.target.valueAsNumber)}/></label>
      <label>Duration (s)<input type="number" min="0.1" max="10000" value={config.duration_s} onChange={e=>patch('duration_s',e.target.valueAsNumber)}/></label>
    </div>

    <details open><summary>Edit pipeline</summary><div className="pipeline-editor">
      {(['vision','correction','estimator','predictor','controller','reacquisition'] as const).map((stage,index)=><span key={stage}><label>{stage}<select value={stage==='vision'?config.vision.algorithm:stage==='correction'?config.vision.correction:config[stage]} onChange={e=>{
        if(stage==='correction'){setConfig(current=>current?{...current,vision:{...current.vision,correction:e.target.value,cnn_correction:e.target.value==='cnn_residual'},cnn:{...current.cnn,enabled:e.target.value==='cnn_residual'},pipeline_preset:'CUSTOM'}:current)}
        else patch(stage==='vision'?'vision.algorithm':stage,e.target.value)
      }}>{(algorithms[stage]||[]).map(x=><option key={x}>{x}</option>)}</select></label>{index<5&&<b>↓</b>}</span>)}</div></details>

    <details><summary>Platform and geometry</summary><div className="config-grid">
      <label>Observer<input value={config.platform.observer_type} disabled/></label><label>Target<input value={config.platform.target_type} disabled/></label>
      <label>{isUav?'UAV altitude':'Observer altitude'} (m)<input type="number" min="0" value={config.platform.altitude_m} onChange={e=>patch('platform.altitude_m',e.target.valueAsNumber)}/></label>
      <label>Target altitude (m)<input type="number" min="0" value={config.platform.target_altitude_m} onChange={e=>patch('platform.target_altitude_m',e.target.valueAsNumber)}/></label>
      <label>Speed (m/s)<input type="number" min="0" value={config.platform.speed_mps} onChange={e=>patch('platform.speed_mps',e.target.valueAsNumber)}/></label>
      <label>Angular rate (rad/s)<input type="number" min="0" step="any" value={config.platform.angular_rate_rad_s} onChange={e=>patch('platform.angular_rate_rad_s',e.target.valueAsNumber)}/></label>
      {isUav&&<label>Wind (m/s)<input type="number" min="0" value={config.platform.wind_speed_mps} onChange={e=>patch('platform.wind_speed_mps',e.target.valueAsNumber)}/></label>}
    </div></details>

    <details><summary>Optics, beacon and camera</summary><div className="config-grid">
      <label>Beacon intensity<input type="number" min="0" max="255" value={config.disturbance.beacon.nominal_intensity} onChange={e=>patch('disturbance.beacon.nominal_intensity',e.target.valueAsNumber)}/></label>
      <label>Spot radius (px)<input type="number" min="1" value={config.disturbance.beacon.spot_radius_px??config.camera.beacon_radius_px} onChange={e=>patch('disturbance.beacon.spot_radius_px',e.target.valueAsNumber)}/></label>
      <label>Boresight pan (rad)<input type="number" step="any" value={config.disturbance.boresight.pan_offset_rad} onChange={e=>patch('disturbance.boresight.pan_offset_rad',e.target.valueAsNumber)}/></label>
      <label>Boresight tilt (rad)<input type="number" step="any" value={config.disturbance.boresight.tilt_offset_rad} onChange={e=>patch('disturbance.boresight.tilt_offset_rad',e.target.valueAsNumber)}/></label>
      <label><input type="checkbox" checked={config.disturbance.blur} onChange={e=>patch('disturbance.blur',e.target.checked)}/> Blur</label>
      <label><input type="checkbox" checked={config.disturbance.gaussian_noise} onChange={e=>patch('disturbance.gaussian_noise',e.target.checked)}/> Gaussian noise</label>
    </div></details>

    {hasAtmosphere&&<details><summary>Atmosphere and background</summary><div className="config-grid">
      <label><input type="checkbox" checked={config.disturbance.atmosphere.enabled} onChange={e=>patch('disturbance.atmosphere.enabled',e.target.checked)}/> Atmosphere enabled</label>
      <label>Attenuation<input type="number" min="0" max="1" step="any" value={config.disturbance.atmosphere.attenuation} onChange={e=>patch('disturbance.atmosphere.attenuation',e.target.valueAsNumber)}/></label>
      <label>Cloud attenuation<input type="number" min="0" max="1" step="any" value={config.disturbance.atmosphere.cloud_attenuation} onChange={e=>patch('disturbance.atmosphere.cloud_attenuation',e.target.valueAsNumber)}/></label>
      <label>Turbulence strength<input type="number" min="0" max="1" step="any" value={config.disturbance.atmosphere.turbulence_strength} onChange={e=>patch('disturbance.atmosphere.turbulence_strength',e.target.valueAsNumber)}/></label>
      <label>Sky brightness<input type="number" min="0" max="255" value={config.disturbance.atmosphere.environment_brightness} onChange={e=>patch('disturbance.atmosphere.environment_brightness',e.target.valueAsNumber)}/></label>
    </div></details>}

    <details><summary>Failures and gimbal</summary><div className="config-grid">
      <label>Random beacon dropout<input type="number" min="0" max="1" step="any" value={config.disturbance.beacon.random_dropout_probability} onChange={e=>patch('disturbance.beacon.random_dropout_probability',e.target.valueAsNumber)}/></label>
      <label>Frame dropout<input type="number" min="0" max="1" step="any" value={config.disturbance.sensor.frame_dropout_probability} onChange={e=>patch('disturbance.sensor.frame_dropout_probability',e.target.valueAsNumber)}/></label>
      <label><input type="checkbox" checked={config.disturbance.distractors.enabled} onChange={e=>patch('disturbance.distractors.enabled',e.target.checked)}/> False sources</label>
      <label>False-source count<input type="number" min="0" max="20" value={config.disturbance.distractors.count} onChange={e=>patch('disturbance.distractors.count',e.target.valueAsNumber)}/></label>
      <label>Lock threshold (px)<input type="number" min="0.1" value={config.lock.lock_error_threshold_px} onChange={e=>patch('lock.lock_error_threshold_px',e.target.valueAsNumber)}/></label>
      <label>Lock frames<input type="number" min="1" value={config.lock.lock_required_frames} onChange={e=>patch('lock.lock_required_frames',e.target.valueAsNumber)}/></label>
    </div></details>

    <details>
      <summary>User presets and resolved configuration</summary>
      <div className="preset-tools">
        <input value={presetName} onChange={e=>setPresetName(e.target.value)}/>
        <button onClick={()=>{saveAll({...saved,[presetName]:structuredClone(config)});setSelectedSaved(presetName)}}>Save configuration</button>
        <button onClick={()=>{const name=`${presetName} copy`;saveAll({...saved,[name]:structuredClone(config)});setSelectedSaved(name)}}>Duplicate</button>
        <button disabled={!selectedSaved} onClick={()=>{
          if(selectedSaved){
            const next={...saved},value=next[selectedSaved]
            delete next[selectedSaved];next[presetName]=value;saveAll(next);setSelectedSaved(presetName)
          }
        }}>Rename selected</button>
        <button onClick={()=>void chooseProfile('SAT_SAT_NOMINAL')}>Reset to default</button>
        <select value={selectedSaved} onChange={e=>{
          setSelectedSaved(e.target.value)
          if(saved[e.target.value]){setConfig(structuredClone(saved[e.target.value]));setPresetName(e.target.value)}
        }}>
          <option value="">Load saved…</option>{Object.keys(saved).map(x=><option key={x}>{x}</option>)}
        </select>
      </div>
      <pre>{JSON.stringify(config,null,2)}</pre>
      <button className="config-export" onClick={()=>download(`pat-config-${config.scenario_preset}-${config.seed}.json`,config)}>Export rerunnable configuration JSON</button>
    </details>

    <details><summary>Monte Carlo distribution</summary><div className="config-grid">
      <label>Variable<select value={distributionPath} onChange={e=>setDistributionPath(e.target.value)}><option value="disturbance.vibration.stochastic_sigma_rad">Vibration sigma · rad</option><option value="camera.noise_sigma">Image noise · DN</option><option value="disturbance.beacon.nominal_intensity">Beacon intensity · DN</option><option value="disturbance.beacon.random_dropout_probability">Dropout probability</option><option value="disturbance.gimbal.command_latency_s">Gimbal latency · s</option></select></label>
      <label>Sampling<select value={distributionMode} onChange={e=>setDistributionMode(e.target.value as typeof distributionMode)}><option>fixed</option><option>uniform</option><option>normal</option></select></label>
      <label>{distributionMode==='normal'?'Mean':distributionMode==='uniform'?'Low':'Value'}<input type="number" step="any" value={distributionA} onChange={e=>setDistributionA(e.target.valueAsNumber)}/></label>
      {distributionMode!=='fixed'&&<label>{distributionMode==='normal'?'Standard deviation':'High'}<input type="number" step="any" value={distributionB} onChange={e=>setDistributionB(e.target.valueAsNumber)}/></label>}
      <label><input type="checkbox" checked={summaryOnly} onChange={e=>setSummaryOnly(e.target.checked)}/> Summary-only storage</label>
    </div></details>

    <div className="run-controls"><button disabled={busy} onClick={()=>void live()}>Live run</button><button disabled={busy} onClick={()=>void batch(1)}>Single headless</button><label>Runs<input type="number" min="1" max="10000" value={runs} onChange={e=>setRuns(e.target.valueAsNumber)}/></label><button disabled={busy} onClick={()=>void batch()}>Monte Carlo</button><button disabled={busy} onClick={()=>void compare()}>Compare A/B</button><button disabled={busy} onClick={()=>void history()}>Compatible history</button></div>
    {message&&<p role="alert">{message}</p>}{result&&<details open><summary>Results · {result.mode}</summary><pre>{JSON.stringify(result.aggregate??result,null,2)}</pre></details>}
  </section>
}
