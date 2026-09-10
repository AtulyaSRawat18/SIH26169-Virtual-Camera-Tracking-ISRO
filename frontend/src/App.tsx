import { useState } from 'react'
import ObserverScene from './ObserverScene'
import ExperimentPanel from './ExperimentPanel'
import VisionLab from './VisionLab'
import EstimatorLab from './EstimatorLab'
import AICorrectionLab from './AICorrectionLab'
import TemporalPredictionLab from './TemporalPredictionLab'
import ControllerLab from './ControllerLab'
import AcquisitionLab from './AcquisitionLab'
import OpticalLinkPanel from './OpticalLinkPanel'
import ComparisonWorkbench from './ComparisonWorkbench'
import EvidenceDashboard from './EvidenceDashboard'
import LiveControlPanel from './LiveControlPanel'
import LiveErrorFlow from './LiveErrorFlow'
import SystemBrief from './SystemBrief'
import { useTelemetry } from './useTelemetry'

const degrees = (radians?: number) => (radians === undefined ? '—' : `${(radians * 180 / Math.PI).toFixed(2)}°`)
const value = (number?: number | null, suffix = '') => number === undefined || number === null ? '—' : `${number.toFixed(2)}${suffix}`

export default function App() {
  const { telemetry, connected } = useTelemetry()
  const [view,setView]=useState<'live'|'experiment'|'evidence'|'advanced'|'about'>('live')
  const [lab,setLab]=useState<'vision'|'estimation'|'cnn'|'prediction'|'control'|'acquisition'>('vision')
  const [cameraMode,setCameraMode]=useState<'annotated'|'raw'>('annotated')
  const [evaluationOverlay,setEvaluationOverlay]=useState(false)

  return (
    <main>
      <header className="app-header">
        <div>
          <p className="eyebrow">SIH26169 · TEAM DOOMSDAY SQUAD</p>
          <h1>Virtual Camera Tracking</h1>
        </div>
        <div className={`connection ${connected ? 'online' : ''}`}>
          <span /> {connected ? 'LIVE TELEMETRY' : 'CONNECTING'}
        </div>
      </header>
      <nav className="view-nav" aria-label="Dashboard sections">
        {([['live','Live demo'],['experiment','Experiment lab'],['evidence','Evidence'],['advanced','Advanced'],['about','About']] as const).map(([item,label])=><button key={item} className={view===item?'active':''} onClick={()=>setView(item)}>{label}</button>)}
      </nav>

      {view==='live'&&<>
      <section className="mission-summary"><div><span>ACTIVE SCENARIO</span><strong>{telemetry?.scenario_preset?.replaceAll('_',' ')??'Loading'}</strong></div><div><span>MOTION / CASE</span><strong>{telemetry?.motion_model?.replaceAll('_',' ')??'—'} · {telemetry?.cw_case?.replaceAll('_',' ')??'custom'}</strong></div><div><span>PIPELINE</span><strong>{telemetry?.vision_algorithm??'OpenCV'} / {telemetry?.estimator??'KF'} / {telemetry?.controller_name??'PID'}</strong></div><div><span>STATE</span><strong className={telemetry?.locked?'positive':''}>{telemetry?.lock_state??'Connecting'}</strong></div></section>
      <section className="workspace">
        <article className="panel observer-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">OBSERVER VIEW</p>
              <h2>Interactive 3D relative motion</h2>
            </div>
            <div className="view-badges"><span>LVLH FRAME</span><span className={telemetry?.cw_bounded_residual_m_s!=null&&Math.abs(telemetry.cw_bounded_residual_m_s)<1e-5?'good':''}>{telemetry?.cw_bounded_residual_m_s==null?'KINEMATIC':Math.abs(telemetry.cw_bounded_residual_m_s)<1e-5?'BOUNDED':'DRIFTING'}</span></div>
          </div>
          <div className="scene"><ObserverScene telemetry={telemetry} /><div className="scene-hud"><span><i className="axis-x"/>X radial</span><span><i className="axis-y"/>Y along-track</span><span><i className="axis-z"/>Z cross-track</span></div><div className="interaction-hint">DRAG ORBIT · SCROLL ZOOM · RIGHT-DRAG PAN</div></div>
        </article>

        <article className="panel camera-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">TRACKING CAMERA</p>
              <h2>Pixels seen by OpenCV</h2>
            </div>
            <div className="camera-modes"><button className={cameraMode==='raw'?'active':''} onClick={()=>setCameraMode('raw')}>RAW</button><button className={cameraMode==='annotated'?'active':''} onClick={()=>setCameraMode('annotated')}>ANNOTATED</button><button className={evaluationOverlay?'active evaluation':''} onClick={()=>setEvaluationOverlay(v=>!v)}>EVAL TRUTH</button><span className={`lock ${telemetry?.locked ? 'locked' : ''}`}>{telemetry?.lock_state ?? 'SEARCHING'}</span></div>
          </div>
          <div className="camera-frame">
            <img src={cameraMode==='raw'?'/api/camera-raw.mjpeg':'/api/camera.mjpeg'} alt={`${cameraMode} synthetic tracking camera feed`} />
            <div className="sensor-overlay" aria-hidden="true"><span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/><span className="sensor-id">VIRTUAL OPTICAL SENSOR · {telemetry?.camera_size_px?.join('×')??'—'}</span><span className="sensor-mode">{telemetry?.vision_algorithm??'OPENCV'} · {telemetry?.measurement_quality?'MEASURING':'WAITING'}</span><span className="sensor-error">ERR {value(telemetry?.tracking_error_px,' px')}</span></div>
            {evaluationOverlay&&telemetry?.ground_truth_pixel&&<span className="truth-marker" style={{left:`${telemetry.ground_truth_pixel[0]/telemetry.camera_size_px[0]*100}%`,top:`${telemetry.ground_truth_pixel[1]/telemetry.camera_size_px[1]*100}%`}}>GROUND TRUTH — EVALUATION ONLY</span>}
          </div>
          <div className="legend">
            <span><i className="opencv" />OpenCV measurement</span>
            <span><i className="kalman" />{telemetry?.estimator ?? 'kalman'} estimate</span>
            <span><i className="axis" />Optical axis</span>
          </div>
        </article>
      </section>

      <LiveControlPanel telemetry={telemetry}/>
      <LiveErrorFlow telemetry={telemetry}/>
      <OpticalLinkPanel telemetry={telemetry}/>

      <section className="telemetry">
        <article><p>Tracking error</p><strong>{value(telemetry?.tracking_error_px, ' px')}</strong><small>Estimate to image centre</small></article>
        <article><p>CV accuracy</p><strong>{value(telemetry?.detection_error_px, ' px')}</strong><small>OpenCV against hidden ground truth</small></article>
        <article><p>Camera attitude</p><strong>{degrees(telemetry?.camera_pan_rad)} / {degrees(telemetry?.camera_tilt_rad)}</strong><small>Pan / tilt</small></article>
        <article><p>Hill position</p><strong>{telemetry ? telemetry.target_hill_position_m.map(n => n.toFixed(2)).join(', ') : '—'}</strong><small>Radial, along-track, cross-track · m</small></article>
        <article><p>Simulated time</p><strong>{value(telemetry?.simulation_time_s, ' s')}</strong><small>Orbital time accelerated {telemetry?.time_scale ?? 60}×</small></article>
        <article><p>True tracking RMSE</p><strong>{value(telemetry?.metrics.rmse_tracking_error_px, ' px')}</strong><small>Projected truth to image centre · whole run</small></article>
        <article><p>Lock retention</p><strong>{value(telemetry?.metrics.locked_percentage, '%')}</strong><small>Dropout frames: {telemetry?.metrics.measurement_dropout_count ?? 0}</small></article>
      </section>
      </>}

      {view==='experiment'&&<section className="experiment-workspace"><ExperimentPanel /></section>}

      {view==='advanced'&&<section className="experiment-workspace">
        <div className="lab-switcher"><div><p className="panel-kicker">MODULE LABS</p><h2>Inspect one stage at a time</h2></div><div>{(['vision','estimation','cnn','prediction','control','acquisition'] as const).map(item=><button key={item} className={lab===item?'active':''} onClick={()=>setLab(item)}>{item}</button>)}</div></div>
        {lab==='vision'&&<VisionLab/>}{lab==='estimation'&&<EstimatorLab telemetry={telemetry}/>} {lab==='cnn'&&<AICorrectionLab telemetry={telemetry}/>} {lab==='prediction'&&<TemporalPredictionLab telemetry={telemetry}/>} {lab==='control'&&<ControllerLab telemetry={telemetry}/>} {lab==='acquisition'&&<AcquisitionLab telemetry={telemetry}/>}
      </section>}

      {view==='evidence'&&<><EvidenceDashboard telemetry={telemetry}/><ComparisonWorkbench telemetry={telemetry}/></>}
      {view==='about'&&<SystemBrief/>}

      <footer>
        <span>Pipeline</span>
        Physics <b>→</b> camera <b>→</b> {telemetry?.vision_algorithm??'OpenCV'} <b>→</b> {telemetry?.correction_algorithm??'none'} <b>→</b> {telemetry?.estimator??'kalman'} <b>→</b> {telemetry?.predictor_name??'none'} <b>→</b> {telemetry?.controller_name??'pid'} <b>→</b> gimbal <b>→</b> {telemetry?.search_strategy??'reacquisition'}
      </footer>
    </main>
  )
}

