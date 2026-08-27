import ObserverScene from './ObserverScene'
import { useTelemetry } from './useTelemetry'

const degrees = (radians?: number) => (radians === undefined ? '—' : `${(radians * 180 / Math.PI).toFixed(2)}°`)
const value = (number?: number | null, suffix = '') => number === undefined || number === null ? '—' : `${number.toFixed(2)}${suffix}`

export default function App() {
  const { telemetry, connected } = useTelemetry()

  return (
    <main>
      <header>
        <div>
          <p className="eyebrow">SIH26169 · TEAM DOOMSDAY SQUAD</p>
          <h1>Virtual Camera Tracking Lab</h1>
        </div>
        <div className={`connection ${connected ? 'online' : ''}`}>
          <span /> {connected ? 'LIVE TELEMETRY' : 'CONNECTING'}
        </div>
      </header>

      <section className="workspace">
        <article className="panel observer-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">OBSERVER VIEW</p>
              <h2>Interactive 3D relative motion</h2>
            </div>
            <p className="hint">Drag to orbit · Scroll to zoom · Right-drag to pan</p>
          </div>
          <div className="scene"><ObserverScene telemetry={telemetry} /></div>
        </article>

        <article className="panel camera-panel">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">TRACKING CAMERA</p>
              <h2>Pixels seen by OpenCV</h2>
            </div>
            <span className={`lock ${telemetry?.locked ? 'locked' : ''}`}>{telemetry?.locked ? 'LOCKED' : 'ACQUIRING'}</span>
          </div>
          <div className="camera-frame">
            <img src="/api/camera.mjpeg" alt="Live synthetic tracking camera feed" />
          </div>
          <div className="legend">
            <span><i className="opencv" />OpenCV measurement</span>
            <span><i className="kalman" />Kalman estimate</span>
            <span><i className="axis" />Optical axis</span>
          </div>
        </article>
      </section>

      <section className="telemetry">
        <article><p>Tracking error</p><strong>{value(telemetry?.tracking_error_px, ' px')}</strong><small>Kalman estimate to image centre</small></article>
        <article><p>CV accuracy</p><strong>{value(telemetry?.detection_error_px, ' px')}</strong><small>OpenCV against hidden ground truth</small></article>
        <article><p>Camera attitude</p><strong>{degrees(telemetry?.camera_pan_rad)} / {degrees(telemetry?.camera_tilt_rad)}</strong><small>Pan / tilt</small></article>
        <article><p>Hill position</p><strong>{telemetry ? telemetry.target_hill_position_m.map(n => n.toFixed(2)).join(', ') : '—'}</strong><small>Radial, along-track, cross-track · m</small></article>
        <article><p>Simulated time</p><strong>{value(telemetry?.simulation_time_s, ' s')}</strong><small>Orbital time accelerated 60×</small></article>
      </section>

      <footer>
        <span>Pipeline</span>
        CW physics <b>→</b> camera pixels <b>→</b> OpenCV <b>→</b> Kalman <b>→</b> PID <b>→</b> pan/tilt actuator
      </footer>
    </main>
  )
}

