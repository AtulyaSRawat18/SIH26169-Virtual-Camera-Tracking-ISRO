const groups=[
  {name:'Simulation and physics',items:[['Python','Simulation runtime'],['NumPy','Vectors, matrices and integration'],['Clohessy–Wiltshire + RK4','Relative orbital motion'],['Pinhole camera','3D position to image pixels']]},
  {name:'Vision and estimation',items:[['OpenCV','Synthetic sensor and spot localization'],['KF / EKF / UKF','Noise filtering and uncertainty'],['PyTorch','Experimental CNN and GRU/LSTM'],['Guard rails','Fallback on latency, uncertainty or OOD input']]},
  {name:'Control and interface',items:[['PID + FF-PID','Pan and tilt commands'],['Actuator model','Rate limits and response lag'],['FastAPI + WebSocket','Live simulation and telemetry'],['React + Three.js','Dashboard and interactive observer']]},
  {name:'Engineering evidence',items:[['TypeScript + Vite','Typed, reproducible web build'],['pytest','Physics and pipeline verification'],['JSON + SQLite','Configurations and experiment records'],['Git / GitHub','Versioned team collaboration']]},
]
const usps=[
  ['End-to-end PAT loop','One reproducible system connects orbital motion, image formation, localization, estimation, prediction, control and actuator response.'],
  ['Fair algorithm comparison','Methods receive identical frames and seeds. Hidden ground truth scores outcomes but never steers the operational pipeline.'],
  ['Failure-aware experiments','Cloud loss, glare, distractors, vibration, dropout and actuator limits reveal where an algorithm stops working.'],
  ['Evidence-gated AI','CNN and temporal models earn promotion only when they improve held-out, OOD and closed-loop results without unsafe latency.'],
]

export default function SystemBrief(){
  return <section className="system-brief">
    <div className="brief-hero">
      <div><p className="panel-kicker">PROJECT BRIEF</p><h2>Virtual pointing, acquisition and tracking testbed</h2><p>The prototype lets a team develop camera-tracking algorithms against measurable orbital and sensor conditions before hardware integration.</p></div>
      <dl><div><dt>Problem</dt><dd>Reliable optical target tracking under motion and visual disturbance</dd></div><div><dt>Output</dt><dd>Camera command, lock state, uncertainty and reproducible performance evidence</dd></div></dl>
    </div>
    <div className="section-intro"><div><p className="panel-kicker">CURRENT STACK</p><h2>Technology mapped to responsibility</h2></div><span className="status-note">IMPLEMENTED UNLESS MARKED EXPERIMENTAL</span></div>
    <div className="stack-grid">{groups.map(group=><article className="stack-group" key={group.name}><h3>{group.name}</h3>{group.items.map(([tech,role])=><div key={tech}><strong>{tech}</strong><span>{role}</span></div>)}</article>)}</div>
    <div className="section-intro"><div><p className="panel-kicker">PROPOSED VALUE</p><h2>What is distinct in this implementation</h2></div></div>
    <div className="usp-grid">{usps.map(([title,body],index)=><article key={title}><span>0{index+1}</span><h3>{title}</h3><p>{body}</p></article>)}</div>
    <div className="reality-grid"><article><p className="panel-kicker">FEASIBILITY</p><h3>Runs on a standard laptop</h3><p>Python supplies the simulation and API. The browser supplies 3D interaction and telemetry. No specialist hardware is required for the digital prototype.</p></article><article><p className="panel-kicker">KNOWN LIMIT</p><h3>Simulation does not certify flight hardware</h3><p>Sensor calibration, thermal behavior, structural vibration and real optics still require hardware-in-loop and field validation.</p></article></div>
    <div className="section-intro"><div><p className="panel-kicker">IMPLEMENTATION STATUS</p><h2>No unavailable module is presented as functional</h2></div></div>
    <div className="status-grid"><article><span className="status-stable">STABLE / TESTED</span><p>Physics, OpenCV, estimators, classical controllers, gimbal, PAT search, FastAPI, Three.js, Monte Carlo and evidence export.</p></article><article><span className="status-experimental">EXPERIMENTAL</span><p>CNN correction, GRU/LSTM prediction and MPC remain selectable with safety fallbacks. They are not the startup path.</p></article><article><span className="status-unavailable">NOT IMPLEMENTED</span><p>Flight hardware, fine steering, qualified atmosphere, calibrated BER receiver, Transformers and reinforcement learning.</p></article></div>
  </section>
}
