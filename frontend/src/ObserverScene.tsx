import { Canvas } from '@react-three/fiber'
import { Grid, Html, Line, OrbitControls, Stars } from '@react-three/drei'
import type { Telemetry } from './types'

function Satellite({position=[0,0,0],target=false}:{position?:[number,number,number],target?:boolean}){
  const body=target?'#d8a34c':'#94a9b7',panel=target?'#5b3f19':'#173a59'
  return <group position={position} scale={target ? .72 : 1}>
    <mesh><boxGeometry args={[.72,.42,.85]}/><meshStandardMaterial color={body} metalness={.72} roughness={.28}/></mesh>
    <mesh position={[0,0,.52]} rotation={[Math.PI/2,0,0]}><cylinderGeometry args={[.17,.25,.3,24]}/><meshStandardMaterial color="#142631" metalness={.62} roughness={.22}/></mesh>
    <mesh position={[-1.05,0,0]}><boxGeometry args={[1.25,.035,.58]}/><meshStandardMaterial color={panel} metalness={.25} roughness={.32}/></mesh>
    <mesh position={[1.05,0,0]}><boxGeometry args={[1.25,.035,.58]}/><meshStandardMaterial color={panel} metalness={.25} roughness={.32}/></mesh>
    {[-1.05,1.05].map(x=><group key={x}>{[-.24,0,.24].map(z=><Line key={z} points={[[x-.6,.022,z],[x+.6,.022,z]]} color="#7194b0" lineWidth={.45}/>)}</group>)}
    <mesh position={[0,0,.73]}><torusGeometry args={[.22,.035,10,28]}/><meshStandardMaterial color={target?'#ffd166':'#48d5c2'} emissive={target?'#ff9f1c':'#11766b'} emissiveIntensity={1.4}/></mesh>
  </group>
}

function FieldOfView({telemetry}:{telemetry:Telemetry}){
  const range=7
  const halfWidth=Math.tan(telemetry.fov_deg[0]*Math.PI/360)*range
  const halfHeight=Math.tan(telemetry.fov_deg[1]*Math.PI/360)*range
  const origin:[number,number,number]=[0,0,.72]
  const corners:[number,number,number][]=[[-halfWidth,-halfHeight,range],[halfWidth,-halfHeight,range],[halfWidth,halfHeight,range],[-halfWidth,halfHeight,range]]
  return <group rotation={[-telemetry.camera_tilt_rad,telemetry.camera_pan_rad,0]}>
    {corners.map((corner,index)=><Line key={index} points={[origin,corner]} color="#46dcc6" lineWidth={1} transparent opacity={.55}/>)}
    <Line points={[...corners,corners[0]]} color="#46dcc6" lineWidth={1} transparent opacity={.7}/>
    <Line points={[origin,[0,0,range]]} color="#f46f7d" lineWidth={1.5} transparent opacity={.86}/>
  </group>
}

function Scene({telemetry}:{telemetry:Telemetry}){
  const target=telemetry.target_world_m
  return <>
    <color attach="background" args={['#020711']}/><fog attach="fog" args={['#020711',28,90]}/>
    <ambientLight intensity={.48}/><directionalLight position={[8,10,-4]} intensity={2.2} color="#d8ebff"/>
    <pointLight position={target} intensity={5} distance={8} color="#ffbd59"/>
    <Stars radius={85} depth={45} count={1500} factor={2.2} saturation={.1} fade speed={.18}/>
    <Grid args={[80,80]} position={[0,-2.2,8]} cellSize={1} cellThickness={.35} cellColor="#153044" sectionSize={5} sectionThickness={.8} sectionColor="#27546b" fadeDistance={48} infiniteGrid/>
    <axesHelper args={[3.2]}/>
    <Satellite/>
    <Html position={[0,.9,0]} center distanceFactor={11}><span className="scene-label observer-label">CHIEF / OBSERVER</span></Html>
    <FieldOfView telemetry={telemetry}/>
    <Line points={[[0,0,.72],target]} color="#f4b759" lineWidth={1.1} dashed dashSize={.2} gapSize={.13} transparent opacity={.54}/>
    {telemetry.trajectory_m.length>1&&<Line points={telemetry.trajectory_m} color="#69b8ff" lineWidth={2.2} transparent opacity={.82}/>}
    <Satellite position={target} target/>
    <Html position={[target[0],target[1]+.8,target[2]]} center distanceFactor={11}><span className="scene-label target-label">DEPUTY / TARGET</span></Html>
    <OrbitControls makeDefault enableDamping dampingFactor={.075} minDistance={4} maxDistance={58}/>
  </>
}

export default function ObserverScene({telemetry}:{telemetry:Telemetry|null}){
  return <Canvas dpr={[1,1.6]} camera={{position:[12,8,-14],fov:44,near:.1,far:180}} gl={{antialias:true}}>
    {telemetry&&<Scene telemetry={telemetry}/>}
  </Canvas>
}
