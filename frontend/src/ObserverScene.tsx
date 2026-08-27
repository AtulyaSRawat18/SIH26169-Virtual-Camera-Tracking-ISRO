import { Canvas } from '@react-three/fiber'
import { Grid, Line, OrbitControls, Stars } from '@react-three/drei'
import type { Telemetry } from './types'

function FieldOfView({ telemetry }: { telemetry: Telemetry }) {
  const range = 5
  const halfWidth = Math.tan((telemetry.fov_deg[0] * Math.PI) / 360) * range
  const halfHeight = Math.tan((telemetry.fov_deg[1] * Math.PI) / 360) * range
  const origin: [number, number, number] = [0, 0, 0]
  const corners: [number, number, number][] = [
    [-halfWidth, -halfHeight, range],
    [halfWidth, -halfHeight, range],
    [halfWidth, halfHeight, range],
    [-halfWidth, halfHeight, range],
  ]

  return (
    <group rotation={[-telemetry.camera_tilt_rad, telemetry.camera_pan_rad, 0]}>
      {corners.map((corner, index) => (
        <Line key={index} points={[origin, corner]} color="#22b8cf" lineWidth={1} transparent opacity={0.55} />
      ))}
      <Line points={[...corners, corners[0]]} color="#22b8cf" lineWidth={1} transparent opacity={0.55} />
    </group>
  )
}

function Scene({ telemetry }: { telemetry: Telemetry }) {
  return (
    <>
      <color attach="background" args={['#050b12']} />
      <ambientLight intensity={0.6} />
      <pointLight position={[6, 8, -4]} intensity={35} color="#b8e1ff" />
      <Stars radius={70} depth={30} count={900} factor={2} saturation={0} fade speed={0.2} />
      <Grid
        args={[24, 24]}
        position={[0, -2, 5]}
        cellSize={1}
        cellThickness={0.5}
        cellColor="#234050"
        sectionSize={5}
        sectionThickness={1}
        sectionColor="#315f72"
        fadeDistance={24}
        infiniteGrid
      />
      <axesHelper args={[2.5]} />

      <group>
        <mesh>
          <boxGeometry args={[0.55, 0.35, 0.8]} />
          <meshStandardMaterial color="#7994a5" metalness={0.7} roughness={0.35} />
        </mesh>
        <mesh position={[0, 0, 0.5]}>
          <cylinderGeometry args={[0.18, 0.28, 0.35, 20]} />
          <meshStandardMaterial color="#142b3a" metalness={0.5} roughness={0.25} />
        </mesh>
      </group>

      <FieldOfView telemetry={telemetry} />
      <Line points={[[0, 0, 0], telemetry.target_world_m]} color="#ffd166" lineWidth={1.5} transparent opacity={0.7} />
      {telemetry.trajectory_m.length > 1 && (
        <Line points={telemetry.trajectory_m} color="#7bdff2" lineWidth={2} transparent opacity={0.75} />
      )}
      <mesh position={telemetry.target_world_m}>
        <sphereGeometry args={[0.18, 24, 24]} />
        <meshStandardMaterial color="#ffd166" emissive="#ff9f1c" emissiveIntensity={2.5} />
      </mesh>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} minDistance={4} maxDistance={35} />
    </>
  )
}

export default function ObserverScene({ telemetry }: { telemetry: Telemetry | null }) {
  return (
    <Canvas camera={{ position: [11, 7, -12], fov: 46, near: 0.1, far: 150 }}>
      {telemetry && <Scene telemetry={telemetry} />}
    </Canvas>
  )
}

