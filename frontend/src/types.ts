export type Telemetry = {
  frame: number
  simulation_time_s: number
  target_world_m: [number, number, number]
  target_hill_position_m: [number, number, number]
  target_hill_velocity_m_s: [number, number, number]
  ground_truth_pixel: [number, number] | null
  opencv_pixel: [number, number] | null
  kalman_pixel: [number, number] | null
  detection_error_px: number | null
  tracking_error_px: number | null
  camera_pan_rad: number
  camera_tilt_rad: number
  camera_pan_rate_rad_s: number
  camera_tilt_rate_rad_s: number
  fov_deg: [number, number]
  locked: boolean
  trajectory_m: [number, number, number][]
}

