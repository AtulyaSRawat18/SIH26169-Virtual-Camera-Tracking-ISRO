export type Telemetry = {
  estimator: string
  predictor: string
  predictor_name: string
  predictor_latency_ms: number | null
  predictor_fallback: boolean
  predictor_failure_reason: string | null
  prediction_horizon_s: number
  predicted_pixel: [number,number] | null
  correction_algorithm: string
  correction_vector_px: [number,number] | null
  correction_latency_ms: number | null
  correction_fallback: boolean
  correction_failure_reason: string | null
  corrected_pixel: [number,number] | null
  vision_algorithm: string
  controller_name: string
  controller_latency_ms: number | null
  controller_deadline_missed: boolean
  controller_fallback: boolean
  controller_command_saturated: boolean
  controller_output: [number,number]
  search_active: boolean
  search_strategy: string
  search_mode: string | null
  search_fallback: boolean
  confirmation_valid: boolean
  optical_link_enabled: boolean
  pointing_error_rad: number | null
  pointing_loss_db: number | null
  received_power_dbm: number | null
  link_margin_db: number | null
  link_available: boolean
  link_state: string
  pat_link_combination: string
  optical_link: {range_m:number|null;range_mode:string|null;atmospheric_loss_db:number|null;communication_snr_db:number|null}|null
  error_budget: Record<string,number|number[]|null>
  error_waterfall: {error:string;stage:string;value:number;unit:string;change_from_previous:number|null;relative_change_percent:number|null;status:string|null}[]
  resolved_error_config: Record<string,unknown>
  pat_events: {timestamp_s:number;event:string;from?:string;reason:string}[]
  time_in_pat_state_s: number
  measurement_quality: {cnn_predicted_sigma_px?:number[]}|null
  time_scale: number
  scenario: string
  scenario_preset: string
  motion_model: string
  cw_case: string
  cw_bounded_residual_m_s: number | null
  seed: number
  lock_state: 'SEARCH' | 'ACQUIRE' | 'TRACK' | 'LOCKED' | 'LOST' | 'REACQUIRE'
  metrics: { rmse_tracking_error_px: number | null; locked_percentage: number; measurement_dropout_count: number; failure_counts: Record<string,number>; [key:string]:unknown }
  estimator_nis: number | null
  estimator_nees_position: number | null
  estimator_prediction_only: boolean
  estimator_r_scale: number | null
  estimator_latency_ms: number | null
  estimator_uncertainty_ellipse: {major_axis_px:number;minor_axis_px:number;orientation_deg:number;confidence_level:number}|null
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
  camera_size_px: [number,number]
  locked: boolean
  trajectory_m: [number, number, number][]
}

