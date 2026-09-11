export type EpisodeStatus =
  | "in_progress"
  | "landed"
  | "crashed"
  | "out_of_bounds"
  | "failed_approach";

export type ControllerName = "manual" | "expert" | "expert_observing";

export interface RunwayInfo {
  name: string;
  lat_deg: number;
  lon_deg: number;
  alt_m: number;
  heading_deg: number;
  length_m: number;
  width_m: number;
}

export interface HelloMessage {
  type: "hello";
  milestone: number;
  controller: ControllerName | string;
  control_authority?: string;
  physics: string;
  aircraft: string;
  runway: RunwayInfo;
  approach: {
    distance_m: number;
    agl_m: number;
    airspeed_kts: number;
    flight_path_deg: number;
  };
  integrity: {
    authoritative_physics: string;
    visuals: string;
    male_cns: boolean;
    male_cns_observing?: boolean;
    fly_controls_aircraft?: boolean;
    expert_is_biological?: boolean;
    note: string;
  };
  schedule?: {
    physics_hz: number;
    vision_hz: number;
    neural_hz: number;
    state_broadcast_hz: number;
  };
}

export interface ExpertTelemetry {
  kind: string;
  label?: string;
  male_cns?: boolean;
  phase: string;
  target_airspeed_kts: number;
  glideslope_error_deg: number;
  glideslope_error_m: number;
  centerline_error_m: number;
  heading_error_deg?: number;
}

export interface FlyObservingTelemetry {
  controlling: boolean;
  label: string;
  mode?: string;
  neural_step?: number | null;
  sim_time_s?: number | null;
  n_spikes: number;
  spikes_per_sec: number;
  n_outgoing_edges?: number;
  spike_checksum?: string;
  n_r1r6: number;
  n_descending: number;
  retina: {
    mean_luminance: number;
    mean_current: number;
    mean_temporal: number;
    left_mean_luminance: number;
    right_mean_luminance: number;
    current_sha256?: string;
  };
  descending: {
    n_descending: number;
    n_left: number;
    n_right: number;
    mean_hz: number;
    left_mean_hz: number;
    right_mean_hz: number;
    max_hz: number;
    window_steps: number;
  };
  visual_rates_hz?: Record<string, number>;
}

export interface StateMessage {
  type: "state";
  sim_time: number;
  paused: boolean;
  controller: ControllerName | string;
  control_authority?: string;
  observing?: boolean;
  spawn_seed?: number | null;
  position: {
    lat_deg: number;
    lon_deg: number;
    alt_msl_m: number;
    alt_agl_m: number;
    east_m: number;
    north_m: number;
    up_m: number;
    along_m: number;
    right_m: number;
  };
  velocity: {
    airspeed_kts: number;
    groundspeed_kts: number;
    vertical_speed_fpm: number;
  };
  attitude: {
    pitch_deg: number;
    roll_deg: number;
    heading_deg: number;
    alpha_deg: number;
    p_deg_s?: number;
    q_deg_s?: number;
    r_deg_s?: number;
    beta_deg?: number;
  };
  controls: {
    aileron: number;
    elevator: number;
    rudder: number;
    throttle: number;
  };
  surfaces?: {
    elevator_pos: number;
    aileron_pos: number;
    rudder_pos: number;
    throttle_pos: number;
  };
  gear: {
    on_ground: boolean;
    wow: boolean[];
  };
  episode: {
    status: EpisodeStatus;
    reason: string | null;
    touchdown_fpm: number | null;
  };
  expert?: ExpertTelemetry;
  fly_observing?: FlyObservingTelemetry;
}

export type ServerMessage = HelloMessage | StateMessage;

export interface PilotControls {
  aileron: number;
  elevator: number;
  rudder: number;
  throttle: number;
}
