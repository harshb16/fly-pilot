export type EpisodeStatus =
  | "in_progress"
  | "landed"
  | "crashed"
  | "out_of_bounds"
  | "failed_approach";

export type ControllerName = "manual" | "expert";

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
    expert_is_biological?: boolean;
    note: string;
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

export interface StateMessage {
  type: "state";
  sim_time: number;
  paused: boolean;
  controller: ControllerName | string;
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
}

export type ServerMessage = HelloMessage | StateMessage;

export interface PilotControls {
  aileron: number;
  elevator: number;
  rudder: number;
  throttle: number;
}
