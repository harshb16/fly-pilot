export type EpisodeStatus =
  | "in_progress"
  | "landed"
  | "crashed"
  | "out_of_bounds"
  | "failed_approach";

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
  controller: string;
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
    note: string;
  };
}

export interface StateMessage {
  type: "state";
  sim_time: number;
  paused: boolean;
  controller: string;
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
  };
  controls: {
    aileron: number;
    elevator: number;
    rudder: number;
    throttle: number;
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
}

export type ServerMessage = HelloMessage | StateMessage;

export interface PilotControls {
  aileron: number;
  elevator: number;
  rudder: number;
  throttle: number;
}
