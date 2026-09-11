"""Visual observation path for MaleCNS (Milestone 4).

Rendered frames → cubemap atlas → R1–R6 currents. The fly does not control
the aircraft.
"""

from fly_pilot.brain.vision.encoder import EncoderConfig, FlyEyeEncoder
from fly_pilot.brain.vision.geometry import CameraCalibration
from fly_pilot.brain.vision.mapping import PhotoreceptorMap
from fly_pilot.brain.vision.scene import FlyViewPose, render_cubemap_atlas
from fly_pilot.brain.vision.stimulus import RETINAL_SCHEMA, RetinalStimulusFrame

__all__ = [
    "CameraCalibration",
    "EncoderConfig",
    "FlyEyeEncoder",
    "FlyViewPose",
    "PhotoreceptorMap",
    "RETINAL_SCHEMA",
    "RetinalStimulusFrame",
    "render_cubemap_atlas",
]
