import random

from fly_pilot.initial_conditions import DECODER_SPAWN, SpawnRandomization, sample_spawn
from fly_pilot.runway import Runway


def test_seeded_spawn_is_reproducible() -> None:
    rwy = Runway()
    a = sample_spawn(rwy, seed=7)
    b = sample_spawn(rwy, seed=7)
    c = sample_spawn(rwy, seed=8)
    assert a == b
    assert a.right_m != c.right_m or a.agl_m != c.agl_m or a.heading_deg != c.heading_deg
    assert a.seed == 7


def test_spawn_ranges_are_modest() -> None:
    rwy = Runway()
    spec = SpawnRandomization()
    rng = random.Random(0)
    for i in range(200):
        s = sample_spawn(rwy, rng=rng, spec=spec)
        assert 3050.0 <= s.distance_m <= 3350.0
        assert -80.0 <= s.right_m <= 80.0
        assert 220.0 <= s.agl_m <= 280.0
        assert 65.0 <= s.airspeed_kts <= 75.0
        assert abs(s.heading_deg - 90.0) <= 8.0
        assert abs(s.roll_deg) <= 8.0


def test_decoder_spawn_is_wider_and_seeded() -> None:
    rwy = Runway()
    a = sample_spawn(rwy, seed=11, spec=DECODER_SPAWN)
    b = sample_spawn(rwy, seed=11, spec=DECODER_SPAWN)
    assert a == b
    spec = DECODER_SPAWN
    rng = random.Random(0)
    seen_right = False
    for _ in range(400):
        s = sample_spawn(rwy, rng=rng, spec=spec)
        assert 2950.0 <= s.distance_m <= 3450.0
        assert -140.0 <= s.right_m <= 140.0
        assert 200.0 <= s.agl_m <= 300.0
        assert 62.0 <= s.airspeed_kts <= 78.0
        assert abs(s.heading_deg - 90.0) <= 12.0
        assert abs(s.roll_deg) <= 14.0
        if abs(s.right_m) > 80.0:
            seen_right = True
    assert seen_right
