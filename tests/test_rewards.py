from __future__ import annotations

from chrono_a2rl.common.types import TrackState, VehicleCommand, VehicleState
from chrono_a2rl.rl.rewards import (
    compute_corner_braking_reward,
    compute_reward,
    compute_terminal_reward,
)


def test_reward_prefers_faster_progress() -> None:
    cfg = {
        "progress_weight": 1.25,
        "speed_reward_weight": 0.08,
        "target_speed_reward_weight": 0.04,
        "max_reward_speed": 81.94,
        "time_penalty_per_step": 0.003,
    }
    track_state = TrackState(on_track=True, distance_left_boundary=5.0, distance_right_boundary=5.0)
    command = VehicleCommand(throttle_target=0.5)

    slow = compute_reward(
        progress_delta=0.2,
        state=VehicleState(speed=10.0),
        track_state=track_state,
        command=command,
        target_speed=20.0,
        config=cfg,
    )
    fast = compute_reward(
        progress_delta=0.8,
        state=VehicleState(speed=40.0),
        track_state=track_state,
        command=command,
        target_speed=60.0,
        config=cfg,
    )

    assert fast > slow


def test_profile_tracking_reward_prefers_matching_actual_speed() -> None:
    cfg = {
        "profile_speed_tracking_reward_weight": 0.35,
        "profile_speed_error_penalty_weight": 0.50,
        "profile_speed_error_scale_kmh": 15.0,
        "positive_reward_progress_distance": 0.20,
    }
    track_state = TrackState(
        on_track=True,
        distance_left_boundary=5.0,
        distance_right_boundary=5.0,
    )

    matched = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=40.0),
        track_state=track_state,
        command=VehicleCommand(throttle_target=0.5),
        target_speed=40.0,
        config=cfg,
    )
    overspeed = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=50.0),
        track_state=track_state,
        command=VehicleCommand(throttle_target=0.5),
        target_speed=40.0,
        config=cfg,
    )

    assert matched > overspeed


def test_corner_entry_prefers_controlled_deceleration_over_no_braking() -> None:
    cfg = {
        "max_reward_speed": 81.94,
        "corner_speed_cap_margin_fraction": 0.08,
        "corner_overspeed_penalty_weight": 3.0,
        "corner_controlled_braking_reward_weight": 0.60,
        "corner_braking_target_decel": 6.0,
        "corner_braking_max_decel": 10.0,
        "corner_braking_max_command": 0.65,
        "corner_excessive_braking_penalty_weight": 1.0,
    }
    coasting = compute_corner_braking_reward(
        speed=60.0,
        future_speed_cap=30.0,
        braking_demand=1.0,
        actual_deceleration=0.0,
        brake_command=0.0,
        trail_brake_target=0.45,
        line_safety=1.0,
        config=cfg,
    )
    controlled = compute_corner_braking_reward(
        speed=60.0,
        future_speed_cap=30.0,
        braking_demand=1.0,
        actual_deceleration=6.0,
        brake_command=0.45,
        trail_brake_target=0.45,
        line_safety=1.0,
        config=cfg,
    )

    assert controlled.total > coasting.total
    assert controlled.controlled_braking_reward > 0.0


def test_corner_entry_penalizes_excessive_braking() -> None:
    cfg = {
        "max_reward_speed": 81.94,
        "corner_controlled_braking_reward_weight": 0.60,
        "corner_braking_target_decel": 6.0,
        "corner_braking_max_decel": 10.0,
        "corner_braking_max_command": 0.65,
        "corner_excessive_braking_penalty_weight": 1.0,
    }
    controlled = compute_corner_braking_reward(
        speed=60.0,
        future_speed_cap=30.0,
        braking_demand=1.0,
        actual_deceleration=6.0,
        brake_command=0.45,
        trail_brake_target=0.45,
        line_safety=1.0,
        config=cfg,
    )
    excessive = compute_corner_braking_reward(
        speed=60.0,
        future_speed_cap=30.0,
        braking_demand=1.0,
        actual_deceleration=16.0,
        brake_command=1.0,
        trail_brake_target=0.45,
        line_safety=1.0,
        config=cfg,
    )

    assert excessive.total < controlled.total
    assert excessive.excessive_braking_penalty > 0.0


def test_corner_braking_shaping_is_inactive_on_clear_straight() -> None:
    shaping = compute_corner_braking_reward(
        speed=70.0,
        future_speed_cap=30.0,
        braking_demand=0.0,
        actual_deceleration=6.0,
        brake_command=0.5,
        line_safety=1.0,
        config={"corner_overspeed_penalty_weight": 3.0},
    )

    assert shaping.total == 0.0


def test_drag_only_deceleration_cannot_earn_controlled_braking_reward() -> None:
    shaping = compute_corner_braking_reward(
        speed=60.0,
        future_speed_cap=30.0,
        braking_demand=1.0,
        actual_deceleration=6.0,
        brake_command=0.0,
        trail_brake_target=0.45,
        line_safety=1.0,
        config={"corner_controlled_braking_reward_weight": 0.60},
    )

    assert shaping.controlled_braking_reward == 0.0
    assert shaping.missing_brake_penalty > 0.0


def test_previous_full_throttle_trace_gets_overspeed_and_missing_brake_costs() -> None:
    shaping = compute_corner_braking_reward(
        speed=76.0,
        future_speed_cap=20.0,
        braking_demand=1.0,
        actual_deceleration=7.0,
        brake_command=0.0,
        trail_brake_target=0.55,
        line_safety=1.0,
        config={
            "max_reward_speed": 81.94,
            "corner_overspeed_penalty_weight": 3.0,
            "corner_controlled_braking_reward_weight": 0.60,
        },
    )

    assert shaping.controlled_braking_reward == 0.0
    assert shaping.overspeed_penalty > 0.0
    assert shaping.missing_brake_penalty > 0.0
    assert shaping.total < 0.0


def test_missing_small_reference_remains_a_dense_penalty() -> None:
    shaping = compute_corner_braking_reward(
        speed=45.0,
        future_speed_cap=30.0,
        braking_demand=0.2,
        actual_deceleration=2.0,
        brake_command=0.0,
        trail_brake_target=0.05,
        line_safety=1.0,
    )

    assert shaping.missing_brake_penalty == 1.25
    assert shaping.alignment_reward == 0.0


def test_matched_moderate_braking_beats_coasting_and_full_braking() -> None:
    cfg = {
        "max_reward_speed": 81.94,
        "corner_controlled_braking_reward_weight": 0.60,
        "corner_braking_target_decel": 6.0,
        "corner_braking_max_decel": 10.0,
        "corner_braking_max_command": 0.65,
        "corner_excessive_braking_penalty_weight": 1.0,
        "trail_braking_alignment_reward_weight": 0.35,
        "trail_braking_missing_penalty_weight": 1.25,
        "trail_braking_excess_reference_penalty_weight": 0.75,
    }
    common = {
        "speed": 60.0,
        "future_speed_cap": 30.0,
        "braking_demand": 1.0,
        "trail_brake_target": 0.45,
        "line_safety": 1.0,
        "config": cfg,
    }

    coasting = compute_corner_braking_reward(
        actual_deceleration=6.0,
        brake_command=0.0,
        **common,
    )
    matched = compute_corner_braking_reward(
        actual_deceleration=6.0,
        brake_command=0.45,
        **common,
    )
    full = compute_corner_braking_reward(
        actual_deceleration=16.0,
        brake_command=1.0,
        **common,
    )

    assert matched.total > coasting.total
    assert matched.total > full.total
    assert matched.alignment_error == 0.0


def test_stationary_car_cannot_collect_apex_or_speed_reward() -> None:
    cfg = {
        "progress_weight": 1.25,
        "speed_reward_weight": 0.08,
        "straight_speed_reward_weight": 0.12,
        "apex_line_reward_weight": 0.16,
        "apex_speed_reward_weight": 0.08,
        "positive_reward_progress_distance": 0.05,
        "stationary_penalty_per_step": 0.25,
        "time_penalty_per_step": 0.015,
        "max_reward_speed": 81.94,
    }
    track_state = TrackState(
        on_track=True,
        distance_left_boundary=5.0,
        distance_right_boundary=5.0,
    )

    parked = compute_reward(
        progress_delta=0.0,
        state=VehicleState(speed=0.0),
        track_state=track_state,
        command=VehicleCommand(),
        target_speed=40.0,
        apex_strength=1.0,
        config=cfg,
    )
    moving = compute_reward(
        progress_delta=0.2,
        state=VehicleState(speed=10.0),
        track_state=track_state,
        command=VehicleCommand(throttle_target=0.3),
        target_speed=40.0,
        apex_strength=1.0,
        config=cfg,
    )

    assert parked < 0.0
    assert moving > parked


def test_straight_reward_prefers_high_target_speed_at_same_progress() -> None:
    cfg = {
        "max_reward_speed": 81.94,
        "straight_target_speed_reward_weight": 0.30,
        "straight_min_target_speed_fraction": 0.92,
        "straight_low_target_speed_penalty": 1.50,
        "straight_min_actual_speed_fraction": 0.70,
        "straight_low_actual_speed_penalty": 0.25,
    }
    track_state = TrackState(on_track=True, distance_left_boundary=5.0, distance_right_boundary=5.0)
    command = VehicleCommand(throttle_target=1.0)

    low_target = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=45.0),
        track_state=track_state,
        command=command,
        target_speed=45.0,
        apex_strength=0.0,
        config=cfg,
    )
    high_target = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=45.0),
        track_state=track_state,
        command=command,
        target_speed=81.94,
        apex_strength=0.0,
        config=cfg,
    )

    assert high_target > low_target


def test_unachieved_target_speed_does_not_earn_extra_positive_reward() -> None:
    cfg = {
        "max_reward_speed": 80.0,
        "target_speed_reward_weight": 1.0,
        "straight_target_speed_reward_weight": 1.0,
    }
    track_state = TrackState(on_track=True, distance_left_boundary=5.0, distance_right_boundary=5.0)
    command = VehicleCommand(throttle_target=1.0)

    reachable_target = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=40.0),
        track_state=track_state,
        command=command,
        target_speed=40.0,
        config=cfg,
    )
    unreachable_target = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=40.0),
        track_state=track_state,
        command=command,
        target_speed=80.0,
        config=cfg,
    )

    assert unreachable_target == reachable_target


def test_straight_speed_pressure_can_be_reduced_for_upcoming_corner() -> None:
    cfg = {
        "max_reward_speed": 81.94,
        "straight_min_target_speed_fraction": 0.92,
        "straight_low_target_speed_penalty": 1.50,
        "straight_min_actual_speed_fraction": 0.70,
        "straight_low_actual_speed_penalty": 0.25,
    }
    track_state = TrackState(on_track=True, distance_left_boundary=5.0, distance_right_boundary=5.0)
    command = VehicleCommand(throttle_target=0.4)

    straight_low_speed = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=30.0),
        track_state=track_state,
        command=command,
        target_speed=45.0,
        apex_strength=0.0,
        config=cfg,
    )
    corner_ahead_low_speed = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=30.0),
        track_state=track_state,
        command=command,
        target_speed=45.0,
        apex_strength=0.0,
        speed_pressure_strength=0.0,
        config=cfg,
    )

    assert corner_ahead_low_speed > straight_low_speed


def test_terminal_reward_completion_beats_timeout() -> None:
    cfg = {"completion_bonus": 500.0, "timeout_penalty": 150.0}
    completed = compute_terminal_reward(
        completed=True,
        offtrack=False,
        timeout=False,
        progress_fraction=1.0,
        config=cfg,
    )
    timed_out = compute_terminal_reward(
        completed=False,
        offtrack=False,
        timeout=True,
        progress_fraction=0.5,
        config=cfg,
    )

    assert completed > 0.0
    assert timed_out < 0.0
    assert completed > timed_out


def test_terminal_reward_penalizes_stalling() -> None:
    reward = compute_terminal_reward(
        completed=False,
        offtrack=False,
        timeout=False,
        stalled=True,
        progress_fraction=0.1,
        config={"stall_terminal_penalty": 600.0},
    )

    assert reward == -600.0


def test_early_high_speed_offtrack_is_strongly_penalized() -> None:
    cfg = {
        "offtrack_terminal_penalty": 250.0,
        "offtrack_remaining_lap_penalty": 600.0,
        "offtrack_speed_penalty": 250.0,
    }
    early_fast_crash = compute_terminal_reward(
        completed=False,
        offtrack=True,
        timeout=False,
        progress_fraction=0.08,
        speed_fraction=0.8,
        config=cfg,
    )
    later_slow_crash = compute_terminal_reward(
        completed=False,
        offtrack=True,
        timeout=False,
        progress_fraction=0.60,
        speed_fraction=0.3,
        config=cfg,
    )

    assert early_fast_crash < -900.0
    assert early_fast_crash < later_slow_crash


def test_kinetic_crash_penalty_grows_quadratically() -> None:
    cfg = {
        "offtrack_terminal_penalty": 400.0,
        "offtrack_kinetic_penalty": 1500.0,
        "offtrack_corner_penalty": 400.0,
    }
    slow_crash = compute_terminal_reward(
        completed=False,
        offtrack=True,
        timeout=False,
        progress_fraction=0.2,
        speed_fraction=0.3,
        corner_failure=True,
        config=cfg,
    )
    fast_crash = compute_terminal_reward(
        completed=False,
        offtrack=True,
        timeout=False,
        progress_fraction=0.2,
        speed_fraction=0.9,
        corner_failure=True,
        config=cfg,
    )

    assert fast_crash < slow_crash - 1000.0


def test_safe_achieved_speed_progress_requires_line_and_boundary_margin() -> None:
    cfg = {
        "max_reward_speed": 80.0,
        "safe_speed_progress_weight": 0.5,
        "safe_speed_boundary_margin": 1.5,
        "racing_line_error_scale": 1.5,
    }
    command = VehicleCommand(throttle_target=0.5)
    clean = compute_reward(
        progress_delta=1.0,
        state=VehicleState(speed=60.0),
        track_state=TrackState(
            n=0.0,
            on_track=True,
            distance_left_boundary=5.0,
            distance_right_boundary=5.0,
        ),
        command=command,
        racing_line_offset=0.0,
        config=cfg,
    )
    boundary_risk = compute_reward(
        progress_delta=1.0,
        state=VehicleState(speed=60.0),
        track_state=TrackState(
            n=0.0,
            on_track=True,
            distance_left_boundary=0.1,
            distance_right_boundary=5.0,
        ),
        command=command,
        racing_line_offset=0.0,
        config=cfg,
    )

    assert clean > boundary_risk


def test_offtrack_step_cannot_collect_progress_or_speed_reward() -> None:
    reward = compute_reward(
        progress_delta=10.0,
        state=VehicleState(speed=70.0),
        track_state=TrackState(on_track=False),
        command=VehicleCommand(throttle_target=1.0),
        target_speed=80.0,
        config={
            "progress_weight": 10.0,
            "speed_reward_weight": 10.0,
            "target_speed_reward_weight": 10.0,
            "straight_speed_reward_weight": 10.0,
            "straight_target_speed_reward_weight": 10.0,
            "offtrack_penalty": 20.0,
        },
    )

    assert reward < 0.0


def test_curb_overuse_penalty_escalates_with_fraction_and_streak() -> None:
    cfg = {
        "curb_penalty_scale": 1.5,
        "curb_high_speed_penalty_scale": 1.0,
        "curb_overuse_fraction_limit": 0.03,
        "curb_overuse_penalty_scale": 8.0,
        "curb_streak_time_limit": 0.35,
        "curb_streak_penalty_scale": 4.0,
        "max_reward_speed": 81.94,
    }
    track_state = TrackState(
        on_track=True,
        on_curb=True,
        curb_penalty_weight=0.2,
        distance_left_boundary=0.3,
        distance_right_boundary=10.0,
    )
    command = VehicleCommand(throttle_target=0.4)

    brief_touch = compute_reward(
        progress_delta=0.0,
        state=VehicleState(speed=30.0),
        track_state=track_state,
        command=command,
        curb_usage_fraction=0.01,
        curb_streak_time=0.1,
        config=cfg,
    )
    overuse = compute_reward(
        progress_delta=0.0,
        state=VehicleState(speed=30.0),
        track_state=track_state,
        command=command,
        curb_usage_fraction=0.4,
        curb_streak_time=1.0,
        config=cfg,
    )

    assert overuse < brief_touch - 4.0


def test_past_curb_usage_is_not_charged_again_after_leaving_curb() -> None:
    cfg = {
        "curb_overuse_fraction_limit": 0.03,
        "curb_overuse_penalty_scale": 8.0,
        "curb_streak_time_limit": 0.35,
        "curb_streak_penalty_scale": 4.0,
    }
    command = VehicleCommand()
    clean_history = compute_reward(
        progress_delta=0.0,
        state=VehicleState(),
        track_state=TrackState(on_track=True, on_curb=False),
        command=command,
        curb_usage_fraction=0.0,
        curb_streak_time=0.0,
        config=cfg,
    )
    curb_history = compute_reward(
        progress_delta=0.0,
        state=VehicleState(),
        track_state=TrackState(on_track=True, on_curb=False),
        command=command,
        curb_usage_fraction=0.5,
        curb_streak_time=2.0,
        config=cfg,
    )

    assert curb_history == clean_history


def test_terminal_reward_penalizes_curb_overuse() -> None:
    cfg = {
        "completion_bonus": 500.0,
        "curb_overuse_fraction_limit": 0.03,
        "curb_overuse_terminal_penalty": 120.0,
        "curb_streak_time_limit": 0.35,
        "curb_streak_terminal_penalty": 50.0,
    }

    clean = compute_terminal_reward(
        completed=True,
        offtrack=False,
        timeout=False,
        progress_fraction=1.0,
        curb_usage_fraction=0.01,
        max_curb_streak_time=0.1,
        config=cfg,
    )
    curb_riding = compute_terminal_reward(
        completed=True,
        offtrack=False,
        timeout=False,
        progress_fraction=1.0,
        curb_usage_fraction=0.4,
        max_curb_streak_time=1.0,
        config=cfg,
    )

    assert curb_riding < clean


def test_apex_reward_prefers_fast_clean_racing_line() -> None:
    cfg = {
        "max_reward_speed": 80.0,
        "racing_line_error_penalty": 0.18,
        "racing_line_error_scale": 1.5,
        "target_offset_penalty": 0.10,
        "target_offset_error_scale": 0.75,
        "apex_line_reward_weight": 0.12,
        "apex_speed_reward_weight": 0.22,
    }
    command = VehicleCommand(throttle_target=0.5)
    clean_apex = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=50.0),
        track_state=TrackState(n=1.0, on_track=True, distance_left_boundary=5.0, distance_right_boundary=5.0),
        command=command,
        racing_line_offset=1.0,
        target_lateral_offset=1.0,
        apex_strength=1.0,
        config=cfg,
    )
    slow_missed_apex = compute_reward(
        progress_delta=0.5,
        state=VehicleState(speed=25.0),
        track_state=TrackState(n=3.0, on_track=True, distance_left_boundary=5.0, distance_right_boundary=5.0),
        command=command,
        racing_line_offset=1.0,
        target_lateral_offset=3.0,
        apex_strength=1.0,
        config=cfg,
    )

    assert clean_apex > slow_missed_apex


def test_apex_speed_reward_is_gated_near_boundary() -> None:
    cfg = {
        "max_reward_speed": 80.0,
        "racing_line_error_scale": 1.5,
        "apex_speed_reward_weight": 0.2,
        "apex_boundary_margin": 1.2,
        "apex_boundary_penalty": 0.25,
    }
    command = VehicleCommand(throttle_target=0.5)
    safe = compute_reward(
        progress_delta=0.0,
        state=VehicleState(speed=50.0),
        track_state=TrackState(n=1.0, on_track=True, distance_left_boundary=4.0, distance_right_boundary=4.0),
        command=command,
        racing_line_offset=1.0,
        target_lateral_offset=1.0,
        apex_strength=1.0,
        config=cfg,
    )
    crowding_boundary = compute_reward(
        progress_delta=0.0,
        state=VehicleState(speed=50.0),
        track_state=TrackState(n=1.0, on_track=True, distance_left_boundary=0.15, distance_right_boundary=4.0),
        command=command,
        racing_line_offset=1.0,
        target_lateral_offset=1.0,
        apex_strength=1.0,
        config=cfg,
    )

    assert safe > crowding_boundary
