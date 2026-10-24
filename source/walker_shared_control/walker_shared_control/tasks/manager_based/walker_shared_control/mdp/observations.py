import torch


SLOPE_RAD = 0.13962634   # 8 degrees

UP_START = 3.0
UP_END = 6.96

DOWN_START = 9.0
DOWN_END = 12.96

PREVIEW_DISTANCE = 1.0


def _robot_x(env):
    robot = env.scene["robot"]

    # Position relative to each cloned environment origin
    return (
        robot.data.root_link_pos_w[:, 0]
        - env.scene.env_origins[:, 0]
    )


def _slope_at_x(x):
    slope = torch.zeros_like(x)

    # Uphill
    slope = torch.where(
        (x >= UP_START) & (x <= UP_END),
        torch.full_like(x, SLOPE_RAD),
        slope,
    )

    # Downhill
    slope = torch.where(
        (x >= DOWN_START) & (x <= DOWN_END),
        torch.full_like(x, -SLOPE_RAD),
        slope,
    )

    return slope


def slope_now(env):
    x = _robot_x(env)

    return _slope_at_x(x).unsqueeze(-1)


def slope_ahead(env):
    x = _robot_x(env) + PREVIEW_DISTANCE

    return _slope_at_x(x).unsqueeze(-1)

def zero_slope_ahead(env):
    return torch.zeros_like(slope_now(env))

def forward_velocity(env):
    robot = env.scene["robot"]

    return robot.data.root_link_vel_w[:, 0].unsqueeze(-1)

def human_force(env):
    action_term = env.action_manager.get_term(
        "admittance_damping"
    )

    return action_term.human_force_voluntary.unsqueeze(-1)

def current_damping(env):
    action_term = env.action_manager.get_term(
        "admittance_damping"
    )

    B = action_term.B_drive
    return ((B - 60.0) / 30.0).unsqueeze(-1)