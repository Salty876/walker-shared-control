import torch


def track_target_velocity(
    env,
    target_velocity: float = 0.35,
    sigma: float = 0.05,
):
    """Reward maintaining the desired walking velocity."""

    robot = env.scene["robot"]
    velocity = robot.data.root_link_lin_vel_b[:, 0]

    error = (velocity - target_velocity) / sigma

    return torch.exp(-torch.square(error))


def voluntary_effort(
    env,
    force_scale: float = 20.0,
):
    """Penalty proportional to squared human interaction force."""

    action_term = env.action_manager.get_term(
        "admittance_damping"
    )

    force = action_term.human_force_voluntary

    return -torch.square(force / force_scale)

def overspeed(
        env,
        target_velocity: float = 0.35,
        sigma: float = 0.10
        ):
    robot = env.scene["robot"]

    velocity = robot.data.root_link_lin_vel_b[:, 0]
    overspeed = torch.clamp(
        velocity - target_velocity,
        min=0.0,
    )

    return torch.square(overspeed / sigma)
