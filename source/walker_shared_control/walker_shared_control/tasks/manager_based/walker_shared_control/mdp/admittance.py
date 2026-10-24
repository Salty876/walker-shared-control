import torch


class LongitudinalAdmittance:
    """Vectorized copy of the ROS longitudinal admittance controller."""

    def __init__(
        self,
        num_envs: int,
        device: str,
        dt: float = 0.02,
        M_drive: float = 50.0,
        B_drive: float = 60.0,
        deadband: float = 1.0,
        impulse_threshold: float = 800.0,
    ):
        self.num_envs = num_envs
        self.device = device

        self.dt = dt
        self.default_M = M_drive
        self.default_B = B_drive

        self.deadband = deadband
        self.impulse_threshold = impulse_threshold

        self.v_x = torch.zeros(num_envs, device=device)
        self.prev_f_drive = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.v_x.zero_()
            self.prev_f_drive.zero_()
        else:
            self.v_x[env_ids] = 0.0
            self.prev_f_drive[env_ids] = 0.0

    def step(
        self,
        f_drive: torch.Tensor,
        B_drive=None,
        M_drive=None,
    ):
        # ROS deadband:
        # abs(force) < 1 N -> 0
        f_drive = torch.where(
            torch.abs(f_drive) < self.deadband,
            torch.zeros_like(f_drive),
            f_drive,
        )

        # Allow one M/B value per simulated walker.
        if B_drive is None:
            B_drive = torch.full_like(
                f_drive, self.default_B
            )

        if M_drive is None:
            M_drive = torch.full_like(
                f_drive, self.default_M
            )

        # Same impulse check as ROS controller
        dF_drive = (
            f_drive - self.prev_f_drive
        ) / self.dt

        safety_halt = (
            torch.abs(dF_drive)
            > self.impulse_threshold
        )

        # Exact longitudinal admittance update from ROS
        dv = (
            f_drive - B_drive * self.v_x
        ) / M_drive

        new_v = self.v_x + dv * self.dt

        # Same behavior as ROS safety halt
        self.v_x = torch.where(
            safety_halt,
            torch.zeros_like(new_v),
            new_v,
        )

        self.prev_f_drive = f_drive.clone()

        return self.v_x, safety_halt
