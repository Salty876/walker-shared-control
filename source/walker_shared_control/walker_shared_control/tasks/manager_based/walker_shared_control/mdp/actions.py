from __future__ import annotations

import torch
import csv
import cv2
import numpy as np
import warp as wp
from pathlib import Path
from datetime import datetime


from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from .admittance import LongitudinalAdmittance

from . import observations as walker_obs
from . import camera_perception

class AdmittanceDampingAction(ActionTerm):
    """One RL action selects longitudinal damping B_drive."""

    cfg: "AdmittanceDampingActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.debug_counter = 0
        self._eval_episode = 0
        self._last_epispde_step = 0
        self._last_logged_episode = -1
        self._last_logged_episode_step = -1
        self._last_logged_key = None
        self._camera_debug_printed = False # Camera debug print flag
        self._last_valid_slope_cam = torch.zeros(
            (self.num_envs, 1), device=self.device
        )
        self.fixed_b_eval = False  # Set to True to evaluate with fixed B=60 N/(m/s)



        '''LOGGING STUFF'''
        self.enable_eval_logging = True  # Set to True to log evaluation data to CSV

        if self.enable_eval_logging:
            log_dir = Path("eval_logs")
            log_dir.mkdir(exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.fixed_b_eval:
                self.csv_path = log_dir / f"v3_fixed_b_{timestamp}.csv"
            else:
                self.csv_path = log_dir / f"v3_reactive_seed_3.csv"

            self.csv_file = open(self.csv_path, "w", newline="")
            self.csv_writer = csv.writer(self.csv_file)

            self.csv_writer.writerow([
                "episode",
                "episode_step",
                "episode_time_s",
                "x_m",
                "velocity_mps",
                "target_velocity_mps",
                "human_force_N",
                "human_voluntary_force_N",
                "human_gait_force_N",
                "slope_now_deg",
                "slope_ahead_deg",
                "oracle_slope_deg",
                "action",
                "B_drive",
            ])

            self.eval_step = 0

            print(f"[INFO] Logging evaluation to: {self.csv_path}")




        self._raw_actions = torch.zeros(
            (self.num_envs, 1), device=self.device
        )

        # Store actual B_drive here
        self._processed_actions = torch.zeros(
            (self.num_envs, 1), device=self.device
        )

        self._wheel_ids, self._wheel_names = self._asset.find_joints(
            list(cfg.wheel_names),
            preserve_order=True,
        )

        print("Walker wheel joints:", self._wheel_names)

        self._wheel_signs = torch.tensor(
            cfg.wheel_signs,
            dtype=torch.float32,
            device=self.device,
        ).view(1, 4)

        self.controller = LongitudinalAdmittance(
            num_envs=self.num_envs,
            device=self.device,
            dt=self._env.physics_dt,
            M_drive=cfg.M_drive,
            B_drive=cfg.B_nominal,
        )

        self._B_drive = torch.full(
            (self.num_envs,),
            cfg.B_nominal,
            device=self.device,
        )

        self._v_cmd = torch.zeros(
            self.num_envs,
            device=self.device,
        )

        self._human_force_voluntary = torch.full(
            (self.num_envs,),
            self.cfg.human_force_bias,
            device=self.device,
        )

        self._human_force_gait = torch.zeros(
            self._env.num_envs,
            device=self._env.device,
        )

        # Total interaction force 
        self._human_force = torch.full(
            (self.num_envs,),
            self.cfg.human_force_bias,
            device=self.device,
        )

        self._gait_freq_hz = torch.zeros(
            self._env.num_envs,
            device=self._env.device,
        )

        self._gait_phase0 = torch.zeros(
            self._env.num_envs,
            device=self._env.device,
        )

    def _randomize_gait(self, env_ids):
        if env_ids.numel() == 0:
            return

        n = env_ids.numel()
        device = self._env.device

        self._gait_freq_hz[env_ids] = (
            self.cfg.gait_freq_min_hertz
            + (
                self.cfg.gait_freq_max_hertz - self.cfg.gait_freq_min_hertz
            )
            * torch.rand(n, device=device)
        )

        self._gait_phase0[env_ids] = (
            2.0 * torch.pi * torch.rand(n, device=device)
        )


    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        return self._processed_actions

    @property
    def B_drive(self):
        return self._B_drive

    @property
    def v_cmd(self):
        return self._v_cmd

    @property
    def human_force(self):
        return self._human_force

    @property
    def human_force_voluntary(self):
        return self._human_force_voluntary

    @property
    def human_force_gait(self):
        return self._human_force_gait

    @property
    def gait_frequency(self):
        return self._gait_freq_hz

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        # PPO action [-1, 1]
        a = torch.clamp(actions[:, 0], -1.0, 1.0)

        # Map:
        # -1 -> B_min
        #  0 -> nominal B = 60
        # +1 -> B_max
        if self.fixed_b_eval:
            self._B_drive[:] = 60.0
        else:
            self._B_drive[:] = (
                self.cfg.B_min
                + 0.5
                * (a + 1.0)
                * (self.cfg.B_max - self.cfg.B_min)
            )

        self._processed_actions[:, 0] = self._B_drive

        # =========================================================================
        # V2 HUMAN MODEL    
        # =========================================================================
        # 1. Voluntary user force
        # The walker body frame uses +X as its forward direction.
        v_actual = self._asset.data.root_link_lin_vel_b[:, 0]

        force_desired = (
            self.cfg.human_force_bias
            + self.cfg.human_kp * (self.cfg.target_speed - v_actual)
        )

        force_desired = torch.clamp(
            force_desired,
            self.cfg.human_force_min,
            self.cfg.human_force_max,
        )

        max_change = (
            self.cfg.human_force_rate_limit * self._env.step_dt
        )

        force_delta = torch.clamp(
            force_desired - self._human_force_voluntary,
            -max_change,
            max_change
        )

        self._human_force_voluntary += force_delta


        # 2. Gait-periodic force

        episode_time = (
            self._env.episode_length_buf.to(torch.float32) *
            self._env.step_dt
        )

        phase = (
            2.0
            * torch.pi
            * self._gait_freq_hz
            * episode_time
            + self._gait_phase0
        )

        self._human_force_gait = (
            self.cfg.gait_amplitude_1 * torch.sin(phase) + self.cfg.gait_amplitude_2 * torch.sin(2.0 * phase)
        )

        # 3. Total physical interaction force

        self._human_force = (
            self._human_force_voluntary + self._human_force_gait
        )
        self._human_force = torch.clamp(
            self._human_force,
            self.cfg.human_force_min,
            self.cfg.human_force_max
        )


        self.debug_counter += 1
        slope_ahead = camera_perception.slope_ahead_camera_safe(self._env)[0, 0]
        raw = camera_perception.slope_ahead_camera(self._env)[0, 0]
        if self.debug_counter % 25 == 0:
            # print(
            #     f"[HUMAN V2] "
            #     f"v={v_actual[0].item():.3f} | "
            #     f"Fdes={force_desired[0].item():.2f} | "
            #     f"Fvol={self._human_force_voluntary[0].item():.2f} | "
            #     f"Fgait={self._human_force_gait[0].item():+.2f} | "
            #     f"Ftotal={self._human_force[0].item():.2f} | "
            #     f"B={self._B_drive[0].item():.2f}"
            # )
            print(f"ZED SLOPE:{slope_ahead}");
            print(f"ZED RAW:{raw}")

        # more ZED debug


    def apply_actions(self):
        # apply_actions() runs once per physics step, so the controller uses
        # self._env.physics_dt while holding the control-rate force and damping.
        self._v_cmd, _ = self.controller.step(
            f_drive=self._human_force,
            B_drive=self._B_drive,
        )

        wheel_speed = self._v_cmd / self.cfg.wheel_radius

        wheel_targets = (
            wheel_speed.unsqueeze(1) * self._wheel_signs
        )

        self._asset.set_joint_velocity_target(
            wheel_targets,
            joint_ids=self._wheel_ids,
        )

        episode_step = int(self._env.episode_length_buf[0].item())

        if episode_step < self._last_epispde_step:
            self._eval_episode += 1

        self._last_epispde_step = episode_step
        
        x = (
                self._asset.data.root_link_pos_w[0, 0]
                - self._env.scene.env_origins[0, 0]
            )

        v = self._asset.data.root_link_lin_vel_b[0, 0]


        # if self.debug_counter % 25 == 0:
            # slope_cam = camera_perception.slope_ahead_camera(self._env)
            # slope_gt = walker_obs.slope_ahead(self._env)

            # if torch.isfinite(slope_cam).all():
            #     self._last_valid_slope_cam.copy_(slope_cam)
            # else:
            #     slope_cam = self._last_valid_slope_cam.clone()

            # print(
            #     f"[CAMERA DEBUG] x={x.item():.3f} m, "
            #     f"slope_cam={torch.rad2deg(slope_cam).item():.1f} deg, "
            #     f"slope_gt={torch.rad2deg(slope_gt).item():.1f} deg"
            # )

            # camera = self._env.scene["camera"]
            # robot = self._env.scene["robot"]

            # cam_pos = camera.data.pos_w.torch[0]
            # robot_pos = robot.data.root_pos_w.torch[0]

            # print(
            #     f"robot_x={robot_pos[0].item():.3f}, "
            #     f"camera_x={cam_pos[0].item():.3f}, "
            #     f"dx={(cam_pos[0] - robot_pos[0]).item():.3f}"
            # )

        slope_now = walker_obs.slope_now(self._env)[0, 0]
        slope_ahead = camera_perception.slope_ahead_camera_safe(self._env)[0, 0]
        slope_oracle = walker_obs.slope_ahead(self._env)[0, 0]

        # Camera debug print
        

        if not self._camera_debug_printed:

            # points = camera_perception.depth_to_base_points(self._env)

            # print("\n=== POINT CLOUD DEBUG ===")
            # print("Number of points:", points.shape[0])

            # print(
            #     "X forward:",
            #     points[:, 0].min().item(),
            #     points[:, 0].max().item(),
            # )

            # print(
            #     "Y lateral:",
            #     points[:, 1].min().item(),
            #     points[:, 1].max().item(),
            # )

            # print(
            #     "Z height:",
            #     points[:, 2].min().item(),
            #     points[:, 2].max().item(),
            # )

            # print("=========================\n")

            # camera_perception.debug_terrain_profile(self._env)


            # camera = self._env.scene["camera"]
            # mx = camera.data.intrinsic_matrices.torch
            # # intrinsic matrices
            # print("\n=== CAMERA INTRINSICS ===")
            # print(mx)
            # print("shape:", mx.shape)
            # print("=============================")

            # depth_wp = camera.data.output["distance_to_image_plane"]

            # depth = wp.to_torch(depth_wp)

            # depth_img = depth[0, :, :, 0]

            # finite = torch.isfinite(depth_img)

            # if finite.any():
            #     print("Finite min:", depth_img[finite].min().item())
            #     print("Finite max:", depth_img[finite].max().item())
            #     print("Finite mean:", depth_img[finite].mean().item())

            # vis = depth_img.clone()

            # vis[~finite] = 5.0  # Set non-finite values to a specific value for visualization

            # vis = torch.clamp(vis, 0.0, 5.0)  # Clamp values for better visualization

            # vis = (vis / 5.0 * 255).to(torch.uint8).to(torch.uint8)

            # vis_np = vis.cpu().numpy()

            # vis_np = 255 - vis_np  # Invert for better visualization

            # cv2.imwrite(f"depth_debug.png", vis_np)
            # print("Saved depth image to depth_debug.png")

            # print("\n=== CAMERA DEBUG ===")
            # print("Depth shape:", depth.shape)
            # print("Depth dtype:", depth.dtype)
            # print("Depth device:", depth.device)
            # print("Depth min:", depth.min().item())
            # print("Depth max:", depth.max().item())
            self._camera_debug_printed = True

            s = camera_perception.slope_ahead_camera(self._env)

            print(
                f"shape={s.shape}, ",
                "[CAMERA DEBUG] env_0 slope_ahead_camera="
                f"{torch.rad2deg(s[0, 0]).item():.2f} deg"
            )

        if self.enable_eval_logging:

            log_key = (self._eval_episode, episode_step)

            if log_key != self._last_logged_key:
                self.csv_writer.writerow([
                    self._eval_episode,
                    episode_step,
                    episode_step * 0.02,
                    x.item(),
                    v.item(),
                    self.cfg.target_speed,
                    self._human_force[0].item(),
                    self._human_force_voluntary[0].item(),
                    self._human_force_gait[0].item(),
                    torch.rad2deg(slope_now).item(),
                    torch.rad2deg(slope_ahead).item(),
                    torch.rad2deg(slope_oracle).item(),
                    self._raw_actions[0, 0].item(),
                    self._B_drive[0].item(),
                ])

            self._last_logged_key = log_key
            self.eval_step += 1
            self._last_logged_episode = self._eval_episode
            self._last_logged_episode_step = episode_step

            if self.eval_step % 250 == 0:
                print(f"[INFO] Logged {self.eval_step} evaluation steps.")
                print(f"ZED SLOPE:{slope_ahead}");

                # More zed debug
                


            # print(
            #     f"x={x.item():.3f} "
            #     f"v={v.item():.3f} "
            #     f"F={self._human_force[0].item():.2f} "
            #     f"B={self._B_drive[0].item():.1f}"
            #     f" slope_now={torch.rad2deg(slope_now).item():.1f} "
            #     f" slope_ahead={torch.rad2deg(slope_ahead).item():.1f}"
            # )

    def reset(self, env_ids=None):
        self.controller.reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._v_cmd.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._v_cmd[env_ids] = 0.0

        self._B_drive[env_ids] = 60.0 if self.fixed_b_eval else self.cfg.B_nominal
        self._human_force_voluntary[env_ids] = self.cfg.human_force_bias
        self._human_force_gait[env_ids] = 0.0
        self._human_force[env_ids] = self.cfg.human_force_bias
        self._randomize_gait(env_ids)


@configclass
class AdmittanceDampingActionCfg(ActionTermCfg):

    class_type: type[ActionTerm] = AdmittanceDampingAction

    wheel_names: tuple[str, ...] = (
        "fr_wheel",
        "fl_wheel",
        "rl_wheel",
        "rr_wheel",
    )

    # correct these if  joint axes are flipped.
    wheel_signs: tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        1.0,
    )

    M_drive: float = 50.0

    B_min: float = 30.0
    B_nominal: float = 60.0
    B_max: float = 90.0

    target_speed: float = 0.35

    human_force_bias: float = 20.0
    human_kp: float = 50.0
    human_force_rate_limit: float = 400.0

    human_force_min: float = 0.0
    human_force_max: float = 60.0

    wheel_radius: float = 0.10

    # v2 gait-like interaction force stuff
    gait_freq_min_hertz: float = 1.0
    gait_freq_max_hertz: float = 1.5

    gait_amplitude_1: float = 2.0 # Fundemental N
    gait_amplitude_2: float = 0.5 # Second harmonic N
