import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

def depth_to_base_points(env):
    """Convert every environment's depth image to world and base points.

    Returns tensors shaped ``[num_envs, height * width, 3]`` plus a
    ``[num_envs, height * width]`` mask identifying valid depth pixels.
    """

    camera = env.scene["camera"]
    robot = env.scene["robot"]

    # ---------------------------------------------------------
    # Camera data
    # ---------------------------------------------------------

    depth = camera.data.output["distance_to_image_plane"].torch
    K = camera.data.intrinsic_matrices.torch

    # Camera pose in world
    cam_pos_w = camera.data.pos_w.torch
    cam_quat_w = camera.data.quat_w_ros.torch

    # Robot/base pose in world
    robot_pos_w = robot.data.root_pos_w.torch
    robot_quat_w = robot.data.root_quat_w.torch

    depth = depth[..., 0]
    num_envs, height, width = depth.shape

    fx = K[:, 0, 0].view(num_envs, 1, 1)
    fy = K[:, 1, 1].view(num_envs, 1, 1)
    cx = K[:, 0, 2].view(num_envs, 1, 1)
    cy = K[:, 1, 2].view(num_envs, 1, 1)

    # ---------------------------------------------------------
    # Pixel coordinates
    # ---------------------------------------------------------

    v, u = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )

    # ---------------------------------------------------------
    # Remove invalid depth
    # ---------------------------------------------------------

    valid_depth = (
        torch.isfinite(depth)
        & (depth > 0.1)
        & (depth < 5.0)
    )
    safe_depth = torch.where(valid_depth, depth, torch.zeros_like(depth))

    # ---------------------------------------------------------
    # Unproject pixels into ROS optical camera coordinates
    #
    # camera:
    #   X = right
    #   Y = down
    #   Z = forward
    # ---------------------------------------------------------

    x_cam = (u.unsqueeze(0) - cx) * safe_depth / fx
    y_cam = (v.unsqueeze(0) - cy) * safe_depth / fy
    z_cam = safe_depth

    points_cam = torch.stack(
        (x_cam, y_cam, z_cam),
        dim=-1,
    )

    # ---------------------------------------------------------
    # Camera -> world
    # ---------------------------------------------------------

    points_cam = points_cam.reshape(num_envs, -1, 3)
    num_pixels = points_cam.shape[1]

    cam_quat = cam_quat_w[:, None, :].expand(-1, num_pixels, -1)
    points_w = quat_apply(
        cam_quat.reshape(-1, 4), points_cam.reshape(-1, 3)
    ).reshape(num_envs, num_pixels, 3)
    points_w = points_w + cam_pos_w[:, None, :]

    # ---------------------------------------------------------
    # World -> walker/base frame
    # ---------------------------------------------------------

    points_relative_w = points_w - robot_pos_w[:, None, :]
    robot_quat = robot_quat_w[:, None, :].expand(-1, num_pixels, -1)
    points_base = quat_apply_inverse(
        robot_quat.reshape(-1, 4), points_relative_w.reshape(-1, 3)
    ).reshape(num_envs, num_pixels, 3)

    return points_base, points_w, valid_depth.reshape(num_envs, num_pixels)


def slope_ahead_camera(env):
    """Estimate the gravity-referenced terrain slope about 1 m ahead.

    A NaN result means that the camera did not provide enough usable terrain
    points.  Callers can then distinguish an unavailable estimate from a
    genuine zero-degree slope.
    """
    points_base, points_w, valid_depth = depth_to_base_points(env)

    x_base = points_base[..., 0]
    y_base = points_base[..., 1]

    # Start near the same 1 m preview point used by slope_gt. Widen the
    # lateral region first, then the longitudinal region if the image is
    # sparse or part of the terrain is outside the camera view.
    candidate_masks = (
        valid_depth & (x_base > 0.75) & (x_base < 1.25) & (torch.abs(y_base) < 0.30),
        valid_depth & (x_base > 0.75) & (x_base < 1.25) & (torch.abs(y_base) < 0.50),
        valid_depth & (x_base > 0.65) & (x_base < 1.40) & (torch.abs(y_base) < 0.60),
    )

    narrow_mask, lateral_mask, wide_mask = candidate_masks
    narrow_count = narrow_mask.sum(dim=1)
    lateral_count = lateral_mask.sum(dim=1)

    # Select the narrowest usable ROI independently for every environment.
    mask = torch.where(
        (lateral_count >= 20).unsqueeze(1), lateral_mask, wide_mask
    )
    mask = torch.where((narrow_count >= 20).unsqueeze(1), narrow_mask, mask)
    num_points = mask.sum(dim=1)

    # ZED cam debug rq

    robot = env.scene["robot"]
    robot_x = robot.data.root_pos_w[0, 0].item()
    if env.num_envs == 1:
        total_valid = valid_depth.sum(dim=1)

        print(
            "ZED ROI |"
            f"x={robot_x:.2f} "
            f"all={total_valid[0].item()} "
            f"narrow={narrow_count[0].item()} "
            f"lateral={lateral_count[0].item()} "
            f"wide={wide_mask[0].sum().item()} "
            f"selected={num_points[0].item()} "
        )


    # Use consistent, gravity-aligned coordinates so this angle is directly
    # comparable with the course's hard-coded world-frame slope.
    x_world = points_w[..., 0]
    z_world = points_w[..., 2]

    # Center the data before fitting z = slope*x + intercept. This is the
    # closed-form least-squares solution and avoids a poorly scaled intercept.
    weights = mask.to(dtype=x_world.dtype)
    safe_count = num_points.clamp_min(1).to(dtype=x_world.dtype)
    x_mean = torch.sum(x_world * weights, dim=1) / safe_count
    z_mean = torch.sum(z_world * weights, dim=1) / safe_count
    x_centered = x_world - x_mean.unsqueeze(1)
    z_centered = z_world - z_mean.unsqueeze(1)
    denominator = torch.sum(x_centered.square() * weights, dim=1)
    numerator = torch.sum(x_centered * z_centered * weights, dim=1)


    # Continuance of ZED camera debug

    fit_valid = (num_points >=10) & (denominator >= 1.0e-6)

    if env.num_envs == 1 and not fit_valid[0]:
        print(
            ">>>> ZED FIT FAILED AGAIN"
            f"x={robot_x:.2f} "
            f"points={num_points[0].item()} "
            f"denom={denominator[0].item():.8f}"
        )

    fit_valid = (num_points >= 10) & (denominator >= 1.0e-6)
    slope = numerator / denominator.clamp_min(1.0e-6)
    slope_rad = torch.where(
        fit_valid,
        torch.atan(slope),
        torch.full_like(slope, torch.nan),
    )

    # Don't let perception produce values wildly
    # outside PPO's original training range.
    max_slope = torch.deg2rad(
        torch.tensor(
            8.0,
            device=slope_rad.device,
            dtype=slope_rad.dtype,
        )
    )

    slope_rad = torch.clamp(
        slope_rad,
        -max_slope,
        max_slope,
    )

    return slope_rad.unsqueeze(1)


def debug_zed(env):
    left = env.scene["zed_left"]
    right = env.scene["zed_right"]

    left_rgb = left.data.output["rgb"].torch
    right_rgb = right.data.output["rgb"].torch
    left_depth = left.data.output["distance_to_image_plane"].torch

    left_pos = left.data.pos_w.torch
    right_pos = right.data.pos_w.torch

    baseline = torch.linalg.norm(left_pos - right_pos, dim=-1)

    print("\n=== ZED DEBUG ===")
    print("left rgb:   ", left_rgb.shape)
    print("right rgb:  ", right_rgb.shape)
    print("left depth: ", left_depth.shape)
    print("left K:     ", left.data.intrinsic_matrices.torch.shape)
    print("baseline:   ", baseline[:5])
    print("=================\n")

def debug_terrain_profile(env):
    points, _, valid_depth = depth_to_base_points(env)

    # This helper intentionally prints only environment 0.
    points = points[0]
    valid_depth = valid_depth[0]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Only use a narrow strip directly in front of the walker.
    mask = (
        valid_depth
        & (torch.abs(y) < 0.30)
        & (x > 0.95)
        & (x < 5.0)
    )

    x = x[mask]
    z = z[mask]

    bin_width = 0.10

    bin_edges = torch.arange(
        0.95,
        5.0 + bin_width,
        bin_width,
        device=x.device,
    )


def slope_ahead_camera_safe(env):
    """
    Camera slope observation for PPO.

    Uses the latest valid camera estimate during short perception
    dropouts. Never returns NaN.
    """

    slope = slope_ahead_camera(env)

    if slope.shape != (env.num_envs, 1):
        raise RuntimeError(
            "Camera slope must have shape "
            f"({env.num_envs}, 1), received {tuple(slope.shape)}"
        )

    # Create storage the first time this function runs
    if (
        not hasattr(env, "_camera_last_valid_slope")
        or env._camera_last_valid_slope.shape != slope.shape
    ):
        env._camera_last_valid_slope = torch.zeros(
            (env.num_envs, 1),
            device=slope.device,
            dtype=slope.dtype,
        )

        env._camera_invalid_age = torch.zeros(
            (env.num_envs, 1),
            device=slope.device,
            dtype=torch.long,
        )

    valid = torch.isfinite(slope)

    # -------------------------------------------------
    # Valid camera measurement
    # -------------------------------------------------

    env._camera_last_valid_slope = torch.where(
        valid,
        slope,
        env._camera_last_valid_slope,
    )

    # Reset age whenever perception is valid
    env._camera_invalid_age = torch.where(
        valid,
        torch.zeros_like(env._camera_invalid_age),
        env._camera_invalid_age + 1,
    )

    # -------------------------------------------------
    # Handle temporary perception failure
    # -------------------------------------------------

    MAX_HOLD_STEPS = 10   # 10 * 0.02 s = 0.2 s

    use_last_valid = (
        (~valid)
        & (env._camera_invalid_age <= MAX_HOLD_STEPS)
    )

    # Start with actual measurement
    safe_slope = slope.clone()

    # Short dropout -> use most recent valid estimate
    safe_slope = torch.where(
        use_last_valid,
        env._camera_last_valid_slope,
        safe_slope,
    )

    # Long dropout -> assume neutral/flat for now
    long_dropout = (
        (~valid)
        & (env._camera_invalid_age > MAX_HOLD_STEPS)
    )

    safe_slope = torch.where(
        long_dropout,
        torch.zeros_like(safe_slope),
        safe_slope,
    )

    # Absolute final safety check
    safe_slope = torch.nan_to_num(
        safe_slope,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return safe_slope
