import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


WALKER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/tbs-create/walker_shared_control/assets/walker/Collected_walker/walker_complete_before_camera.usd",


    ),
    

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
    ),

    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "fr_wheel",
                "fl_wheel",
                "rl_wheel",
                "rr_wheel",
            ],
            stiffness=0.0,
            damping=100.0,
            effort_limit_sim=400.0,
        ),

        "steering": ImplicitActuatorCfg(
            joint_names_expr=[
                "fr_steering_joint",
                "fl_steering_joint",
                "rl_steering_joint",
                "rr_steering_joint",
            ],
            stiffness=1000.0,
            damping=100.0,
        ),
    },



)