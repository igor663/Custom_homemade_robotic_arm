import sys
import numpy as np
from pathlib import Path

# Python 3.11+ built-in tomllib with fallback to tomli for older versions
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit("[ERROR] Neither 'tomllib' nor 'tomli' found. Install with: pip install tomli")


class ParameterManager:
    """
    Configuration manager for the robotic arm.
    Reads config.toml, performs required unit conversions (degrees -> radians),
    resolves absolute file paths, and exposes variables with the exact attribute
    names expected by PinocchioWrapper and TrajectoryPlanner.
    """

    def __init__(self, config_path: str | Path | None = None):
        # Resolve project directories (assumes this file is in <project_root>/src/)
        self.src_dir = Path(__file__).resolve().parent
        self.project_root = self.src_dir.parent

        if config_path is None:
            self.config_path = self.project_root / "config" / "config.toml"
        else:
            self.config_path = Path(config_path).resolve()

        self._raw_config = {}
        self.load()

    def load(self):
        """Loads and parses the TOML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Config file not found at: {self.config_path}")

        print(f"[ParamManager] Loading configuration from: {self.config_path}")
        with open(self.config_path, "rb") as f:
            self._raw_config = tomllib.load(f)

        self._parse_parameters()
        print("[ParamManager] Configuration parsed and validated successfully.")

    def _parse_parameters(self):
        """Extracts and formats parameters matching PinocchioWrapper and TrajectoryPlanner requirements."""
        
        # ---------------------------------------------------------
        # 1. SERIAL COMMUNICATION [serial]
        # ---------------------------------------------------------
        serial_cfg = self._raw_config.get("serial", {})
        self.serial_port = serial_cfg.get("port", "/dev/ttyACM0")
        self.baud_rate = int(serial_cfg.get("baud_rate", 115200))

        # ---------------------------------------------------------
        # 2. ROBOT BASE PARAMETERS [robot]
        # ---------------------------------------------------------
        robot_cfg = self._raw_config.get("robot", {})
        self.home_pos_deg = np.array(robot_cfg.get("home_pos", [90, 90, 90, 90, 90]), dtype=float)
        self.home_pos_rad = np.deg2rad(self.home_pos_deg)
        self.num_joints = len(self.home_pos_deg)

        # Joint limits: convert [[min, max], ...] in deg to radians for Pinocchio
        limits_deg = np.array(robot_cfg.get("joints_limits", [[0, 180]] * self.num_joints), dtype=float)
        self.lower_limit_deg = limits_deg[:, 0]
        self.upper_limit_deg = limits_deg[:, 1]
        
        # Required by PinocchioWrapper (manager.lower_limit, manager.upper_limit in radians)
        self.lower_limit = np.deg2rad(self.lower_limit_deg)
        self.upper_limit = np.deg2rad(self.upper_limit_deg)

        # ---------------------------------------------------------
        # 3. ROBOT MODEL & URDF [robot_model]
        # ---------------------------------------------------------
        model_cfg = self._raw_config.get("robot_model", {})
        rel_urdf_path = model_cfg.get("urdf_path", "models/robot.urdf")
        
        # Required by PinocchioWrapper: manager.urdf_absolute_path and manager.urdf_ee
        self.urdf_absolute_path = (self.project_root / rel_urdf_path).resolve()
        self.urdf_ee = model_cfg.get("ee_frame_name", "end_effector_link")

        # ---------------------------------------------------------
        # 4. PINOCCHIO & IK PARAMETERS [pinocchio]
        # ---------------------------------------------------------
        pin_cfg = self._raw_config.get("pinocchio", {})
        self.iterations = int(pin_cfg.get("iterations", 100))
        self.tolerance = float(pin_cfg.get("tolerance", 1e-4))
        self.norm = float(pin_cfg.get("norm_step_limit", 0.1)) # maps to manager.norm
        self.damping = float(pin_cfg.get("damping", 0.05))
        self.adaptive_damping = bool(pin_cfg.get("adaptive_damping", True))
        self.lambda_min = float(pin_cfg.get("lambda_min", 0.01))
        self.lambda_max = float(pin_cfg.get("lambda_max", 0.5))
        self.det_threshold = float(pin_cfg.get("det_threshold", 1e-3))

        # ---------------------------------------------------------
        # 5. CONTROLLER SETTINGS [controller]
        # ---------------------------------------------------------
        ctrl_cfg = self._raw_config.get("controller", {})
        self.controller_type = ctrl_cfg.get("controller_type", "KIN")
        self.control_space = ctrl_cfg.get("control_space", "cartesian") # 'joint' or 'cartesian'

        # ---------------------------------------------------------
        # 6. TRAJECTORY PLANNER [trajectory] & [target]
        # ---------------------------------------------------------
        traj_cfg = self._raw_config.get("trajectory", {})
        self.trajectory_type = traj_cfg.get("trajectory_type", "target")
        self.trajectory = self.trajectory_type  # alias for backwards compatibility
        self.t_sampling = float(traj_cfg.get("t_sampling", 0.02))
        self.t_goto = float(traj_cfg.get("t_goto", 3.0))
        self.t_horizon = float(traj_cfg.get("t_horizon", 2.0))

        # Extract target configurations
        target_cfg = self._raw_config.get("target", {})
        self.target_joint_deg = np.array(target_cfg.get("target_joint", self.home_pos_deg), dtype=float)
        self.target_joint_rad = np.deg2rad(self.target_joint_deg)
        self.target_cartesian_m = np.array(target_cfg.get("target_cartesian", [0.15, 0.0, 0.20]), dtype=float)

        # TrajectoryPlanner expects 'manager.target' formatted for the chosen control space
        if self.control_space == "joint":
            self.target = self.target_joint_rad.copy()  # Radians for joint control
        elif self.control_space == "cartesian":
            self.target = self.target_cartesian_m.copy() # Meters for 3D XYZ
        else:
            raise ValueError(f"[ERROR] Unsupported control space: {self.control_space}")


if __name__ == "__main__":
    # Self-test routine, runs when executing this file directly. Prints all loaded parameters for verification.
    manager = ParameterManager()
    print("\n--- PARAMETER MANAGER VERIFICATION ---")
    print(f"Project root:         {manager.project_root}")
    print(f"URDF path:            {manager.urdf_absolute_path}")
    print(f"End-Effector frame:   {manager.urdf_ee}")
    print(f"Home Position (rad):  {np.round(manager.home_pos_rad, 3)}")
    print(f"Lower Limits (rad):   {np.round(manager.lower_limit, 3)}")
    print(f"Upper Limits (rad):   {np.round(manager.upper_limit, 3)}")
    print(f"Control Space:        {manager.control_space}")
    print(f"Target selected:      {manager.target}")