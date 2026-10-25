import numpy as np
import traceback

POSITION_ONLY = True  # Restrict task space to translation (XYZ) only. If True, ignores rotational error components

class InverseKinematicsControl:
    """
    Implements task-space velocity control using Jacobian pseudoinverse.
    Computes desired joint velocity based on Cartesian errors and feedforward.
    Supports 'position_only' to restrict task space strictly to translation (XYZ).
    """
    def __init__(self):
        """Initializes the InverseKinematicsController."""
        self.control_command = None # 'position' or 'velocity'
        self.control_space = None   # Expected 'cartesian'
        self.kp = None              # Proportional gain matrix (Cartesian space)
        self.n_dof = None
        self.ee_dim = None
        self.position_only = POSITION_ONLY  
        self.pinocchio = None       # Pinocchio wrapper reference

        self._is_initialized = False
        print("InverseKinematicsController: READY")

    def initialize(self, manager, pinocchio):
        """Initializes the controller with parameters."""
        try:
            print("InverseKinematicsController: Initializing...")
            self.pinocchio = pinocchio
            self.n_dof = pinocchio.n_dof
            
            # Read settings from manager
            self.control_command = getattr(manager, 'control_command', 'velocity')
            self.control_space = manager.control_space

            if self.control_space != 'cartesian':
                raise ValueError("IK Control operates only in 'cartesian' control space.")

            # Set task-space dimension based on position_only flag
            if self.position_only:
                self.ee_dim = 3  # Strictly X, Y, Z translation
                print("  [INFO] 'position_only' active: controlling translation (XYZ) only.")
            else:
                self.ee_dim = getattr(pinocchio, 'ee_dim', 3)

            # Gains applied in Cartesian space
            kp_scalar = manager.gains.get('kp', 1.0) if hasattr(manager, 'gains') else 1.0
            if isinstance(kp_scalar, (int, float)):
                self.kp = float(kp_scalar) * np.eye(self.ee_dim)
            else:
                self.kp = np.diag(kp_scalar[:self.ee_dim])

            self._is_initialized = True
            print(f"InverseKinematicsController Initialized: Command='{self.control_command}', Space='{self.control_space}', TaskDim={self.ee_dim}")
            print(f"  Cartesian Kp Gain:\n{self.kp}")
            return True

        except Exception as e:
            print(f"[ERROR] InverseKinematicsController 'initialize()': {e}")
            traceback.print_exc()
            self.__init__() # Reset state
            return False

    def compute_law(self, control_input, k):
        """
        Computes the desired joint velocity via Inverse Kinematics.

        Args:
            control_input (dict): Dictionary containing current state and targets.
            k (int): Current time step index.

        Returns:
            tuple: (cmd_pos, cmd_vel, cmd_eff, ctrl_log_data)
        """
        try:
            if not self._is_initialized:
                raise RuntimeError("IK Controller compute_law: Not initialized.")

            # Extract required data
            x_dot_des = np.asarray(control_input.get('x_dot_des'), dtype=float)
            err_x = np.asarray(control_input.get('err_x'), dtype=float)
            J_act = np.asarray(control_input.get('J_act'), dtype=float)
            J_pinv = control_input.get('J_pinv')

            if x_dot_des is None or err_x is None or J_act is None:
                raise ValueError("Missing required keys (x_dot_des, err_x, J_act) in control_input.")

            # If position_only is active, slice vectors and Jacobian to translation only (first 3 rows)
            if self.position_only:
                err_x_active = err_x[:3]
                x_dot_des_active = x_dot_des[:3]
                J_active = J_act[:3, :]
                
                # Recompute pseudoinverse on the 3xN Jacobian if pinocchio wrapper provides it,
                # otherwise compute Moore-Penrose pinv for the 3xN matrix
                if hasattr(self.pinocchio, 'damped_pseudoinverse'):
                    J_pinv_active = self.pinocchio.damped_pseudoinverse(J_active)
                else:
                    J_pinv_active = np.linalg.pinv(J_active)
            else:
                err_x_active = err_x
                x_dot_des_active = x_dot_des
                J_pinv_active = J_pinv if J_pinv is not None else np.linalg.pinv(J_act)

            # IK velocity control law: q_dot = J# * (x_dot_des + Kp * error_x)
            cartesian_velocity_target = x_dot_des_active + self.kp @ err_x_active
            cmd_vel = J_pinv_active @ cartesian_velocity_target

            cmd_pos = None
            cmd_eff = None
            ctrl_log_data = {}

            return cmd_pos, cmd_vel, cmd_eff, ctrl_log_data

        except Exception as e:
            print(f"[ERROR] InverseKinematicsController 'compute_law()' at step {k}: {e}")
            traceback.print_exc()
            return None, None, None, {}