import numpy as np
import traceback

class TrajectoryPlanner:
    """
    Calculates desired robot states (joint and Cartesian positions, velocities,
    accelerations) on-the-fly based on the current time within a defined maneuver.
    Pre-computes waypoints and phase timings during initialization.
    """

    def __init__(self):
        """Initialize the trajectory planner with default values."""
        self.t_sampling = None
        self.t_horizon = None
        self.t_goto = None
        self.trajectory_type = 'target' # Fixed to point-to-point 'target'
        self.control_space = None # 'joint' or 'cartesian'
  
        self.target = None # Raw target from config (joint or cartesian)

        # Model info (from pinocchio)
        self.n_dof = None
        self.ee_dim = 3 # Fixed to 3D (XYZ)

        # References
        self.manager = None
        self.pinocchio = None

        # Initial State 
        self.q0 = None
        self.x0 = None

        # Pre-computed values (calculated in initial_state)
        self.q_target = None # Target joint configuration
        self.x_target = None # Target cartesian configuration (XYZ)
        self.t_total = None # Total maneuver duration (approach + horizon + return)
        self.t_end_approach = None
        self.t_end_horizon = None # End of the main maneuver phase
        self.t_end_return = None # Same as t_total

        # Internal state for on-the-fly calculation
        self.q_ik_guess = None # Stores the previously computed q_des for IK guess

        self._is_initialized = False # Config loaded
        self._is_ready = False # Initial state set and pre-computation done

        print("TrajectoryPlanner: READY")

    # --- Helper Functions ---

    def _sigma(self, tau):
        """Quintic polynomial for smooth trajectory interpolation."""
        tau = np.clip(tau, 0.0, 1.0) # Clamp tau to [0, 1]
        return 10 * tau**3 - 15 * tau**4 + 6 * tau**5

    def _sigma_dot(self, tau):
        """First derivative of quintic polynomial."""
        tau = np.clip(tau, 0.0, 1.0)
        return 30 * tau**2 - 60 * tau**3 + 30 * tau**4

    def _sigma_ddot(self, tau):
        """Second derivative of quintic polynomial."""
        tau = np.clip(tau, 0.0, 1.0)
        return 60 * tau - 180 * tau**2 + 120 * tau**3

    # --- Core Methods ---

    def initialize(self, manager, pinocchio):
        """
        Initializes the planner with configuration parameters and model info.
        Does NOT perform calculations dependent on initial state.

        Args:
            manager (ParameterManager): Instance containing configuration.
            pinocchio (PinocchioWrapper): Instance for model info and kinematics.

        Returns:
            bool: True if initialization successful, False otherwise.
        """

        try:
            # Store references
            self.manager = manager
            self.pinocchio = pinocchio

            # Extract parameters
            self.t_sampling = manager.t_sampling
            self.t_horizon = manager.t_horizon
            self.t_goto = manager.t_goto
            self.trajectory_type = 'target'
            self.control_space = manager.control_space
            self.target = np.array(manager.target)

            self.n_dof = pinocchio.n_dof
            self.ee_dim = 3 # Fixed to 3D Cartesian (XYZ)

            # Basic validation
            if self.t_sampling <= 0 or self.t_horizon < 0 or self.t_goto <= 0:
                raise ValueError("Invalid timing parameters (sampling, horizon, goto).")
            if self.control_space not in ['joint', 'cartesian']:
                raise ValueError(f"Invalid control space: {self.control_space}")
            if self.target is None:
                raise ValueError("Target configuration not found in parameters.")

            print(f"  Type: {self.trajectory_type}, Space: {self.control_space}")
            print(f"  Timings: Ts={self.t_sampling:.4f}, Th={self.t_horizon:.2f}, Tg={self.t_goto:.2f}")

            self._is_initialized = True

            print("Initialization COMPLETE (waiting for initial state)")
            
        except Exception as e:
            print(f"[ERROR] TrajectoryPlanner 'initialize()': {e}")
            traceback.print_exc()
            self.__init__()
        
        return self._is_initialized

    def plan_trajectory(self, q0, x0, id=None):
        """
        Sets the initial robot state and performs pre-computations like
        target calculation and phase timing.

        Args:
            q0 (np.ndarray): Initial joint configuration.
            x0 (np.ndarray): Initial Cartesian configuration (XYZ).

        Returns:
            t_total (float): Total duration of the maneuver.
        """

        if not self._is_initialized:
            print("[ERROR] Planner must be initialized before setting initial state.")
            return False
        
        if self._is_ready:
            print("[WARNING] Planner already pre-computed. Re-computing.")
            self.q_target = None
            self.x_target = None

        print("TrajectoryPlanner: Setting initial state...")
        try:
            self.q0 = np.array(q0)
            self.x0 = np.array(x0)
            self.q_target = np.zeros(self.n_dof) # Reset target joint configuration
            self.x_target = np.zeros(self.ee_dim) # Reset target cartesian configuration
            self.q_ik_guess = self.q0.copy() # Initialize previous q_des for IK

            print(f"  Initial state received: q0={np.round(self.q0, 3)}, x0={np.round(self.x0, 3)}")

            # Calculate Phase Timings
            self.t_end_approach = self.t_goto

            self.t_end_horizon = self.t_end_approach + self.t_horizon
            self.t_total = self.t_end_horizon + self.t_goto
            self.t_end_return = self.t_total

            print(f"  Phase timings: ApproachEnd={self.t_end_approach:.2f}, HorizonEnd={self.t_end_horizon:.2f}, Total={self.t_total:.2f}")

            # Calculate Target 
            print("  Calculating target configuration...")
            if self.control_space == 'joint':
                self.q_target = self.target.copy()   
                self.x_target = np.array(self.pinocchio.forward_kinematics(self.q_target))


                print(f"    Joint Target: q={np.round(self.q_target, 3)}")
                print(f"    Cart. Target (FK): x={np.round(self.x_target, 3)}")

            elif self.control_space == 'cartesian':
                self.x_target = self.target.copy()
                self.q_target = self.pinocchio.inverse_kinematics(self.x_target, self.q_ik_guess) # Use initial guess

                print(f"    Cart. Target: x={np.round(self.x_target, 3)}")
                print(f"    Joint Target (IK): q={np.round(self.q_target, 3)}")

            print("Pre-computation COMPLETE.")
            self._is_ready = True

        except Exception as e:
            print(f"[ERROR] TrajectoryPlanner 'plan_trajectory()': {e}")
            traceback.print_exc()
            self._is_ready = False

        return self.t_total

    def desired_state(self, time_instance, J_k):
        """
        Computes the desired state (q, q_dot, q_ddot, x, x_dot, x_ddot)
        for a specific time instance within the maneuver.

        Args:
            time_instance (float): The current time elapsed since the start
                                   of the maneuver (starts at 0).
            J_k (np.ndarray): Jacobian matrix at the current time instance.

        Returns:
            dict: A dictionary containing the 6 desired state vectors
                  {'q_des', 'q_dot_des', ..., 'x_ddot_des'}, or None if
                  a critical computation fails.
        """
        if not self._is_ready:
            print("[ERROR] Planner not ready.")
            return None

        try:
            # Determine Phase and Parameters
            q_start, q_end, x_start, x_end = None, None, None, None
            duration, phase_start_time = 0.0, 0.0

            if time_instance < self.t_end_approach: # Approach phase
                q_start, q_end = self.q0, self.q_target
                x_start, x_end = self.x0, self.x_target
                duration, phase_start_time = self.t_goto, 0.0

            elif time_instance < self.t_end_horizon: # Hold phase (target)
                q_start, q_end = self.q_target, self.q_target
                x_start, x_end = self.x_target, self.x_target
                duration, phase_start_time = self.t_horizon, self.t_end_approach

            elif time_instance < self.t_end_return: # Return phase
                q_start, q_end = self.q_target, self.q0
                x_start, x_end = self.x_target, self.x0
                duration, phase_start_time = self.t_goto, self.t_end_horizon

            else: # Beyond maneuver end time, hold at initial state
                q_start, q_end = self.q0, self.q0
                x_start, x_end = self.x0, self.x0
                duration, phase_start_time = 1.0, self.t_end_return # Avoid division by zero

            # Calculate normalized time tau
            tau = (time_instance - phase_start_time) / duration if duration > 1e-9 else 1.0

            # Calculate Desired State based on Control Space
            q_des_k = np.zeros(self.n_dof)
            q_dot_des_k = np.zeros(self.n_dof)
            q_ddot_des_k = np.zeros(self.n_dof)
            x_des_k = np.zeros(self.ee_dim)
            x_dot_des_k = np.zeros(self.ee_dim)
            x_ddot_des_k = np.zeros(self.ee_dim)
            Jdot_qdot_k = np.zeros((self.ee_dim, self.n_dof))
            J_pinv_k = np.zeros((self.n_dof, self.ee_dim))

            if self.control_space == 'joint':
                h_vec_q = q_end - q_start
                q_des_k = q_start + h_vec_q * self._sigma(tau)
                q_dot_des_k = (h_vec_q / duration) * self._sigma_dot(tau) if duration > 1e-9 else np.zeros(self.n_dof)
                q_ddot_des_k = (h_vec_q / duration**2) * self._sigma_ddot(tau) if duration > 1e-9 else np.zeros(self.n_dof)

                # Calculate Cartesian State via FK
                x_des_k = self.pinocchio.forward_kinematics(q_des_k.copy())
                x_dot_des_k = J_k @ q_dot_des_k.copy()

                Jdot_qdot_k = self.pinocchio.frame_jacobian_derivative(q_des_k.copy(), q_dot_des_k.copy())
                if Jdot_qdot_k is None:
                    print(f"[ERROR] Jacobian dot product failed at t={time_instance:.3f} for q_des={np.round(q_des_k,3)}")
                    return None

                x_ddot_des_k = Jdot_qdot_k + J_k @ q_ddot_des_k.copy()

            elif self.control_space == 'cartesian':
                h_vec_x = x_end - x_start
                x_des_k = x_start + h_vec_x * self._sigma(tau)
                x_dot_des_k = (h_vec_x / duration) * self._sigma_dot(tau) if duration > 1e-9 else np.zeros(self.ee_dim)
                x_ddot_des_k = (h_vec_x / duration**2) * self._sigma_ddot(tau) if duration > 1e-9 else np.zeros(self.ee_dim)

                # Calculate Joint State via IK
                q_des_k = self.pinocchio.inverse_kinematics(x_des_k.copy(), self.q_ik_guess.copy())
                if q_des_k is None:
                    print(f"[WARNING] IK failed at t={time_instance:.3f} for x_des={np.round(x_des_k,3)}. Reusing previous q_des.")
                    q_des_k = self.q_ik_guess # Reuse previous solution

                J_pinv_k = self.pinocchio.damped_pseudoinverse(J_k.copy())
                if J_pinv_k is None:
                    print(f"[ERROR] Jacobian pseudoinverse failed at t={time_instance:.3f} for J={np.round(J_k,3)}")
                    return None
                
                q_dot_des_k = J_pinv_k @ x_dot_des_k.copy()

                Jdot_qdot_k = self.pinocchio.frame_jacobian_derivative(q_des_k.copy(), q_dot_des_k.copy())
                if Jdot_qdot_k is None:
                    print(f"[ERROR] Jacobian dot product failed at t={time_instance:.3f} for q_des={np.round(q_des_k.copy(),3)}")
                    return None
                
                q_ddot_des_k = J_pinv_k @ (x_ddot_des_k.copy() - Jdot_qdot_k)

            # Update the previous desired joint state for the next IK guess
            self.q_ik_guess = q_des_k.copy()

            # Return results
            return {
                'q_des': q_des_k, 'q_dot_des': q_dot_des_k, 'q_ddot_des': q_ddot_des_k,
                'x_des': x_des_k, 'x_dot_des': x_dot_des_k, 'x_ddot_des': x_ddot_des_k,
                'q_target': self.q_target, 'x_target': self.x_target,
                'Jdot_qdot': Jdot_qdot_k, 'J_pinv': J_pinv_k
            }

        except Exception as e:
            print(f"[ERROR] TrajectoryPlanner 'desired_state()' at t={time_instance:.3f}: {e}")
            traceback.print_exc()
            return None