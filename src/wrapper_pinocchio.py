import traceback
import numpy as np
import os
import pinocchio as pin


class PinocchioWrapper:
    """
    Wrapper for Pinocchio library functionalities.

    Handles model loading, kinematics (FK, IK), dynamics (gravity, RNEA),
    Jacobians for a fixed-base robot. Controlled in 3D Cartesian space (XYZ).
    Receives configuration directly and uses basic error handling.
    """

    def __init__(self):
        """Initialize the Pinocchio wrapper with default values."""
        self.model_path = None # Path to the URDF model
        self.model_name = None # Name of the model
        self.model = None # Pinocchio model object
        self.data = None # Pinocchio data object
        self.frame_id = None # Frame ID for end-effector
        self.n_dof = None # Number of degrees of freedom
        self.damping = None # Static damping value
        self.adaptive_damping = False # Flag from config
        self.lambda_min = None # Min adaptive damping
        self.lambda_max = None # Max adaptive damping
        self.det_threshold = None # Determinant threshold for adaptive damping
        self.lower_limit = None
        self.upper_limit = None # Joint limits
        self.tolerance = None
        self.iterations = None # Max iterations for IK
        self.norm = None

        self.manager = None # Reference to ParameterManager

        self._is_initialized = False # Initialization status

        print("PinocchioWrapper: READY")

    def initialize(self, manager):
        """
        Builds the Pinocchio model from URDF specified in the configuration.

        Args:
            manager: ParameterManager instance containing configuration.
        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            # Store module references
            self.manager = manager

            # Extract parameters from the manager
            self.lower_limit = np.array(manager.lower_limit)
            self.upper_limit = np.array(manager.upper_limit)
            self.damping = manager.damping
            self.adaptive_damping = manager.adaptive_damping
            self.lambda_min = manager.lambda_min
            self.lambda_max = manager.lambda_max
            self.det_threshold = manager.det_threshold
            self.tolerance = manager.tolerance
            self.iterations = manager.iterations
            self.norm = manager.norm
            self.urdf_ee = manager.urdf_ee

            # Construct path relative to config parameters using ParameterManager resolution
            self.model_path = str(manager.urdf_absolute_path)
            print(f"Loading URDF from: {self.model_path}")

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"URDF file not found at: {self.model_path}")

            print(f"PinocchioWrapper: Building URDF file from path...")
            self.model = pin.buildModelFromUrdf(self.model_path)

            if self.model is None:
                raise RuntimeError(f"Failed to build Pinocchio model from URDF.")

            print("PinocchioWrapper: Unpacking URDF model:")

            self.data = self.model.createData()
            self.frame_id = self.model.getFrameId(self.urdf_ee)
            if not self.model.existFrame(self.urdf_ee):
                raise ValueError(f"End-effector frame '{self.urdf_ee}' not found in model.")

            self.n_dof = self.model.nv # Number of joints
            self.model_name = self.model.name

            print(f"    Name: {self.model_name}")
            print(f"    End-effector: {self.urdf_ee} [ID: {self.frame_id}]")
            print(f"    Number of DoFs: {self.n_dof}")
            print(f"    Joint Limits (Lower): {self.lower_limit}")
            print(f"    Joint Limits (Upper): {self.upper_limit}")

            self._is_initialized = True
            print("PinocchioWrapper: Initialization COMPLETE")
            return self._is_initialized

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'initialize()': {e}")
            traceback.print_exc()
            self.__init__()
            return False

    def _update_kinematics(self, q):
        """
        Updates kinematics based on joint positions q.
        Computes placements and joint Jacobians.
        """
        try:
            q = np.asarray(q)

            pin.forwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            return True
        
        except Exception as e:
            print(f"[ERROR] PinocchioWrapper '_update_kinematics(q)': {e}")
            return False

    def _update_kinematics_velocity(self, q, q_dot):
        """
        Updates kinematics based on joint positions q and velocities q_dot.
        Computes placements, Jacobians, and frame accelerations (for Jdot*qdot).
        """
        try:
            q = np.asarray(q)
            q_dot = np.asarray(q_dot)

            # Compute forward kinematics including velocities and accelerations (with qddot=0)
            # This populates data.a which contains the Jdot*qdot term
            pin.forwardKinematics(self.model, self.data, q, q_dot, np.zeros_like(q_dot))
            # Compute standard Jacobians
            pin.computeJointJacobians(self.model, self.data, q)
            # Compute Jacobian time variations (spatial) - might not be needed if using classical accel below
            # pin.computeJointJacobiansTimeVariation(self.model, self.data, q, q_dot)
            # Update frame placements based on the computed kinematics
            pin.updateFramePlacements(self.model, self.data)

            return True

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper '_update_kinematics_velocity(q, q_dot)': {e}")
            return False

    # --- Core Methods ---

    def forward_kinematics(self, q):
        """
        Computes and returns the forward kinematics (XYZ position) for the given joint state.

        Args:
            q: Current joint positions.

        Returns:
            x: XYZ position (np.ndarray shape (3,)) or None if computation fails.
        """
        try:

            if not self._update_kinematics(q):
                raise Exception("Failed to update kinematics.")

            # Extract XYZ position from translation (indices 0, 1 and 2)
            position_full = self.data.oMf[self.frame_id].translation
            return position_full.copy() # Return XYZ

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'forward_kinematics()': {e}")
            return None

    def frame_jacobian(self, q):
        """
        Computes and returns the XYZ end-effector Jacobian for the given joint state.

        Args:
            q: Current joint positions.

        Returns:
            The Jacobian matrix (np.ndarray shape (3, n_dof)) or None if computation fails.
        """
        try:
            #if not self._is_initialized:
            #    raise RuntimeError("Not initialized.")
            
            if not self._update_kinematics(q):
                raise Exception("Failed to update kinematics.")

            # Get the Jacobian in world-aligned frame (geometric Jacobian J = [J_v; J_w] (6xN))
            ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            J_full = pin.getFrameJacobian(self.model, self.data, self.frame_id, ref_frame)

            # Extract rows for linear velocity (0, 1, 2 for X, Y, Z velocity)
            J_linear = J_full[0:3, :]

            if np.isnan(J_linear).any():
                print(f"[WARNING] PinocchioWrapper 'frame_jacobian()': Computed Jacobian contains NaN for q = {q}.")

            return J_linear.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'frame_jacobian()': {e}")
            return None

    def get_jacobian_derivative_over_time(self, q, q_dot):
        """
        The function uses the method computeJointJacobiansTimeVariation from the Pinocchio library 
        to compute the time variation of the Jacobian matrix for a robotic model.

        Args:
            q: Current joint positions (np.ndarray shape (n_dof,)).
            q_dot: Current joint velocities (np.ndarray shape (n_dof,)).

        Returns:
            Computes the full model Jacobian variations with respect to time. It corresponds to dJ/dt which depends both on q and v.
        """
        try:
            J_dot = pin.computeJointJacobiansTimeVariation(self.model, self.data, q, q_dot)
            J_dot_cut = J_dot[0:3] # X, Y, Z components
            return J_dot_cut.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'get_jacobian_derivative_over_time': {e}")
            return None

    def frame_jacobian_derivative(self, q, q_dot):
        """
        Computes the classical acceleration term (Jdot * q_dot) for the end-effector's
        linear part (XYZ). This term represents the velocity-product
        acceleration contribution (Coriolis, centrifugal).

        Uses the relationship: classical_accel = J*qddot + Jdot*qdot.
        By calling forwardKinematics with qddot=0, the resulting classical
        acceleration is precisely the Jdot*qdot term we need.

        Args:
            q: Current joint positions (np.ndarray shape (n_dof,)).
            q_dot: Current joint velocities (np.ndarray shape (n_dof,)).

        Returns:
            The relevant component (XYZ) of the classical (Jdot * q_dot) vector
            (np.ndarray shape (3,)) or None if computation fails.
        """
        try:
            # This updates kinematics and computes accelerations with qddot=0
            if not self._update_kinematics_velocity(q, q_dot):
                raise Exception("Failed to update kinematics including velocity/acceleration.")

            # Get the classical acceleration of the frame. Since FK was called with
            # qddot=0, this spatial acceleration vector `data.a` corresponds to Jdot*qdot.
            # The classical acceleration includes the w x v term automatically.
            ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            classical_accel_full = pin.getFrameClassicalAcceleration(self.model, self.data, self.frame_id, ref_frame)

            # classical_accel_full is a SpatialAcceleration vector (linear, angular)
            # Extract the linear part (first 3 elements for X, Y, Z)
            jdot_qdot_linear = classical_accel_full.linear.copy()

            if np.isnan(jdot_qdot_linear).any():
                print(f"[WARNING] PinocchioWrapper 'frame_jacobian_derivative()': Computed classical Jdot*qdot contains NaN for q={q}, q_dot={q_dot}.")

            return jdot_qdot_linear.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'frame_jacobian_derivative()': {e}")
            return None

    def damped_pseudoinverse(self, J):
        """
        Computes damped pseudoinverse J_pinv = J.T * (J * J.T + lambda * I)^-1.
        Uses adaptive damping if enabled in config, otherwise uses static damping.

        Args:
            J: The Jacobian matrix (expected shape (3, n_dof)).

        Returns:
            The damped pseudoinverse (np.ndarray) or None if inversion fails or J is invalid.
        """
        try:

            rows, cols = J.shape

            # Calculate Damping Factor (lambda)
            current_damping = 0.0
            if self.adaptive_damping:
                # Calculate J @ J.T
                JJT = J @ J.T
                # Measure: Determinant of J @ J.T (cheap for small matrices)
                measure = np.linalg.det(JJT)

                # Ensure measure is non-negative for threshold comparison
                measure = max(0.0, measure) # Clamp at zero if slightly negative due to precision

                if measure < self.det_threshold:
                    # Linearly interpolate damping between lambda_min and lambda_max
                    scale = 1.0 - (measure / self.det_threshold) if self.det_threshold > 1e-9 else 1.0
                    current_damping = self.lambda_min + (self.lambda_max - self.lambda_min) * scale
                    # Optional: Print when adapting
                    # print(f"Adaptive damping active: det={measure:.4e}, lambda={current_damping:.4f}")
                else:
                    current_damping = self.lambda_min # Use minimum damping when well-conditioned

                # Ensure damping doesn't exceed max (shouldn't with linear interp, but safety check)
                current_damping = min(current_damping, self.lambda_max)

            else:
                # Use static damping value from config
                current_damping = self.damping

            # Compute J*J^T (if not already computed for adaptive)
            if not self.adaptive_damping:
                JJT = J @ J.T

            # Add damping: JJT + lambda * I
            identity_matrix = np.eye(rows)
            damped_JJT = JJT + current_damping * identity_matrix

            # Compute inverse: (JJT + lambda * I)^-1
            # Using pinv is generally more robust than inv for potentially ill-conditioned matrices
            inv_damped_JJT = np.linalg.pinv(damped_JJT)

            # Compute pseudoinverse: J.T @ (JJT + lambda * I)^-1
            J_pinv = J.T @ inv_damped_JJT

            return J_pinv

        except np.linalg.LinAlgError as lae:
            print(f"[ERROR] PinocchioWrapper 'damped_pseudoinverse()': Linear algebra error during inversion - {lae}. J = {J}, Damping = {current_damping}")
            return None
        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'damped_pseudoinverse()': {e}")
            return None

    def inverse_kinematics(self, x_target, q_guess):
        """
        Computes Inverse Kinematics (IK) for the EE target position (XYZ)
        using iterative damped least-squares. Enforces joint limits.

        Args:
            x_target: Target coordinate (np.ndarray shape (3,)).
            q_guess: Initial joint configuration guess (np.ndarray shape (n_dof,)).

        Returns:
            Updated joint configuration (np.ndarray shape (n_dof,)) or None if computation fails or doesn't converge.
        """
        try:

            q_current = np.asarray(q_guess).copy()
            x_target = np.asarray(x_target)

            if q_current.shape != (self.n_dof,):
                raise ValueError(f"Invalid shape for q_guess: expected ({self.n_dof},), got {q_current.shape}")

            lower = np.asarray(self.lower_limit)
            upper = np.asarray(self.upper_limit)
            if lower.shape != (self.n_dof,) or upper.shape != (self.n_dof,):
                raise ValueError("Joint limit dimensions mismatch number of DoFs.")

            success = False
            for i in range(self.iterations):
                # 1. Calculate current position and error
                current_pos = self.forward_kinematics(q_current)
                if current_pos is None:
                    print(f"[WARNING] IK iter {i}: FK failed for q={q_current}. Aborting.")
                    return None

                error_pos = x_target - current_pos
                error_norm = np.linalg.norm(error_pos)

                # 2. Check for convergence
                if error_norm < self.tolerance:
                    success = True
                    # print(f"IK Converged in {i+1} iterations. Error: {error_norm:.4f}")
                    break

                # 3. Get Jacobian
                J_linear = self.frame_jacobian(q_current)
                if J_linear is None:
                    print(f"[WARNING] IK iter {i}: Jacobian computation failed for q={q_current}. Aborting.")
                    return None
                if np.isnan(J_linear).any():
                    print(f"[WARNING] IK iter {i}: Jacobian contains NaN for q={q_current}. Aborting.")
                    return None

                # 4. Compute step using damped pseudoinverse
                J_pinv = self.damped_pseudoinverse(J_linear)
                if J_pinv is None:
                    print(f"[WARNING] IK iter {i}: Pseudoinverse computation failed. Aborting.")
                    return None

                delta_q = J_pinv @ error_pos

                # 5. Limit step size
                delta_q_norm = np.linalg.norm(delta_q)
                if delta_q_norm > self.norm:
                    # print(f"Iter {i}: Scaling delta_q norm from {delta_q_norm:.4f} to {self.norm:.4f}")
                    delta_q *= (self.norm / delta_q_norm)

                # 6. Update configuration using integrate
                q_next = pin.integrate(self.model, q_current, delta_q)
                if q_next is None:
                    print(f"[WARNING] IK iter {i}: integrate_velocity failed. Aborting.")
                    return None
                
                # 8. Enforce joint limits strictly
                q_clipped = np.clip(q_next, lower, upper)
                q_current = q_clipped # Use clipped value for next iteration

            # Post-Loop Check
            final_pos = self.forward_kinematics(q_current)
            final_error_norm = np.linalg.norm(x_target - final_pos) if final_pos is not None else float('inf')

            if final_error_norm < self.tolerance:
                success = True
            else:
                # Still return the best effort result and log a WARNING
                if not success:
                    print(f"[WARNING] IK Failed to converge within tolerance ({self.tolerance:.1e}) after {self.iterations} iterations. "
                          f"Final q: {np.round(q_current,3)}, Final x: {np.round(final_pos,3) if final_pos is not None else 'N/A'}, "
                          f"Target x: {np.round(x_target,3)}, Final Error: {final_error_norm:.4f}")

            return q_current.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'inverse_kinematics()': {e}")
            return None

    def gravity_compensation(self, q):
        """
        Computes and returns the gravity compensation torques for the given joint state.

        Args:
            q: Current joint positions.

        Returns:
            The gravity torque vector (np.ndarray shape (n_dof,)) or None if computation fails.
        """
        try:
 
            q = np.asarray(q)

            g_comp = pin.computeGeneralizedGravity(self.model, self.data, q)

            return g_comp.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'gravity_compensation()': {e}")
            return None

    def inverse_dynamics(self, q, q_dot, q_ddot):
        """
        Computes inverse dynamics using the Recursive Newton-Euler Algorithm (RNEA).
        Calculates the joint torques required to achieve the given state and acceleration.
        tau = M(q)q_ddot + C(q, q_dot)q_dot + g(q)

        Args:
            q: Joint positions (np.ndarray shape (n_dof,)).
            q_dot: Joint velocities (np.ndarray shape (n_dof,)).
            q_ddot: Joint accelerations (np.ndarray shape (n_dof,)).

        Returns:
            The required joint torques (np.ndarray shape (n_dof,)) or None if computation fails.
        """
        try:

            q = np.asarray(q)
            q_dot = np.asarray(q_dot)
            q_ddot = np.asarray(q_ddot)

            # RNEA computes necessary terms internally based on q, q_dot, q_ddot
            tau = pin.rnea(self.model, self.data, q, q_dot, q_ddot)


            return tau.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'inverse_dynamics()': {e}")
            return None

    def random_configuration(self):
        """
        Generates a random joint configuration within limits.

        Returns:
            A random joint configuration vector (shape (n_dof,)) or None if model not built
            or limits are invalid.
        """
        try:
            if not self._is_initialized:
                raise RuntimeError("Not initialized.")

            if self.lower_limit is None or self.upper_limit is None:
                raise RuntimeError("Joint limits not initialized.")
            if self.lower_limit.shape != (self.n_dof,) or self.upper_limit.shape != (self.n_dof,):
                raise ValueError("Joint limit dimensions mismatch number of DoFs.")

            return pin.randomConfiguration(self.model, self.lower_limit, self.upper_limit)

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'random_configuration()': {e}")
            return None

    def get_joint_space_inertia_matrix(self, q):
        """
        Computes the FULL, symmetric joint space inertia matrix M(q).

        This method first calls pin.crba to compute the upper-triangular part
        of the matrix and then explicitly makes it symmetric.

        Args:
            q (np.ndarray): The current joint configuration vector.

        Returns:
            np.ndarray: The full (n_dof x n_dof) symmetric inertia matrix, or None on failure.
        """
        try:
            if not self._is_initialized:
                raise RuntimeError("PinocchioWrapper not initialized.")

            q = np.asarray(q)

            # 1. Call CRBA. This computes the upper-triangular part and stores it in self.data.M.
            pin.crba(self.model, self.data, q)

            # 2. Make the matrix symmetric.
            #    The result of crba is a NumPy array with the lower triangle being zero.
            #    To fill it, Get the upper triangle (including the diagonal)
            #    Transpose it, then fill the lower triangle of the original matrix.
            #    This is efficient as it avoids creating new matrices.

            num_dof = self.n_dof
            Mq = self.data.M
            for i in range(num_dof):
                for j in range(i + 1, num_dof):
                    Mq[j, i] = Mq[i, j]

            return Mq.copy()

        except Exception as e:
            print(f"[ERROR] PinocchioWrapper 'get_joint_space_inertia_matrix()': {e}")
            traceback.print_exc()
            return None

    def get_Coriolis_matrix(self, q, q_dot):
        c_matrix = pin.computeCoriolisMatrix(self.model, self.data, q, q_dot)
        return c_matrix

    def get_gravity_matrix(self, q):
        g_mat = pin.computeGeneralizedGravity(self.model, self.data, q)
        return g_mat