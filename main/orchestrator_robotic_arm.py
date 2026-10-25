import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Ensure 'src' is in python search path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from param_manager import ParameterManager
from wrapper_pinocchio import PinocchioWrapper
from trajectory_planner import TrajectoryPlanner
from arduino_manager import ArduinoManager
from controller_inverse_kinematics import InverseKinematicsControl

class Orchestrator:
    """
    Main coordinator:
    - Loads configuration
    - Plans trajectory
    - Dispatches controller commands
    - Maintains real-time loop synchronization with Arduino
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        print("=" * 60)
        print("          ROBOTIC ARM SYSTEM ORCHESTRATOR           ")
        print("=" * 60)

        # 1. PARAMETER VERIFICATION
        print("\n[Step 1/5] Initializing ParameterManager...")
        self.manager = ParameterManager()

        # 2. PINOCCHIO MODEL & KINEMATICS
        print("\n[Step 2/5] Initializing PinocchioWrapper...")
        self.pinocchio = PinocchioWrapper()
        if not self.pinocchio.initialize(self.manager):
            raise RuntimeError("Pinocchio initialization failed. Check URDF path and parameters.")

        # 3. TRAJECTORY PLANNER
        print("\n[Step 3/5] Initializing TrajectoryPlanner...")
        self.planner = TrajectoryPlanner()
        if not self.planner.initialize(self.manager, self.pinocchio):
            raise RuntimeError("Trajectory planner initialization failed.")

        # 4. CONTROLLER SELECTION
        print("\n[Step 4/5] Initializing Controller...")
        # Selects and executes the requested controller strategy.
        # For our RC servos (DM996), 'KIN' (Kinematic Position Tracking) is the native approach.
        self.controller_type = self.manager.controller_type.upper()
        self.pinocchio = self.pinocchio
        # initiliaze the controller based on the selected type
        if self.controller_type in ["KIN", "KINEMATIC"]:
            self.controller = InverseKinematicsControl()
            if not self.controller.initialize(self.manager, self.pinocchio):
                raise RuntimeError("Inverse Kinematics Controller initialization failed.")
        else:
            raise RuntimeError("[ERROR] Unknown controller type specified in configuration.")


        print(f"[Orchestrator] Selected Controller: {self.controller_type}")

        # 5. HARDWARE INTERFACE (ARDUINO)
        print("\n[Step 5/5] Configuring Hardware Interface...")
        self.arduino = None
        if not self.dry_run:
            self.arduino = ArduinoManager(port=self.manager.serial_port, baud=self.manager.baud_rate)
        else:
            print("[INFO] DRY-RUN MODE ACTIVE: Serial communication skipped.")

    # The Orchestrator is the dynamic state coordinator. It is the only point in the architecture where all the following come together:
    #    - The real data from the sensors (q_act, q_dot_act from the Arduino).
    #    - The desired states from the trajectory planner (state from the TrajectoryPlanner).
    #    - The kinematic/dynamic model (pinocchio).
    # So, it is the orchestrator's responsibility to assemble the control_input dictionary that will be passed to the controller's compute_law() method.
    def assemble_control_input(self, desired_state: dict, q_act: np.ndarray, q_dot_act: np.ndarray = None, tau_act: np.ndarray = None) -> dict:
            """
            Assembles a comprehensive, standardized control_input dictionary
            required by all controller types (IK, PDG, Impedance, etc.).
            
            Args:
                desired_state (dict): Output from TrajectoryPlanner.desired_state(...)
                q_act (np.ndarray): Current measured/estimated joint positions (rad)
                q_dot_act (np.ndarray, optional): Current joint velocities (rad/s). Defaults to zeros.
                tau_act (np.ndarray, optional): Current joint torques (Nm). Defaults to zeros.
                
            Returns:
                dict: Standardized control_input ready for compute_law().
            """
            # Fallbacks for optional feedback states
            if q_dot_act is None:
                q_dot_act = np.zeros(self.manager.num_joints)
            if tau_act is None:
                tau_act = np.zeros(self.manager.num_joints)

            # 1. Forward Kinematics on actual joint state
            x_act = self.pinocchio.forward_kinematics(q_act)
            if x_act is None:
                x_act = np.zeros(3)

            # 2. Actual Jacobians and Task Velocities
            J_act = self.pinocchio.frame_jacobian(q_act)
            if J_act is None:
                J_act = np.zeros((3, self.manager.num_joints))

            # Task-space actual velocity: x_dot_act = J * q_dot_act
            x_dot_act = J_act @ q_dot_act

            # 3. Extract Desired State values
            q_des = desired_state.get('q_des', q_act)
            q_dot_des = desired_state.get('q_dot_des', np.zeros_like(q_act))
            q_ddot_des = desired_state.get('q_ddot_des', np.zeros_like(q_act))

            x_des = desired_state.get('x_des', x_act)
            x_dot_des = desired_state.get('x_dot_des', np.zeros_like(x_act))
            x_ddot_des = desired_state.get('x_ddot_des', np.zeros_like(x_act))

            # Derivatives and pseudoinverses computed by trajectory planner or pinocchio
            Jdot_qdot = desired_state.get('Jdot_qdot')
            if Jdot_qdot is None:
                Jdot_qdot = self.pinocchio.frame_jacobian_derivative(q_act, q_dot_act)

            J_pinv = desired_state.get('J_pinv')
            if J_pinv is None:
                if hasattr(self.pinocchio, 'damped_pseudoinverse'):
                    J_pinv = self.pinocchio.damped_pseudoinverse(J_act)
                else:
                    J_pinv = np.linalg.pinv(J_act)

            # 4. Error Computations (Desired - Actual)
            err_q = q_des - q_act
            err_q_dot = q_dot_des - q_dot_act

            err_x = x_des - x_act
            err_x_dot = x_dot_des - x_dot_act

            # Standard dictionary packaging
            control_input = {
                'q_act': q_act,
                'q_dot_act': q_dot_act,
                'tau_act': tau_act,
                'q_des': q_des,
                'q_dot_des': q_dot_des,
                'q_ddot_des': q_ddot_des,
                'x_des': x_des,
                'x_dot_des': x_dot_des,
                'x_ddot_des': x_ddot_des,
                'x_act': x_act,
                'x_dot_act': x_dot_act,
                'err_q': err_q,
                'err_q_dot': err_q_dot,
                'err_x': err_x,
                'err_x_dot': err_x_dot,
                'J_act': J_act,
                'Jdot_qdot': Jdot_qdot,
                'J_pinv': J_pinv
            }

            return control_input


    def model_to_servo(self, q_model_rad: np.ndarray) -> list[int]:
            """
            Convert angles coming form Pinocchio (radiants,set URDF/DH)
            in the corresponding angels for the Arduino (deg 0-180).
            """
            q_deg = np.rad2deg(q_model_rad)
            
            # Real = URDF * direction + offset
            servo_angles = q_deg * self.manager.joint_directions + self.manager.joint_offsets_deg
            
            return servo_angles.astype(int).tolist()

    def servo_to_model(self, servo_angles_deg: list[int] | np.ndarray) -> np.ndarray:
        """
        Convert the actual position read from the motors (degrees 0-180)
        into the joint space for Pinocchio (radians).
        """
        servo_arr = np.array(servo_angles_deg, dtype=float)
        
        # Inverse formula: URDF = (Real - offset) / direction
        q_deg = (servo_arr - self.manager.joint_offsets_deg) / self.manager.joint_directions
        
        return np.deg2rad(q_deg)


    def run(self):
        """Executes the full trajectory lifecycle with real sensory feedback."""
        try:
            # Connect to Arduino if hardware is active
            if self.arduino:
                self.arduino.connect()
                print("Moving robot to Initial Home Position...")
                # Send the calibrated physical home pose to the servos
                home_servo_cmd = self.model_to_servo(self.manager.home_pos_rad)
                self.arduino.send_command(home_servo_cmd)
                time.sleep(2.0)  # Wait for physical settling

            # Establish initial state (q0 in rad, x0 in meters)
            if self.arduino:
                # Read initial actual pose directly from potentiometer telemetry
                raw_feedback = self.arduino.get_current_positions()
                q0 = self.servo_to_model(raw_feedback)
            else:
                q0 = self.manager.home_pos_rad.copy()

            x0 = self.pinocchio.forward_kinematics(q0)
            if x0 is None:
                raise RuntimeError("Failed to compute initial forward kinematics (x0).")

            # Pre-compute trajectory timings and waypoints
            print("\nPlanning trajectory...")
            t_total = self.planner.plan_trajectory(q0, x0)
            if t_total is None or t_total <= 0:
                raise RuntimeError("Trajectory planning failed.")

            dt = self.manager.t_sampling
            num_steps = int(np.ceil(t_total / dt))

            print("\n" + "=" * 60)
            print(f"Executing Trajectory: Total Duration = {t_total:.2f}s | Steps = {num_steps} | Rate = {1/dt:.1f}Hz")
            print("=" * 60)

            # State initialization for loop execution
            current_q = q0.copy()
            current_q_dot = np.zeros_like(current_q)
            prev_q = q0.copy()

            # --- REAL-TIME CONTROL LOOP (50 Hz) ---
            for step in range(num_steps):
                loop_start_time = time.perf_counter()
                time_instance = step * dt

                # 0. Read sensory feedback from Arduino and estimate actual joint velocities
                if self.arduino:
                    # Read telemetry vector from feedback potentiometers
                    telemetry = self.arduino.get_current_positions()

                    # Convert raw physical servo feedback to Pinocchio model space (rad)
                    current_q = self.servo_to_model(telemetry)

                    # Estimate actual joint velocity using backward finite differences
                    raw_q_dot = (current_q - prev_q) / dt

                    # First-order exponential smoothing filter to suppress ADC noise
                    alpha = 0.7  # Smoothing factor (0.0 = full filter, 1.0 = raw)
                    current_q_dot = alpha * raw_q_dot + (1.0 - alpha) * current_q_dot

                    prev_q = current_q.copy()

                # 1. Current Jacobian evaluated at actual configuration
                J_k = self.pinocchio.frame_jacobian(current_q)
                if J_k is None:
                    print(f"[ERROR] Jacobian evaluation failed at step {step}. Halting.")
                    break

                # 2. Trajectory Planner Desired State
                state = self.planner.desired_state(time_instance, J_k)
                if state is None:
                    print(f"[ERROR] Failed to obtain desired state at t={time_instance:.3f}s. Halting.")
                    break

                # 3. Assemble complete standardized control_input dictionary
                control_input = self.assemble_control_input(
                    desired_state=state,
                    q_act=current_q,
                    q_dot_act=current_q_dot
                )

                # 4. Computes the next joint command based on the desired state and current state.
                cmd_pos, cmd_vel, cmd_eff, ctrl_log = self.controller.compute_law(control_input, step)

                # 5. The DM996 RC Servos accept only position commands. Therefore, we need to convert the computed command into a position command. 
                # If the controller provides a position command, we use it directly. If it provides a velocity command, we integrate it to get the next position.
                if cmd_pos is not None:
                    q_target_rad = cmd_pos

                elif cmd_vel is not None:
                    # Integrate velocity starting from actual measured joint position: q_next = q_act + q_dot * dt
                    q_target_rad = current_q + cmd_vel * dt

                elif cmd_eff is not None:
                    # For torque-based controllers we need to implement a torque-to-position conversion, which is not trivial and requires dynamic modeling. For now, we will log a warning.
                    raise RuntimeError(f"[ERROR] Torque command received at step {step}, but torque-to-position conversion is not implemented.")
                else:
                    raise RuntimeError(f"[ERROR] No valid command from controller at step {step}.")

                # 6. Unit conversion: Radians -> Degrees (including hardware calibration offsets)
                servo_cmd = self.model_to_servo(q_target_rad)

                # 7. Send packet to Arduino
                if self.arduino:
                    self.arduino.send_command(servo_cmd)
                else:
                    # In dry-run mode, simulate state propagation via numerical integration
                    current_q_dot = (q_target_rad - current_q) / dt
                    current_q = q_target_rad.copy()

                # Progress logging every 10 steps (~0.2s)
                if step % 10 == 0 or step == num_steps - 1:
                    print(
                        f"t={time_instance:5.2f}s | "
                        f"Target XYZ: {np.round(state['x_des'], 3)} | "
                        f"Error XYZ: {np.round(control_input['err_x'], 4)} | "
                        f"Cmd Servo Deg: {servo_cmd}"
                    )

                # Maintain 50 Hz deterministic timing
                elapsed = time.perf_counter() - loop_start_time
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print("\n[SUCCESS] Trajectory execution finished!")

        except KeyboardInterrupt:
            print("\n[WARNING] Execution interrupted by user (Ctrl+C).")
        finally:
            if self.arduino:
                print("Safely returning robot to Home position...")
                try:
                    home_servo_cmd = self.model_to_servo(self.manager.home_pos_rad)
                    self.arduino.send_command(home_servo_cmd)
                    time.sleep(1.0)
                except Exception:
                    pass
                self.arduino.disconnect()
            print("[Orchestrator] Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(description="5-DOF Robotic Arm Orchestrator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate trajectory execution without connecting to physical Arduino hardware."
    )
    args = parser.parse_args()

    orchestrator = Orchestrator(dry_run=args.dry_run)
    orchestrator.run()


if __name__ == "__main__":
    main()