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

    def run(self):
        """Executes the full trajectory lifecycle."""
        try:
            if self.arduino:
                self.arduino.connect()
                print("Moving robot to Initial Home Position...")
                self.arduino.send_command(self.manager.home_pos_deg.astype(int).tolist())
                time.sleep(2.0)

            # Establish initial state (q0 in rad, x0 in meters)
            q0 = self.manager.home_pos_rad.copy()
            x0 = self.pinocchio.forward_kinematics(q0)
            if x0 is None:
                raise RuntimeError("Failed to compute initial forward kinematics (x0).")

            print("\nPlanning trajectory...")
            t_total = self.planner.plan_trajectory(q0, x0)
            if t_total is None or t_total <= 0:
                raise RuntimeError("Trajectory planning failed.")

            dt = self.manager.t_sampling
            num_steps = int(np.ceil(t_total / dt))

            print("\n" + "=" * 60)
            print(f"Executing Trajectory: Total Duration = {t_total:.2f}s | Steps = {num_steps} | Rate = {1/dt:.1f}Hz")
            print("=" * 60)

            current_q = q0.copy()
            current_q_dot = np.zeros_like(current_q)

            # --- REAL-TIME CONTROL LOOP (50 Hz) ---
            for step in range(num_steps):
                loop_start_time = time.perf_counter()
                time_instance = step * dt

                # 1. Current Jacobian
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
                    current_q_dot = cmd_vel.copy()
                    # Integrate velocity: q_next = q + q_dot * dt
                    q_target_rad = current_q + cmd_vel * dt

                elif cmd_eff is not None:
                    # For torque-based controllers we need to implement a torque-to-position conversion, which is not trivial and requires dynamic modeling. For now, we will log a warning.
                     raise RuntimeError("[ERROR] Torque command received at step {step}, but torque-to-position conversion is not implemented.")
                else:
                     raise RuntimeError("[ERROR] No valid command from controller at step {step}.")


                # 6. Unit conversion: Radians -> Degrees
                q_cmd_deg = np.rad2deg(q_target_rad)

                # 7. Safety constraints: Clamp to physical robot joint limits
                q_safe_deg = np.clip(q_cmd_deg, self.manager.lower_limit_deg, self.manager.upper_limit_deg)
                q_safe_deg = np.clip(q_safe_deg, 0.0, 180.0)

                # 8. Send packet to Arduino
                if self.arduino:
                    self.arduino.send_command(q_safe_deg.astype(int).tolist())

                # Update state estimate for next iteration
                current_q = np.deg2rad(q_safe_deg)

                # Progress logging every 10 steps (~0.2s)
                if step % 10 == 0 or step == num_steps - 1:
                    print(
                        f"t={time_instance:5.2f}s | "
                        f"Target XYZ: {np.round(state['x_des'], 3)} | "
                        f"Error XYZ: {np.round(control_input['err_x'], 4)} | "
                        f"Cmd Deg: {q_safe_deg.astype(int).tolist()}"
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
                    self.arduino.send_command(self.manager.home_pos_deg.astype(int).tolist())
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