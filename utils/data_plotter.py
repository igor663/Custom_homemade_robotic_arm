import os
import sys
import csv
import traceback
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent if CURRENT_DIR.name in ["src", "main", "utils"] else CURRENT_DIR
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from param_manager import ParameterManager


class DataPlotter:
    """
    Utility class implementing trajectory plotting using Plotly.
    Loads data directly from a standard CSV file, operates strictly
    in 3D Cartesian space (XYZ), and saves figures as PNG and HTML.
    """

    def __init__(self):
        """Initializes the DataPlotter."""
        self.plot_dir = None
        self.log_path = None
        self.data_array = None
        self.column_names = []
        self.column_indices = {}
        self.n_dof = None
        self.time_col_name = "ts"
        self.time_data = None
        self.runs = []

        self._is_initialized = False
        print("DataPlotter: READY")

    def initialize(self, logger, pinocchio):
        """
        Initializes the DataPlotter with logger output structure and model DoFs.

        Args:
            logger: Object containing a `directory` attribute.
            pinocchio (PinocchioWrapper): Model kinematics wrapper.

        Returns:
            bool: True if initialization succeeded, False otherwise.
        """
        try:
            print("DataPlotter: Initializing...")
            if not logger or not getattr(logger, "directory", None):
                raise ValueError("Valid logger instance with 'directory' attribute is required.")
            if not pinocchio:
                raise ValueError("Valid pinocchio instance is required.")

            self.n_dof = pinocchio.n_dof

            self.plot_dir = os.path.join(logger.directory, "plots")
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
            print(f"  Plot directory set to: {self.plot_dir}")

            self._is_initialized = True
            print("DataPlotter: Initialization COMPLETE.")
            return True

        except Exception as e:
            print(f"[ERROR] DataPlotter 'initialize()': {e}")
            traceback.print_exc()
            self.__init__()
            return False

    def _load_log_data(self, log_path):
        """Loads and parses data from a standard comma-separated CSV file."""
        try:
            print(f"DataPlotter: Loading CSV file: {log_path}")
            if not os.path.exists(log_path):
                raise FileNotFoundError(f"CSV file not found: {log_path}")

            with open(log_path, mode="r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    raise ValueError("CSV file is empty or has no header.")

                self.column_names = [col.strip() for col in header]
                self.column_indices = {name: idx for idx, name in enumerate(self.column_names)}

                rows = []
                for row in reader:
                    if not row or row[0].startswith("#"):
                        continue
                    rows.append([float(val) if val != "" else np.nan for val in row])

            if not rows:
                print("  CSV file contains no numerical data rows.")
                return False

            self.data_array = np.array(rows, dtype=float)
            print(f"  Loaded data array shape: {self.data_array.shape}")

            if self.time_col_name not in self.column_indices:
                raise ValueError(f"Timestamp column '{self.time_col_name}' not found in CSV header.")

            ts_index = self.column_indices[self.time_col_name]
            absolute_time = self.data_array[:, ts_index]
            self.time_data = absolute_time - absolute_time[0]
            print(f"  Calculated relative time for entire log (duration: {self.time_data[-1]:.3f}s).")

            # Segment into a single continuous run by default
            self.runs = [{
                'type': 'trajectory',
                'start_idx': 0,
                'end_idx': len(self.data_array) - 1,
                'data': self.data_array,
                'time': self.time_data
            }]

            return True

        except Exception as e:
            print(f"[ERROR] DataPlotter failed to parse CSV '{log_path}': {e}")
            traceback.print_exc()
            self._reset_data()
            return False

    def _reset_data(self):
        """Clears loaded data and run information."""
        self.data_array = None
        self.column_names = []
        self.column_indices = {}
        self.time_data = None
        self.runs = []

    def _save_fig(self, fig: go.Figure, name):
        """Saves a Plotly figure as high-resolution PNG and interactive HTML."""
        if not self.plot_dir:
            print("[ERROR] Plot directory not set. Cannot save figure.")
            return
        try:
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("{", "").replace("}", "")
            base_path = os.path.join(self.plot_dir, safe_name)
            png_path = f"{base_path}.png"
            html_path = f"{base_path}.html"
            print(f"  Saving plot: {png_path}")
            fig.write_image(png_path, width=1200, height=700, scale=2)
            fig.write_html(html_path, auto_open=False)
        except Exception as e:
            print(f"[WARNING] Failed to save plot '{name}': {e}")

    def _get_col_index(self, col_name):
        """Safely gets column index, returning None if not present."""
        return self.column_indices.get(col_name)

    def _get_cols_data(self, col_names, data_source=None):
        """Returns numerical data array for requested column names."""
        if data_source is None:
            data_source = self.data_array

        if data_source is None or data_source.size == 0:
            return None

        indices = [self._get_col_index(name) for name in col_names]
        if None in indices:
            return None
        try:
            return data_source[:, indices]
        except Exception as e:
            print(f"[WARNING] Error accessing columns {col_names}: {e}")
            return None

    def _check_valid_data(self, col_names, data_source=None):
        """Checks if all requested columns exist and contain at least one non-NaN value."""
        data_slice = self._get_cols_data(col_names, data_source)
        if data_slice is None:
            return False
        return not np.all(np.isnan(data_slice))

    def plot_data(self, log_path):
        """Loads data from the given CSV log file and generates all plots."""
        if not self._is_initialized:
            print("[ERROR] DataPlotter not initialized.")
            return False

        self.log_path = log_path
        if not self._load_log_data(self.log_path):
            print("[ERROR] Failed to load CSV data. Aborting plotting.")
            return False

        print("DataPlotter: Generating plots...")
        try:
            if self.time_data is None or len(self.time_data) == 0:
                print("[ERROR] Relative time data is not available.")
            else:
                print("\n--- Generating Time-Series Plots ---")
                self.plot_loop_timing()
                self.plot_traj_pos_time()
                self.plot_traj_err_pos()
                self.plot_traj_vel_time()
                self.plot_traj_err_vel()
                self.plot_joint_pos()
                self.plot_joint_err_pos()
                self.plot_joint_vel()
                self.plot_joint_err_vel()
                self.plot_joint_tau()
                self.plot_joint_err_tau()
                self.plot_control_cmd_pos()
                self.plot_control_cmd_vel()
                self.plot_control_cmd_eff()

                print("\n--- Generating Conditional Plots ---")
                self.plot_gravity_torques()
                self.plot_target_vs_desired_accel()

            if self.runs:
                print("\n--- Generating 2D Cartesian Trajectory Plots ---")
                self.plot_traj_pos_2d()
                self.plot_traj_vel_2d()

            print("\nDataPlotter: Plot generation completed successfully.")
            return True

        except Exception as e:
            print(f"[ERROR] Error during plot generation: {e}")
            traceback.print_exc()
            return False
        finally:
            self._reset_data()

    # --- Time-Series Plotting Methods ---

    def plot_loop_timing(self):
        """Plots the execution duration of each control cycle."""
        if not self._check_valid_data(["loop_time"]):
            print("Skipping plot_loop_timing: 'loop_time' column not available.")
            return

        print("Plotting Loop Timing...")
        fig = go.Figure()
        loop_time_ms = self._get_cols_data(["loop_time"]) * 1000.0
        avg_time = np.nanmean(loop_time_ms)

        fig.add_trace(go.Scatter(x=self.time_data, y=loop_time_ms[:, 0], mode="lines", name="Loop Time", line=dict(color="blue")))
        fig.add_hline(y=avg_time, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_time:.3f} ms", annotation_position="bottom right")

        fig.update_layout(
            title="Control Loop Execution Time",
            xaxis_title="Time [s]", yaxis_title="Duration [ms]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "00_loop_timing")

    def plot_traj_pos_time(self):
        """Plots Cartesian position components (X, Y, Z) vs time."""
        act_cols = [f"x_act_{i}" for i in range(3)]
        des_cols = [f"x_des_{i}" for i in range(3)]
        target_cols = [f"x_target_{i}" for i in range(3)]

        if not self._check_valid_data(act_cols) or not self._check_valid_data(des_cols):
            print("Skipping plot_traj_pos_time: Cartesian position data missing.")
            return

        print("Plotting Trajectory Position vs Time (XYZ)...")
        act_data = self._get_cols_data(act_cols)
        des_data = self._get_cols_data(des_cols)
        has_target = self._check_valid_data(target_cols)
        target_data = self._get_cols_data(target_cols) if has_target else None

        fig = go.Figure()
        colors_act = ["red", "blue", "green"]
        colors_des = ["violet", "grey", "orange"]
        labels = ["X", "Y", "Z"]

        for i in range(3):
            fig.add_trace(go.Scatter(x=self.time_data, y=act_data[:, i], mode="lines", name=f"Actual {labels[i]}", line=dict(color=colors_act[i])))
            fig.add_trace(go.Scatter(x=self.time_data, y=des_data[:, i], mode="lines", name=f"Desired {labels[i]}", line=dict(dash="dash", color=colors_des[i])))
            if has_target and target_data is not None and i < target_data.shape[1]:
                target_val = target_data[0, i]
                if not np.isnan(target_val):
                    fig.add_trace(go.Scatter(x=self.time_data, y=np.full(len(self.time_data), target_val), mode="lines",
                                             name=f"Target {labels[i]}", line=dict(dash="dot", color="grey"), opacity=0.6))

        fig.update_layout(
            title="End-Effector Position vs Time (XYZ)",
            xaxis_title="Time [s]", yaxis_title="Position [m]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "01_traj_pos_time")

    def plot_traj_err_pos(self):
        """Plots Cartesian tracking error (X, Y, Z) and Euclidean norm vs time."""
        err_cols = [f"err_x_{i}" for i in range(3)]
        if not self._check_valid_data(err_cols):
            print("Skipping plot_traj_err_pos: Cartesian error data missing.")
            return

        print("Plotting Cartesian Position Error vs Time (XYZ)...")
        err_data = self._get_cols_data(err_cols)
        fig = go.Figure()
        labels = ["X", "Y", "Z"]
        colors_err = ["purple", "yellow", "orange"]
        err_norm = np.linalg.norm(err_data, axis=1)

        for i in range(3):
            fig.add_trace(go.Scatter(x=self.time_data, y=err_data[:, i], mode="lines", name=f"Err {labels[i]}", line=dict(color=colors_err[i]), opacity=0.6))
        fig.add_trace(go.Scatter(x=self.time_data, y=err_norm, mode="lines", name="Norm", line=dict(color="red", width=2)))

        fig.update_layout(
            title="End-Effector Position Tracking Error vs Time",
            xaxis_title="Time [s]", yaxis_title="Error [m]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "02_traj_err_pos")

    def plot_traj_vel_time(self):
        """Plots Cartesian linear velocity components (X, Y, Z) vs time."""
        act_cols = [f"x_dot_act_{i}" for i in range(3)]
        des_cols = [f"x_dot_des_{i}" for i in range(3)]

        if not self._check_valid_data(act_cols) or not self._check_valid_data(des_cols):
            print("Skipping plot_traj_vel_time: Velocity data missing.")
            return

        print("Plotting Trajectory Velocity vs Time (XYZ)...")
        act_data = self._get_cols_data(act_cols)
        des_data = self._get_cols_data(des_cols)

        fig = go.Figure()
        colors_act = ["pink", "blue", "green"]
        colors_des = ["yellow", "red", "orange"]
        labels = ["X", "Y", "Z"]

        for i in range(3):
            fig.add_trace(go.Scatter(x=self.time_data, y=act_data[:, i], mode="lines", name=f"Actual Vel {labels[i]}", line=dict(color=colors_act[i])))
            fig.add_trace(go.Scatter(x=self.time_data, y=des_data[:, i], mode="lines", name=f"Desired Vel {labels[i]}", line=dict(dash="dash", color=colors_des[i])))

        fig.update_layout(
            title="End-Effector Velocity vs Time (XYZ)",
            xaxis_title="Time [s]", yaxis_title="Velocity [m/s]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "03_traj_vel_time")

    def plot_traj_err_vel(self):
        """Plots Cartesian velocity error components vs time."""
        err_cols = [f"err_x_dot_{i}" for i in range(3)]
        if not self._check_valid_data(err_cols):
            print("Skipping plot_traj_err_vel: Velocity error data missing.")
            return

        print("Plotting Cartesian Velocity Error vs Time...")
        err_data = self._get_cols_data(err_cols)
        fig = go.Figure()
        err_norm = np.linalg.norm(err_data, axis=1)
        labels = ["X", "Y", "Z"]
        colors_err = ["violet", "yellow", "orange"]

        for i in range(3):
            fig.add_trace(go.Scatter(x=self.time_data, y=err_data[:, i], mode="lines", name=f"Err Vel {labels[i]}", line=dict(color=colors_err[i]), opacity=0.6))
        fig.add_trace(go.Scatter(x=self.time_data, y=err_norm, mode="lines", name="Norm", line=dict(color="red", width=2)))

        fig.update_layout(
            title="End-Effector Velocity Error vs Time",
            xaxis_title="Time [s]", yaxis_title="Error [m/s]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "04_traj_err_vel")

    def plot_joint_pos(self):
        """Plots actual and desired joint angles vs time."""
        act_cols = [f"q_act_{i}" for i in range(self.n_dof)]
        des_cols = [f"q_des_{i}" for i in range(self.n_dof)]
        target_cols = [f"q_target_{i}" for i in range(self.n_dof)]

        if not self._check_valid_data(act_cols) or not self._check_valid_data(des_cols):
            print("Skipping plot_joint_pos: Joint position data missing.")
            return

        print("Plotting Joint Positions vs Time...")
        act_data = self._get_cols_data(act_cols)
        des_data = self._get_cols_data(des_cols)
        has_target = self._check_valid_data(target_cols)
        target_data = self._get_cols_data(target_cols) if has_target else None

        fig = go.Figure()
        colors_act = ["blue", "green", "purple", "brown", "pink"]
        colors_des = ["red", "orange", "black", "yellow", "lime"]

        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=act_data[:, i], mode="lines", name=f"q{i+1}_act", line=dict(color=colors_act[i % len(colors_act)])))
            fig.add_trace(go.Scatter(x=self.time_data, y=des_data[:, i], mode="lines", name=f"q{i+1}_des", line=dict(dash="dash", color=colors_des[i % len(colors_des)])))
            if has_target and target_data is not None and i < target_data.shape[1]:
                target_val = target_data[0, i]
                if not np.isnan(target_val):
                    fig.add_trace(go.Scatter(x=self.time_data, y=np.full(len(self.time_data), target_val), mode="lines",
                                             name=f"q{i+1}_target", line=dict(dash="dot", color="grey"), opacity=0.5))

        fig.update_layout(
            title="Joint Positions vs Time",
            xaxis_title="Time [s]", yaxis_title="Position [rad]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "05_joint_pos")

    def plot_joint_err_pos(self):
        """Plots joint position tracking error vs time."""
        err_cols = [f"err_q_{i}" for i in range(self.n_dof)]
        if not self._check_valid_data(err_cols):
            print("Skipping plot_joint_err_pos: Joint error data missing.")
            return

        print("Plotting Joint Position Error vs Time...")
        err_data = self._get_cols_data(err_cols)
        fig = go.Figure()
        err_norm = np.linalg.norm(err_data, axis=1)
        colors_err = ["yellow", "orange", "lime", "pink", "cyan"]

        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=err_data[:, i], mode="lines", name=f"q{i+1}_err", line=dict(color=colors_err[i % len(colors_err)]), opacity=0.6))
        fig.add_trace(go.Scatter(x=self.time_data, y=err_norm, mode="lines", name="Norm", line=dict(color="red", width=2)))

        fig.update_layout(
            title="Joint Position Tracking Error vs Time",
            xaxis_title="Time [s]", yaxis_title="Error [rad]",
            showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
        )
        self._save_fig(fig, "06_joint_err_pos")

    def plot_joint_vel(self):
        """Plots individual joint velocities in subplots."""
        from plotly.subplots import make_subplots

        act_cols = [f"q_dot_act_{i}" for i in range(self.n_dof)]
        des_cols = [f"q_dot_des_{i}" for i in range(self.n_dof)]

        if not self._check_valid_data(act_cols) or not self._check_valid_data(des_cols):
            print("Skipping plot_joint_vel: Joint velocity data missing.")
            return

        print("Plotting Joint Velocities vs Time (Subplots)...")
        act_data = self._get_cols_data(act_cols)
        des_data = self._get_cols_data(des_cols)

        fig = make_subplots(
            rows=self.n_dof,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"Joint {i+1} Velocity" for i in range(self.n_dof)],
            vertical_spacing=0.05
        )

        for i in range(self.n_dof):
            row_idx = i + 1
            fig.add_trace(go.Scatter(x=self.time_data, y=act_data[:, i], mode="lines", name="Actual", line=dict(color="blue", width=1.5), legendgroup="act", showlegend=(i == 0)), row=row_idx, col=1)
            fig.add_trace(go.Scatter(x=self.time_data, y=des_data[:, i], mode="lines", name="Desired", line=dict(color="red", dash="dash", width=2.0), legendgroup="des", showlegend=(i == 0)), row=row_idx, col=1)
            fig.update_yaxes(title_text="[rad/s]", row=row_idx, col=1)

        fig.update_xaxes(title_text="Time [s]", row=self.n_dof, col=1)
        fig.update_layout(height=250 * self.n_dof, title_text="Joint Velocities vs Time", hovermode="x unified")
        self._save_fig(fig, "07_joint_vel_plots")

    def plot_joint_err_vel(self):
        """Plots joint velocity tracking error vs time."""
        err_cols = [f"err_q_dot_{i}" for i in range(self.n_dof)]
        if not self._check_valid_data(err_cols):
            print("Skipping plot_joint_err_vel: Data not available.")
            return

        print("Plotting Joint Velocity Error vs Time...")
        err_data = self._get_cols_data(err_cols)
        fig = go.Figure()
        err_norm = np.linalg.norm(err_data, axis=1)

        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=err_data[:, i], mode="lines", name=f"q{i+1}_dot_err", opacity=0.6))
        fig.add_trace(go.Scatter(x=self.time_data, y=err_norm, mode="lines", name="Norm", line=dict(color="red", width=2)))

        fig.update_layout(title="Joint Velocity Tracking Error vs Time", xaxis_title="Time [s]", yaxis_title="Error [rad/s]")
        self._save_fig(fig, "08_joint_err_vel")

    def plot_joint_tau(self):
        """Plots joint effort/torque data if available."""
        act_cols = [f"tau_act_{i}" for i in range(self.n_dof)]
        cmd_cols = [f"cmd_eff_{i}" for i in range(self.n_dof)]

        if not self._check_valid_data(act_cols):
            print("Skipping plot_joint_tau: 'tau_act' data not logged.")
            return

        print("Plotting Joint Torques vs Time...")
        act_data = self._get_cols_data(act_cols)
        has_cmd = self._check_valid_data(cmd_cols)
        cmd_data = self._get_cols_data(cmd_cols) if has_cmd else None

        fig = go.Figure()
        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=act_data[:, i], mode="lines", name=f"tau{i+1}_act"))
            if has_cmd and cmd_data is not None:
                fig.add_trace(go.Scatter(x=self.time_data, y=cmd_data[:, i], mode="lines", name=f"tau{i+1}_cmd", line=dict(dash="dash")))

        fig.update_layout(title="Joint Torques vs Time", xaxis_title="Time [s]", yaxis_title="Torque [Nm]")
        self._save_fig(fig, "09_joint_tau")

    def plot_joint_err_tau(self):
        """Plots joint effort error if available."""
        act_cols = [f"tau_act_{i}" for i in range(self.n_dof)]
        cmd_cols = [f"cmd_eff_{i}" for i in range(self.n_dof)]

        if not self._check_valid_data(act_cols) or not self._check_valid_data(cmd_cols):
            return

        act_data = self._get_cols_data(act_cols)
        cmd_data = self._get_cols_data(cmd_cols)
        err_data = cmd_data - act_data

        fig = go.Figure()
        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=err_data[:, i], mode="lines", name=f"tau{i+1}_err"))

        fig.update_layout(title="Joint Torque Error vs Time", xaxis_title="Time [s]", yaxis_title="Error [Nm]")
        self._save_fig(fig, "10_joint_err_tau")

    def plot_control_cmd_pos(self):
        """Plots commanded joint positions."""
        cmd_cols = [f"cmd_pos_{i}" for i in range(self.n_dof)]
        if not self._check_valid_data(cmd_cols):
            return

        print("Plotting Control Command (Position)...")
        cmd_data = self._get_cols_data(cmd_cols)
        fig = go.Figure()
        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=cmd_data[:, i], mode="lines", name=f"q{i+1}_cmd"))

        fig.update_layout(title="Control Command (Position) vs Time", xaxis_title="Time [s]", yaxis_title="Position [rad]")
        self._save_fig(fig, "11_control_cmd_pos")

    def plot_control_cmd_vel(self):
        """Plots commanded joint velocities."""
        cmd_cols = [f"cmd_vel_{i}" for i in range(self.n_dof)]
        if not self._check_valid_data(cmd_cols):
            return

        print("Plotting Control Command (Velocity)...")
        cmd_data = self._get_cols_data(cmd_cols)
        fig = go.Figure()
        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=cmd_data[:, i], mode="lines", name=f"q{i+1}_dot_cmd"))

        fig.update_layout(title="Control Command (Velocity) vs Time", xaxis_title="Time [s]", yaxis_title="Velocity [rad/s]")
        self._save_fig(fig, "12_control_cmd_vel")

    def plot_control_cmd_eff(self):
        """Plots commanded joint efforts/torques."""
        cmd_cols = [f"cmd_eff_{i}" for i in range(self.n_dof)]
        if not self._check_valid_data(cmd_cols):
            return

        cmd_data = self._get_cols_data(cmd_cols)
        fig = go.Figure()
        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=cmd_data[:, i], mode="lines", name=f"tau{i+1}_cmd"))

        fig.update_layout(title="Control Command (Effort) vs Time", xaxis_title="Time [s]", yaxis_title="Torque [Nm]")
        self._save_fig(fig, "13_control_cmd_eff")

    def plot_gravity_torques(self):
        """Plots gravity compensation torques if present."""
        grav_cols = [f"tau_g_{i}" for i in range(self.n_dof)]
        if not self._check_valid_data(grav_cols):
            return

        grav_data = self._get_cols_data(grav_cols)
        fig = go.Figure()
        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=grav_data[:, i], mode="lines", name=f"tau_g{i+1}"))

        fig.update_layout(title="Gravity Compensation vs Time", xaxis_title="Time [s]", yaxis_title="Torque [Nm]")
        self._save_fig(fig, "90_gravity_torques")

    def plot_target_vs_desired_accel(self):
        """Plots acceleration targets vs desired acceleration if present."""
        target_cols = [f"a_target_{i}" for i in range(self.n_dof)]
        des_cols = [f"q_ddot_des_{i}" for i in range(self.n_dof)]

        if not self._check_valid_data(target_cols) or not self._check_valid_data(des_cols):
            return

        target_data = self._get_cols_data(target_cols)
        des_data = self._get_cols_data(des_cols)
        fig = go.Figure()

        for i in range(self.n_dof):
            fig.add_trace(go.Scatter(x=self.time_data, y=target_data[:, i], mode="lines", name=f"a_target_{i+1}"))
            fig.add_trace(go.Scatter(x=self.time_data, y=des_data[:, i], mode="lines", name=f"q_ddot_des_{i+1}", line=dict(dash="dash")))

        fig.update_layout(title="Target vs Desired Acceleration", xaxis_title="Time [s]", yaxis_title="Acceleration [rad/s^2]")
        self._save_fig(fig, "91_joint_accel_comparison")

    # --- 2D Cartesian Trajectory Plotting (XY, XZ, YZ) ---

    def _plot_trajectory_plane(self, act_cols_prefix, des_cols_prefix, plot_title_component, unit, base_filename_prefix, target_cols_prefix=None):
        """Plots 2D projections (XY, XZ, YZ) of Cartesian trajectories."""
        all_act_cols = [f"{act_cols_prefix}{i}" for i in range(3)]
        all_des_cols = [f"{des_cols_prefix}{i}" for i in range(3)]

        if not self._check_valid_data(all_act_cols) or not self._check_valid_data(all_des_cols):
            return

        act_data = self._get_cols_data(all_act_cols)
        des_data = self._get_cols_data(all_des_cols)

        has_target = False
        target_data = None
        if target_cols_prefix:
            all_target_cols = [f"{target_cols_prefix}{i}" for i in range(3)]
            has_target = self._check_valid_data(all_target_cols)
            target_data = self._get_cols_data(all_target_cols) if has_target else None

        planes = [
            {"idx1": 0, "idx2": 1, "l1": "X", "l2": "Y", "suffix": "xy"},
            {"idx1": 0, "idx2": 2, "l1": "X", "l2": "Z", "suffix": "xz"},
            {"idx1": 1, "idx2": 2, "l1": "Y", "l2": "Z", "suffix": "yz"},
        ]

        for p in planes:
            i1, i2 = p["idx1"], p["idx2"]
            l1, l2 = p["l1"], p["l2"]
            suffix = p["suffix"]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=act_data[:, i1], y=act_data[:, i2], mode="lines", name="Actual", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=des_data[:, i1], y=des_data[:, i2], mode="lines", name="Desired", line=dict(dash="dash", color="red")))

            if has_target and target_data is not None:
                fig.add_trace(go.Scatter(x=[target_data[0, i1]], y=[target_data[0, i2]], mode="markers", marker=dict(symbol="x", color="red", size=10), name="Target"))

            fig.update_layout(
                title=f"Trajectory {plot_title_component} ({l1}{l2} Plane)",
                xaxis_title=f"{l1}-axis {unit}", yaxis_title=f"{l2}-axis {unit}",
                yaxis_scaleanchor="x", yaxis_scaleratio=1,
                showlegend=True, xaxis_showgrid=True, yaxis_showgrid=True
            )
            self._save_fig(fig, f"{base_filename_prefix}_{suffix}")

    def plot_traj_pos_2d(self):
        """Generates XY, XZ, YZ Cartesian position projections."""
        print("Plotting Cartesian Trajectory Position Planes (XY, XZ, YZ)...")
        self._plot_trajectory_plane("x_act_", "x_des_", "Position", "[m]", "traj_pos_2d", target_cols_prefix="x_target_")

    def plot_traj_vel_2d(self):
        """Generates XY, XZ, YZ Cartesian velocity projections."""
        print("Plotting Cartesian Trajectory Velocity Planes (XY, XZ, YZ)...")
        self._plot_trajectory_plane("x_dot_act_", "x_dot_des_", "Velocity", "[m/s]", "traj_vel_2d")


if __name__ == "__main__":
    from wrapper_pinocchio import PinocchioWrapper

    print("=" * 60)
    print("           EXECUTING DATA PLOTTER STANDALONE MAIN          ")
    print("=" * 60)

    print("\n[1/3] Loading ParameterManager...")
    manager = ParameterManager()

    print("\n[2/3] Initializing PinocchioWrapper...")
    pinocchio = PinocchioWrapper()
    if not pinocchio.initialize(manager):
        print("[ERROR] Failed to initialize PinocchioWrapper. Aborting.")
        sys.exit(1)

    log_dir = Path(manager.log_path).resolve()
    log_file = log_dir / manager.log_file_name

    print(f"\n[3/3] Target log directory: {log_dir}")
    print(f"      Target log file:      {log_file}")

    if not log_file.exists():
        print(f"[ERROR] Target log file does not exist: {log_file}")
        sys.exit(1)

    class DummyLogger:
        def __init__(self, directory):
            self.directory = str(directory)

    plotter = DataPlotter()
    if not plotter.initialize(logger=DummyLogger(directory=log_dir), pinocchio=pinocchio):
        print("[ERROR] DataPlotter initialization failed.")
        sys.exit(1)

    print(f"\nGenerating all plots for: {log_file.name}...")
    success = plotter.plot_data(str(log_file))

    if success:
        print("\n" + "=" * 60)
        print("[SUCCESS] All plots generated successfully!")
        print(f"Output directory: {plotter.plot_dir}")
        print("=" * 60)
    else:
        print("\n[ERROR] Plotting failed. Check errors above.")
        sys.exit(1)