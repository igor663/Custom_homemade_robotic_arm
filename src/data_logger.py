import csv
import time
import numpy as np
from pathlib import Path


class DataLogger:
    """
    High-performance telemetry and control logger.
    Buffers step data in RAM during real-time execution and writes
    a clean, comma-separated CSV file upon completion.
    """

    def __init__(self, filename: str = "trajectory_log.csv", num_joints: int = 5):
        self.filename = Path(filename)
        self.num_joints = num_joints
        self.records = []
        self._is_header_initialized = False
        self.headers = []

    def _flatten_value(self, key: str, val):
        """Unpacks scalars, lists, and numpy arrays using numerical suffixes (_0, _1, ...)."""
        items = {}
        if val is None:
            return items

        arr = np.asarray(val)
        if arr.ndim > 1:  # Skip 2D matrices (e.g., Jacobians)
            return items

        if arr.ndim == 0:
            items[key] = float(arr)
        elif arr.ndim == 1:
            for i, elem in enumerate(arr):
                items[f"{key}_{i}"] = float(elem)
        return items

    def log_step(self, control_input: dict,
                 loop_time: float = 0.0,
                 time_instance: float = None,
                 step: int = None,
                 cmd_pos=None, cmd_vel=None, cmd_eff=None,
                 q_target=None, x_target=None, servo_cmd=None,
                 **kwargs):
        """
        Buffers one execution step into RAM.
        Accepts timestamps, loop timing, targets, and controller outputs.
        """
        row_dict = {
            "ts": time.time(),                            # Epoch timestamp for DataPlotter
            "loop_time": float(loop_time) if loop_time is not None else 0.0,
        }

        if time_instance is not None:
            row_dict["time_instance"] = float(time_instance)
        if step is not None:
            row_dict["step"] = int(step)

        # 1. Trajectory targets (if available)
        if q_target is not None:
            row_dict.update(self._flatten_value("q_target", q_target))
        if x_target is not None:
            row_dict.update(self._flatten_value("x_target", x_target))

        # 2. Control input dictionary
        if control_input:
            for k, v in control_input.items():
                if k in ["J_act", "J_pinv", "Jdot_qdot"]:
                    continue
                row_dict.update(self._flatten_value(k, v))

        # 3. Controller output setpoints
        row_dict.update(self._flatten_value("cmd_pos", cmd_pos))
        row_dict.update(self._flatten_value("cmd_vel", cmd_vel))
        row_dict.update(self._flatten_value("cmd_eff", cmd_eff))

        # 4. Command sent to physical servos
        if servo_cmd is not None:
            row_dict.update(self._flatten_value("servo_cmd", servo_cmd))

        # Initialize header list on the first call
        if not self._is_header_initialized:
            self.headers = list(row_dict.keys())
            self._is_header_initialized = True

        self.records.append(row_dict)

        
    def save_to_csv(self):
        """Flushes buffered records to a standard comma-separated CSV file."""
        if not self.records:
            print("[DataLogger] No records to save.")
            return

        print(f"\n[DataLogger] Saving {len(self.records)} steps to {self.filename}...")
        try:
            self.filename.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filename, mode="w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=self.headers)
                writer.writeheader()
                for row in self.records:
                    writer.writerow(row)

            print(f"[DataLogger] Data saved successfully to: {self.filename.resolve()}")
        except Exception as e:
            print(f"[ERROR] DataLogger failed to write CSV: {e}")