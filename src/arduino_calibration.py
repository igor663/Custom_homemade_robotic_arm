import time
import csv
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        sys.exit("Error: 'tomllib' is missing. Run 'pip install tomli'.")

from arduino_manager import ArduinoManager, NUM_JOINTS

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config" / "config.toml"
print(f"Loading configuration from: {CONFIG_FILE}")

def load_config(file_path: Path) -> dict:
    """Loads and validates parameters from the TOML configuration file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, "rb") as f:
        return tomllib.load(f)

# --- CONFIGURATION ---
OUTPUT_CSV = "joint_calibration_data.csv"
# Angular positions to calibrate
TEST_ANGLES = []
SETTLING_TIME = 1.8       # Seconds to wait for physical servo stabilization
SAMPLES_TO_AVERAGE = 15   # Number of samples to average (filters out ADC noise)

# --- LOAD INFO FROM TOML ---
config = load_config(CONFIG_FILE)
SERIAL_PORT = config["serial"]["port"]         
BAUD_RATE = config["serial"]["baud_rate"]
HOME_POS = config["robot"]["home_pos"]
JOINTS_LIMITS = config["robot"]["joints_limits"]

def main():
    manager = ArduinoManager(port=SERIAL_PORT, baud=BAUD_RATE)

    try:
        manager.connect()
        print("\n--- STARTING JOINT CALIBRATION ---")

        # 1. Drive the robotic arm to home position prior to testing
        print("Moving robot to Home Position...")
        manager.send_command(HOME_POS)
        time.sleep(2.0)

        with open(OUTPUT_CSV, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(["Joint_ID", "Target_Angle_Deg", "Raw_ADC_Value", "Filtered_ADC_Mean"])

            # Iterate over each of the 5 joints
            for joint_id in range(NUM_JOINTS):
                print(f"\n==============================")
                print(f"     CALIBRATING JOINT {joint_id}")
                print(f"==============================")

                TEST_ANGLES = list(range(JOINTS_LIMITS[joint_id][0], JOINTS_LIMITS[joint_id][1], int((JOINTS_LIMITS[joint_id][1] - JOINTS_LIMITS[joint_id][0]) / NUM_JOINTS)))  # Example: 10 steps between min and max   

                for angle in TEST_ANGLES:
                    # Isolate movement to the joint under test; keep remaining joints at Home
                    command_pose = list(HOME_POS)
                    command_pose[joint_id] = angle

                    print(f"Moving Joint {joint_id} to {angle}°...")
                    manager.send_command(command_pose)

                    # Wait for mechanical settling
                    time.sleep(SETTLING_TIME)

                    # Collect N samples for statistical averaging
                    samples = []
                    for _ in range(SAMPLES_TO_AVERAGE):
                        current_telemetry = manager.get_current_positions()
                        samples.append(current_telemetry[joint_id])
                        time.sleep(0.02)  # 50 Hz sampling interval

                    # Compute filtered average and capture last raw reading
                    mean_val = sum(samples) / len(samples)
                    last_val = samples[-1]

                    print(f"-> Target: {angle:3d}° | Feedback (Mean ADC): {mean_val:.2f} (Last: {last_val})")

                    # Log results to CSV
                    writer.writerow([joint_id, angle, last_val, f"{mean_val:.2f}"])
                    file.flush()  # Force immediate write to disk

                # Return calibrated joint to Home position before moving to the next joint
                print(f"Returning Joint {joint_id} to 90°...")
                manager.send_command(HOME_POS)
                time.sleep(1.0)

        print(f"\nCalibration completed successfully! Data saved to: {OUTPUT_CSV}")

    except KeyboardInterrupt:
        print("\nCalibration aborted by user.")
    finally:
        manager.disconnect()

if __name__ == "__main__":
    main()