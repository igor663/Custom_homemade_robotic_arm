import time
import threading
from pySerialTransfer import pySerialTransfer as txfer

NUM_JOINTS = 5
STRUCT_FORMAT = f'{NUM_JOINTS}h'  # 5 signed 16-bit integers ('5h')

class ArduinoManager:
    def __init__(self, port: str, baud: int = 115200):
        self.port = port
        self.baud = baud
        self.link = None

        # Shared telemetry data
        self.current_pos = [0] * NUM_JOINTS
        self.data_lock = threading.Lock()     # Protects reading/writing current_pos
        self.serial_lock = threading.Lock()   # Protects access to the serial link

        # Thread control flags and primitives
        self.is_running = False
        self.rx_paused = threading.Event()    # Set: reception active; Cleared: reception paused
        self.rx_paused.set()                  # Enabled by default
        self._thread = None

    def connect(self):
        """Opens the serial link and starts the background telemetry polling thread."""
        print(f"Connecting to Arduino on {self.port} at {self.baud} baud...")
        self.link = txfer.SerialTransfer(self.port, baud=self.baud)
        self.link.open()
        time.sleep(2.5)  # Wait for Arduino bootloader and hardware auto-reset
        print("Connection established!")

        self.is_running = True
        self._thread = threading.Thread(target=self._update_loop, daemon=True)
        self._thread.start()

    def pause_reception(self):
        """Temporarily pauses the background reception thread."""
        self.rx_paused.clear()
        time.sleep(0.01)  # Brief settling delay to ensure thread is halted

    def resume_reception(self):
        """Resumes the background reception thread."""
        self.rx_paused.set()

    def send_command(self, target_angles: list[int]):
        """
        Transmits a new set of joint target angles (0-180 deg) to Arduino.
        Temporarily pauses telemetry reception to avoid serial port collisions.
        """
        if len(target_angles) != NUM_JOINTS:
            raise ValueError(f"Expected {NUM_JOINTS} angles, received {len(target_angles)}")

        # Clamp values to safe operating range (0 to 180 degrees)
        clamped_targets = [max(0, min(180, int(val))) for val in target_angles]

        # 1. Momentarily halt telemetry reception
        self.pause_reception()

        # 2. Transmit packet under serial lock protection
        with self.serial_lock:
            send_size = self.link.tx_obj(clamped_targets, format=STRUCT_FORMAT)
            self.link.send(send_size)

        # 3. Resume telemetry reception
        self.resume_reception()

    def get_current_positions(self) -> list[int]:
        """Returns a thread-safe copy of the latest potentiometer readings."""
        with self.data_lock:
            return list(self.current_pos)

    def _update_loop(self):
        """Background thread polling incoming telemetry at approximately 100 Hz."""
        while self.is_running:
            # If reception is paused, block here without consuming CPU cycles
            self.rx_paused.wait()

            with self.serial_lock:
                if self.link.available():
                    # Unpack incoming packet into telemetry array
                    data = self.link.rx_obj(obj_type=list, format=STRUCT_FORMAT)
                    with self.data_lock:
                        self.current_pos = data
                elif self.link.status < 0:
                    # Packet drop or CRC validation error
                    pass

            time.sleep(0.01)  # ~100 Hz polling period

    def disconnect(self):
        """Safely stops the thread and closes the serial port."""
        self.is_running = False
        self.rx_paused.set()  # Unblock thread if it was waiting on pause
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.link:
            self.link.close()
        print("Disconnected from Arduino.")