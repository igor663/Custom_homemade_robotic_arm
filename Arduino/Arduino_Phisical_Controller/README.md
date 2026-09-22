
# 5-DOF Robotic Arm Arduino Controller

A lightweight, non-blocking Arduino firmware designed to control a 5-DOF robotic arm as an I/O slave. Kinematics, trajectory generation, and high-level control are offloaded to a host PC, while Arduino manages servo actuation and reads analog potentiometer telemetry.



## Features

- **Bidirectional Packetized Serial:** Uses [`SerialTransfer`](https://github.com/PowerBroker2/SerialTransfer) at **115200 baud** with COBS packet framing and CRC error checking for fast, reliable data exchange.
- **50 Hz Deterministic Loop:** Operates on a fixed time (ms) cycle (`micros()` non-blocking timing), matching the native refresh rate of standard analog RC servos (e.g., MG996R).
- **Safety Deadband & Clamping:** Angle setpoints are constrained to safe limits ($0^\circ$–$180^\circ$) and filtered with a $1^\circ$ deadband threshold to prevent servo jitter.
- **Real-Time Telemetry:** Sends 10-bit raw ADC readings ($0$–$1023$) from joint feedback potentiometers back to the PC every cycle.



## Communication Protocol

Data is exchanged via packed C-structs matching 16-bit signed integer arrays (`int16_t`):

| Direction | Struct | Payload | Description |
| :--- | :--- | :--- | :--- |
| **PC $\to$ Arduino** | `STRUCT_RX` | `target_pos[5]` | Target joint angles (degrees, $0$–$180$) |
| **Arduino $\to$ PC** | `STRUCT_TX` | `current_pos[5]` | Raw analog potentiometer values ($0$–$1023$) |




## Configuration & Usage

Before uploading, assign your actual hardware pins and default home poses in the sketch:

```cpp
int Motor_Pins[NUM_JOINTS]   = {3, 5, 6, 9, 10};       // PWM pins
int Encoder_Pins[NUM_JOINTS] = {A0, A1, A2, A3, A4};  // Analog pins
int Motors_Home_Pos[NUM_JOINTS] = {90, 90, 90, 90, 90}; // Default neutral angles
unsigned long  LOOP_PERIOD = = 20000; // 20000 microsecondi = 50 Hz
```


> **Note:** Ensure `Servo Servomotors[NUM_JOINTS];` uses the `Servo` class type (instead of `int`) so the servo methods compile properly.
