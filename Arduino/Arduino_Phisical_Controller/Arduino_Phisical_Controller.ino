#include <Servo.h>
#include "SerialTransfer.h"

const int NUM_JOINTS = 5;

// Saving basic informations regarding the phisical connection of motors and encoders
int  Motor_Pins[NUM_JOINTS] = {};
int  Encoder_Pins[NUM_JOINTS] = {};

// Creating the Servomotors variable, thinked to store servomotors obj
Servo Servomotors[NUM_JOINTS];

// Creating basics variables for motors control
int Motors_Desired_Pos[NUM_JOINTS] = {};
int Motors_Last_Commanded_Pos[NUM_JOINTS] = {};
int Motors_Read_Pos[NUM_JOINTS] = {};

// Enlisting variables used to monitor cycle time
unsigned long  LOOP_PERIOD = 20000; // 20000 microsecondi = 50 Hz
unsigned long prevTime = 0;

// Here we save the angular  pos the joint must have to be in the home pose, used as default position for the servomotors.
int Motors_Home_Pos[NUM_JOINTS] = {};

//Object of the Arduino library to transfer packetized data fast/reliably via UART/Serial, available at: https://github.com/PowerBroker2/SerialTransfer
SerialTransfer Data_Transfer;

// Data structure for storing data received from PC (Target positions in degrees)
struct __attribute__((__packed__)) STRUCT_RX 
{
  int16_t target_pos[NUM_JOINTS]; // Just one 16-bit int variable to represent the target angles sent by the PC
} rxData; //the name of the structure to be used throughout the rest of the programme to access it

// Data structure to sent data to the PC (Potentiometer readings)
struct __attribute__((__packed__)) STRUCT_TX 
{
  int16_t current_pos[NUM_JOINTS];
} txData; 

//Oneshot variable to make the arduino wait some time bofre running the lkoop at maximum speed, to be sure next time i want to write a new program on teh board there is always enough time after a rest for the board to connect with the pc
int pause = 0;

void setup() {
  // Initializing  serial communication and UART data trasfer
  Serial.begin(115200);
  Data_Transfer.begin(Serial);

  // Initializing servomotors' pins as an obj of the servo library to easily control By PWM the motors positions
  for (int i = 0; i < NUM_JOINTS; i++)
  {
    Servomotors[i].attach(Motor_Pins[i]);
    Servomotors[i].write(Motors_Home_Pos[i]); // Neutral pose
    Motors_Last_Commanded_Pos[i] = Motors_Home_Pos[i];
  }

  // Initialize loop timer
  prevTime = micros();
  
}

void loop() {
  //short oneshot delay to ensure that at the start of the program loop the arduino have alwasys some time were the pc can load new program.
  while (pause == 0)
  {
    delay(2000);
    pause++;
    // if it is the pause has been active, tehn i need to reset the loop timer
    prevTime = micros();
  }

  // If there are incoming packets from PC (Non-blocking check)
  if (Data_Transfer.available()) {
    // store data in the structure dedicated to received data
    Data_Transfer.rxObj(rxData);


    for (int i = 0; i < NUM_JOINTS; i++)
    {
      // Clamp angular setpoints to safe physical limits
      Motors_Desired_Pos[i] =  constrain(rxData.target_pos[i], 0, 180);

      // update servo only if change exceeds threshold to prevent unwanted motor vibrations
      if (abs(Motors_Desired_Pos[i] - Motors_Last_Commanded_Pos[i]) > 1)
      {
        // update servo position
        Servomotors[i].write(Motors_Desired_Pos[i]);
        // save last commanded position
        Motors_Last_Commanded_Pos[i] = Motors_Desired_Pos[i];
      }
    }
  }

  // 2. Loop at constnt frequency (LOOP_PERIOD Hz)
  unsigned long currentTime = micros();
  if (currentTime - prevTime >= LOOP_PERIOD)
  {
    prevTime += LOOP_PERIOD;

    // Potentiometer readings
    for (int i = 0; i < NUM_JOINTS; i++) {
      // store data in the structure dedicated to send them
      txData.current_pos[i] = analogRead(Encoder_Pins[i]);
    }

    // Transmit telemetry data to PC
    Data_Transfer.sendDatum(txData);
  }

}
