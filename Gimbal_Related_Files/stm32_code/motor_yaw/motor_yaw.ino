#include <SimpleFOC.h>
#include <Wire.h>

// --- AS5048B I2C 传感器配置（AS5048_I2C 为库内默认配置模板） ---
MagneticSensorI2C sensor = MagneticSensorI2C(AS5048_I2C);

// --- 电机/驱动：极对数必须是整数 11 ---
BLDCMotor motor = BLDCMotor(11);
BLDCDriver3PWM driver = BLDCDriver3PWM(10, 5, 6, 8);  // A,B,C,EN

float target_angle = 0.3f;

Commander command = Commander(Serial);
void doTarget(char* cmd) { command.scalar(&target_angle, cmd); }
void doMotor(char* cmd)  { command.motor(&motor, cmd); }

void setup() {
  Serial.begin(115200);
  while(!Serial) _delay(10);

  SimpleFOCDebug::enable(&Serial);

  // I2C 提速（要在 sensor.init() 前）                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
  Wire.begin();
  Wire.setClock(400000);

  // 用 AS5048_I2C 模板 + 覆盖地址为 0x40
  MagneticSensorI2CConfig_s cfg = AS5048_I2C;
  cfg.chip_address = 0x40;
  sensor = MagneticSensorI2C(cfg);

  // 初始化传感器并链接到电机
  sensor.init();
  motor.linkSensor(&sensor);

  // 驱动器
  driver.voltage_power_supply = 11.1f;
  driver.voltage_limit = 6.0f;     // 驱动输出上限
  if(!driver.init()){
    Serial.println("Driver init failed!");
    while(1) {}
  }
  motor.linkDriver(&driver);

  // 控制与调制
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.controller = MotionControlType::angle;
  motor.torque_controller = TorqueControlType::voltage; // 无电流采样 -> 电压力矩

  // 滤波/参数（按你原来的保留，略微做了合理性补充）
  motor.LPF_velocity.Tf = 0.02f;

  motor.P_angle.P = 7.0f;

  motor.PID_velocity.P = 0.5f;
  motor.PID_velocity.I = 2.0f;
  motor.PID_velocity.D = 0.0f;

  motor.voltage_limit = 4.0f;         // FOC 运行电压上限（比 driver.voltage_limit 更小）
  motor.velocity_limit = 5.0f;       // rad/s

  // 初始化电机
  motor.init();

  // 对齐：给足够电压，避免对齐不动导致角度/方向判断漂
  motor.voltage_sensor_align = 6.0f;

  Serial.println("-----------------------------------");
  Serial.println("Starting FOC Alignment... Watch the motor!");

  motor.initFOC();

  // Commander
  command.add('T', doTarget, "target angle (rad)");
  command.add('M', doMotor,  "motor tuning");

  // 监控：目标角 + 实际角
  motor.useMonitoring(Serial);
  motor.monitor_variables = _MON_TARGET | _MON_ANGLE;

  Serial.println("Motor ready. Send 'T1.57' to move.");
  Serial.println("-----------------------------------");
  _delay(1000);
}

void loop() {
  motor.loopFOC();
  motor.move(target_angle);

  command.run();
  // motor.monitor();  // 需要连续打印时再打开，否则会拖慢 loopFOC
}