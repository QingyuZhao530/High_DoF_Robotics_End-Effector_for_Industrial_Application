#include <SimpleFOC.h>
#include <Wire.h>

// 使用自定义配置初始化传感器
MagneticSensorI2C sensor = MagneticSensorI2C(AS5048_I2C);

// 2. 初始化电机和驱动 (请确认极对数是 7)
BLDCMotor motor = BLDCMotor(7);
BLDCDriver3PWM driver = BLDCDriver3PWM(6, 10, 5, 8); 

float target_angle = 2.8;
Commander command = Commander(Serial);
void doTarget(char* cmd) { command.scalar(&target_angle, cmd); }
void doMotor(char* cmd) { command.motor(&motor, cmd); }

void setup() {
  Serial.begin(115200);
  while(!Serial) _delay(10); // 等待串口准备好

  // 开启最详细的调试信息
  SimpleFOCDebug::enable(&Serial);

  // 【关键优化】提高 I2C 速度，防止读取角度太慢拖死 FOC 循环
  Wire.begin();
  Wire.setClock(400000); 

  MagneticSensorI2CConfig_s cfg = AS5048_I2C;
  cfg.chip_address = 0x43;
  sensor = MagneticSensorI2C(cfg);

  // 初始化传感器
  sensor.init();
  motor.linkSensor(&sensor);

  // 驱动器初始化
  driver.voltage_power_supply = 11.1; // 供电电压 12V
  driver.voltage_limit = 8;         // 限制驱动器最大电压输出
  if(!driver.init()){
    Serial.println("Driver init failed!");
    return;
  }
  motor.linkDriver(&driver);

  // 设置控制模式为 角度闭环
  motor.foc_modulation = FOCModulationType::SpaceVectorPWM;
  motor.controller = MotionControlType::angle;

  motor.LPF_velocity.Tf = 0.02f;
  // 角度 PID 参数 (P调小一点防止剧烈震荡)
  motor.P_angle.P = 7.0f;
  
  // 速度 PID 参数
  motor.PID_velocity.P = 1.0f;
  motor.PID_velocity.I = 1.5f;
  
  motor.voltage_limit = 6.0f;  // FOC 运行时的最大电压
  motor.velocity_limit = 5.0f; // 最大转速限制

  // 电机初始化
  motor.init();

  // 【最关键的一步】对齐传感器
  motor.voltage_sensor_align = 8.0f; // 给一点力矩保证能找准零点
  Serial.println("-----------------------------------");
  Serial.println("Starting FOC Alignment... Watch the motor!");
  
  // 运行这句代码时，电机会发出“滴”声并微微转动
  motor.initFOC();

  // 添加串口指令监听
  command.add('T', doTarget, "target angle");
  command.add('M', doMotor, "motor tuning");


  // 设置监测内容：输出目标角度和实际角度
  motor.useMonitoring(Serial);
  motor.monitor_variables = _MON_TARGET | _MON_ANGLE;

  Serial.println("Motor ready. Send 'T1.57' to move.");
  Serial.println("-----------------------------------");
  _delay(1000);
}

void loop() {
  // 这两个函数必须尽可能快地循环
  motor.loopFOC();
  motor.move(target_angle);

  // 实时打印状态，并接收串口输入
  // motor.monitor();
  command.run();
}