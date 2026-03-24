from motor_control import MiniFeetechDriver

port = "/dev/ttyUSB0"
motor_id = 1

driver = MiniFeetechDriver(port=port)

for i in range(10):
    pos_data = driver.get_position(motor_id)
    print(pos_data)