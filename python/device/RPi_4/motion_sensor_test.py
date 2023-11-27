from gpiozero import MotionSensor

pir = MotionSensor(4)

print('Motion Sensor Testing Started. Press Ctrl+C to terminate')
while True:
    pir.wait_for_motion()
    print('Motion Detected')
    pir.wait_for_no_motion()
    print('Motion Stopped')
