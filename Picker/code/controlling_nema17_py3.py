import RPi.GPIO as GPIO
import time

dir_pin = 16  # Adjust according to your setup
step_pin = 17  # Adjust according to your setup
steps_per_revolution = 200

GPIO.setmode(GPIO.BCM)
GPIO.setup(dir_pin, GPIO.OUT)
GPIO.setup(step_pin, GPIO.OUT)

def step(channel):
    global step_pin
    GPIO.output(step_pin, not GPIO.input(step_pin))

def rotate_motor(delay):
    GPIO.add_event_detect(step_pin, GPIO.BOTH, callback=step)

    # Set up timer for stepping (no need in Python, GPIO event detect replaces it)

def loop():
    while True:
        # Set motor direction clockwise
        GPIO.output(dir_pin, GPIO.HIGH)

        # Spin motor slowly
        rotate_motor(2000)
        time.sleep(steps_per_revolution / 1000)
        GPIO.remove_event_detect(step_pin)  # stop the step event detection
        time.sleep(1)

        # Set motor direction counterclockwise
        GPIO.output(dir_pin, GPIO.LOW)

        # Spin motor quickly
        rotate_motor(1000)
        time.sleep(steps_per_revolution / 1000)
        GPIO.remove_event_detect(step_pin)  # stop the step event detection
        time.sleep(1)

if __name__ == "__main__":
    try:
        loop()
    finally:
        GPIO.cleanup()
