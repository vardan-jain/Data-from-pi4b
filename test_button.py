import RPi.GPIO as GPIO
import time

BUTTON_PIN = 4
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN)
while True:
    print(GPIO.input(BUTTON_PIN))
GPIO.cleanup()