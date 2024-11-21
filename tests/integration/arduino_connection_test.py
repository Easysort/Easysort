import sys
sys.path.append("/Users/lucasvilsen/Documents/Documents/EasySort")

import time

# To test this file run the this py file while having
# 'arduino_connection_test_motor.ino' running on motor arduino &
# 'arduino_connection_test_suction_cup.ino' running on suction cup arduino
# and have both connected to this computer with usb.

# Or start by having "..._motor.ino" run on one arduino &
# And then change to "..._suction_cup.ino" on the other when the first test has run (script will pause for you)

# The only main difference between these script is delay(1000) to properly test the async.


import asyncio
from easysort.helpers.orchestrator import Orchestrator


MOTOR_ARDUINO = "/dev/cu.usbmodem1401"
SUCTION_CUP_ARDUINO = "/dev/cu.usbmodem1401"

def some_processing(processing_index: int) -> int:
    # Should output something and take about 1 second
    print("Processing started")
    time.sleep(0.01)
    print("Processing ended, the sum of 1 to 1000000 is ", sum(range(1, 1000000)))   
    return processing_index + 1 


class TestArduinoMotorConnection:
    async def test_async_connection(self):
        orchestrator = Orchestrator('Dave', connect_suction_cup=False, _overwrite_config={'motor_arduino': MOTOR_ARDUINO})
        start_time = time.time()
        navigation_task = asyncio.create_task(orchestrator.motor_arduino.navigate_to(10, 10))
        processing_index = 0

        while not navigation_task.done() or time.time() - start_time > 10:
            processing_index = some_processing(processing_index)
            await asyncio.sleep(0.1)  # Allow other tasks to run

        assert navigation_task.done()
        assert orchestrator.database.is_valid_movement_message(navigation_task.result())
        assert processing_index > 4
        assert time.time() - start_time < 10

        print(f"Test comeplete {time.time() - start_time:.2f} seconds")
        orchestrator.quit()


class TestArduinoSuctionCupConnection:
    async def test_async_connection(self):
        # This just needs to check that a connection can be established. This is to make sure that the arduino is on.
        orchestrator = Orchestrator('Dave', connect_motor=False, _overwrite_config={'suction_cup_arduino': SUCTION_CUP_ARDUINO})
        orchestrator.quit()


if __name__ == "__main__":
    asyncio.run(TestArduinoMotorConnection().test_async_connection())
    _ = input("Press enter to continue with suction cup test")
    asyncio.run(TestArduinoSuctionCupConnection().test_async_connection())
