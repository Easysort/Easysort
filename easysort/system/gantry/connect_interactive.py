import sys
import termios
import tty

from easysort.common.environment import Environment
from easysort.system.camera.camera_connector import CameraConnector
from easysort.system.gantry.connector import GantryConnector


def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


if __name__ == "__main__":
    connector = GantryConnector(Environment.GANTRY_PORT, CameraConnector())
    x, y, z = 0, 0, 0

    while True:
        key = get_key()
        if key.isupper():
            scalar = 1
        else:
            scalar = 5

        if key.lower() == "w":
            y += scalar
            print(f"Moving up: ({x}, {y}, {z})")
        elif key.lower() == "a":
            x -= scalar
            print(f"Moving left: ({x}, {y}, {z})")
        elif key.lower() == "s":
            y -= scalar
            print(f"Moving down: ({x}, {y}, {z})")
        elif key.lower() == "d":
            x += scalar
            print(f"Moving right: ({x}, {y}, {z})")
        elif key.lower() == "q":
            z -= scalar
            print(f"Moving down: ({x}, {y}, {z})")
        elif key.lower() == "e":
            z += scalar
            print(f"Moving up: ({x}, {y}, {z})")
        elif key.lower() == "o":
            if connector.suction_state:
                connector.suction_off(x, y, z)
                print(f"Suction off: ({x}, {y}, {z})")
            else:
                connector.suction_on(x, y, z)
                print(f"Suction on: ({x}, {y}, {z})")
        elif key.lower() == "x":
            connector.quit(return_to_start=False)
            break
        elif key.lower() == "t":
            connector.go_to(x - 37, y, z + 12)
        connector.go_to(x, y, z)
        print(f"Position: ({x}, {y}, {z}), Suction: {'On' if connector.suction_state else 'Off'}")
