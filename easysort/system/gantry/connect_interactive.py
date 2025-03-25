import sys
import termios
import tty

from easysort.system.gantry.connector import GantryConnector
from easysort.common.environment import Environment

scalar = 3

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
    connector = GantryConnector(Environment.GANTRY_PORT)
    x, y, z = 0, 0, 0

    while True:
        key = get_key()
        if key == 'w':
            y += scalar
            print(f"Moving up: ({x}, {y}, {z})")
        elif key == 'a':
            x -= scalar
            print(f"Moving left: ({x}, {y}, {z})")
        elif key == 's':
            y -= scalar
            print(f"Moving down: ({x}, {y}, {z})")
        elif key == 'd':
            x += scalar
            print(f"Moving right: ({x}, {y}, {z})")
        elif key == 'q':
            z -= scalar
            print(f"Moving down: ({x}, {y}, {z})")
        elif key == 'e':
            z += scalar
            print(f"Moving up: ({x}, {y}, {z})")
        elif key == 'o':
            if connector.suction_state:
                connector.suction_off(x, y, z)
                print(f"Suction off: ({x}, {y}, {z})")
            else:
                connector.suction_on(x, y, z)
                print(f"Suction on: ({x}, {y}, {z})")
        elif key == 'x':
            connector.quit()
            break
        connector.go_to(x, y, z)
        print(f"Position: ({x}, {y}, {z}), Suction: {'On' if connector.suction_state else 'Off'}")
