import numpy as np

def realsense2gantry(realsense_point: np.ndarray, depth: float) -> np.ndarray:
    """
    Convert a point from the RealSense coordinate system to the Gantry coordinate system.
    """
    REALSENSE_WIDTH = 1280
    REALSENSE_HEIGHT = 720
    REALSENSE_X_CM_ON_CONVEYOR = 90
    REALSENSE_Y_CM_ON_CONVEYOR = 49

    # Move realsense (0,0) to the center of the image
    realsense_mid_x = REALSENSE_WIDTH / 2
    realsense_mid_y = REALSENSE_HEIGHT / 2

    # Convert the pixel point to cm
    x_pixel2cm = REALSENSE_X_CM_ON_CONVEYOR / REALSENSE_WIDTH # x_pixels/cm_that_camera_sees
    y_pixel2cm = REALSENSE_Y_CM_ON_CONVEYOR / REALSENSE_HEIGHT # y_pixels/cm_that_camera_sees

    x_cm = (realsense_point[0] - realsense_mid_x) * x_pixel2cm
    y_cm = (realsense_point[1] - realsense_mid_y) * y_pixel2cm

    print(f"x_cm: {x_cm}, y_cm: {y_cm}, depth: {depth}")

    # Gantry offset:
    gantry_offset_x_cm = 6
    gantry_offset_y_cm = 5
    gantry_offset_z_cm = 48

    depth = adjust_realsense_depth(y_cm, depth)
    return np.array([x_cm + gantry_offset_x_cm, y_cm + gantry_offset_y_cm, -depth + gantry_offset_z_cm])

def adjust_realsense_depth(realsense_y_cm_midpoint: float, depth: float) -> float:
    """
    Adjust the depth of the RealSense camera to account for the pythagorean theorem.
    """
    return np.sqrt(depth**2 - realsense_y_cm_midpoint**2)

if __name__ == "__main__":

    REALSENSE_WIDTH = 1280
    REALSENSE_HEIGHT = 720

    point1 = np.array([0.5 * REALSENSE_WIDTH, 0.5 * REALSENSE_HEIGHT])
    depth1 = 48
    print("running function")
    x, y, z = realsense2gantry(point1, depth1)
    print("--- done ---")
    print("Gantry point 1:")
    print(x, y, z)
    print("---")

    point2 = np.array([0.5 * REALSENSE_WIDTH, 0 * REALSENSE_HEIGHT])
    depth2 = 48
    print("running function")
    x, y, z = realsense2gantry(point2, depth2)
    print("--- done ---")
    print("Gantry point 2:")
    print(x, y, z)
    print("---")

    point3 = np.array([0.5 * REALSENSE_WIDTH, 1 * REALSENSE_HEIGHT])
    depth3 = 48
    print("running function")
    x, y, z = realsense2gantry(point3, depth3)
    print("--- done ---")
    print("Gantry point 3:")
    print(x, y, z)
    print("---")

    point3 = np.array([0 * REALSENSE_WIDTH, 0 * REALSENSE_HEIGHT])
    depth3 = 48
    print("running function")
    x, y, z = realsense2gantry(point3, depth3)
    print("--- done ---")
    print("Gantry point 4:")
    print(x, y, z)
    print("---")

    point3 = np.array([0.2 * REALSENSE_WIDTH, 0.8 * REALSENSE_HEIGHT])
    depth3 = 48
    print("running function")
    x, y, z = realsense2gantry(point3, depth3)
    print("--- done ---")
    print("Gantry point 5:")
    print(x, y, z)
    print("---")





