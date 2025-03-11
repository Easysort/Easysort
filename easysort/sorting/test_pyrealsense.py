

import pyrealsense2 as rs
import numpy as np
import cv2
import logging
import time
def run():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    pipeline.start(config)

    while True:
        frame = pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        depth_frame = frame.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow("Color", color_image)
        cv2.imshow("Depth", depth_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def list_devices():
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    try:
        # Get a context object
        ctx = rs.context()
        time.sleep(1)
        logger.info("RealSense Context created successfully")

        # Get all connected devices
        devices = ctx.query_devices()
        logger.info(f"Found {devices.size()} connected devices")

        if devices.size() == 0:
            logger.warning("No RealSense devices connected!")
            return

        # Print information about each device
        for i in range(devices.size()):
            try:
                device = devices[i]
                print(f"\nDevice {i}:")
                print(f"    Name: {device.get_info(rs.camera_info.name)}")
                print(f"    Serial Number: {device.get_info(rs.camera_info.serial_number)}")
                print(f"    Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")
                print(f"    USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")

                # Try to enable some streams to test device access
                config = rs.config()
                config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

                pipeline = rs.pipeline()
                logger.info("Attempting to start pipeline...")
                pipeline.start(config)
                logger.info("Pipeline started successfully")
                pipeline.stop()
                logger.info("Pipeline stopped successfully")

            except Exception as e:
                logger.error(f"Error accessing device {i}: {str(e)}")

    except Exception as e:
        logger.error(f"Error in list_devices: {str(e)}")

if __name__ == "__main__":
    list_devices()




