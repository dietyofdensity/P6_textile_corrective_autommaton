#!/usr/bin/env python3
import rclpy #Import ROS 2 Python client library
from rclpy.node import Node  #Import Node class from rclpy
from sensor_msgs.msg import Image  #Import Image message type from sensor_msgs
from cv_bridge import CvBridge  #Import CvBridge to convert OpenCV images to ROS Image messages
from vmbpy import VmbSystem #Import Vimba Python API for camera control
import threading  #Import threading for handling input in a separate thread
import time  #Import time for sleep functionality
import os  #Import os for file path operations
import cv2  #Import OpenCV for image processing


# This script captures images from a Mako camera using the Vimba Python API and publishes them as ROS 2 Image messages.

class MakoPublisher(Node): 
    def __init__(self): #Initialize the MakoPublisher node
        super().__init__('mako_publisher') 
        self.publisher_ = self.create_publisher(Image, '/camera/camera/color/image_raw', 10) #Create a publisher for Image messages on the specified topic
        self.bridge = CvBridge() #Create a CvBridge instance for converting OpenCV images to ROS Image messages
        self.running = True # Flag to control the main loop

        # Initialize Vimba
        self.vmb = VmbSystem.get_instance()
        self.vmb.__enter__()

        # Get camera
        cams = self.vmb.get_all_cameras()
        if not cams:
            self.get_logger().error("No cameras found.")
            self.running = False
            return

        self.cam = cams[0] 
        self.cam.__enter__()

        self.get_logger().info("MakoPublisher ready. Type 'n' to capture a frame or 'q' to quit.")

        # Start input thread
        self.input_thread = threading.Thread(target=self.input_loop)
        self.input_thread.daemon = True
        self.input_thread.start()

    def input_loop(self):
        frame_counter = 1  # Counter for naming saved frames
        save_directory = "/home/melina/p6_Minik/Calibration images2"
        os.makedirs(save_directory, exist_ok=True)  # Ensure the directory exists

        while self.running:
            try:
                cmd = input("Enter command (n=capture, s=save, q=quit): ").strip().lower()
                if cmd == 'n':
                    self.get_logger().info("Capturing frame...")
                    self.capture_and_publish_frame()
                elif cmd == 's':
                    self.get_logger().info("Saving frame...")
                    try:
                        frame = self.cam.get_frame()  # Request a single frame
                        img = frame.as_opencv_image() # Convert to OpenCV image
                        filename = os.path.join(save_directory, f"frame_{frame_counter}.png") # Create a filename with a counter
                        cv2.imwrite(filename, img) # Save the OpenCV image to a file
                        self.get_logger().info(f"Frame saved as {filename}.")
                        frame_counter += 1
                    except Exception as e:
                        self.get_logger().error(f"Save error: {e}")
                elif cmd == 'q':
                    self.get_logger().info("Quit command received.")
                    self.running = False
                    rclpy.shutdown()
                    break
                else:
                    self.get_logger().info(f"Unknown command: {cmd}")
            except Exception as e:
                self.get_logger().error(f"Input error: {e}")
                time.sleep(1)

    def capture_and_publish_frame(self):
        try:
            frame = self.cam.get_frame()  # Request a single frame
            img = frame.as_opencv_image() # Convert to OpenCV image
            msg = self.bridge.cv2_to_imgmsg(img, encoding='mono8') # Convert OpenCV image to ROS Image message
            self.publisher_.publish(msg) # Publish the Image message
            self.get_logger().info("Frame captured and published.")
        except Exception as e:
            self.get_logger().error(f"Capture error: {e}")


    def destroy_node(self):
        self.running = False
        if hasattr(self, 'cam'):
            self.cam.__exit__(None, None, None)
        if hasattr(self, 'vmb'):
            self.vmb.__exit__(None, None, None)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MakoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
