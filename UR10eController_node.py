import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import os
import csv


Camera_matrix = np.array([[2.94352929e+03, 0.00000000e+00, 9.98475579e+02],
                        [0.00000000e+00, 2.94681230e+03, 5.69576869e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

Camera_matrix_inv = np.linalg.inv(Camera_matrix)

distCoeffs = np.array([-2.20469671e-01, -1.84026226e-02, 1.48094161e-04, 6.83332051e-04, 2.03456320e+00])

def transform_pixels3(x, y, theta):
    undistorted_image_points = np.array([
        [-29.12822782, -16.49818496],
        [-28.44296656, 1102.53327195],
        [2077.59544377, -16.33455716],
        [2077.10087545, 1102.47021206]
    ], dtype=np.float32)

    # Known positions of the image corners in world coordinates
    c1_img = np.array([-0.03236, -0.32268])
    c2_img = np.array([-0.03558, -0.63524])
    c3_img = np.array([0.54582, -0.33314])
    c4_img = np.array([0.54280, -0.63581])

    # c1_img = np.array([-0.03348, -0.32238])
    # c2_img = np.array([-0.03713, -0.63308])
    # c3_img = np.array([0.54387, -0.33411])
    # c4_img = np.array([0.54013, -0.63428])

    world_points = np.array([[c1_img], [c2_img], [c3_img], [c4_img]])

    H, _ = cv2.findHomography(undistorted_image_points, world_points)
    undistorted_pixels = cv2.undistortPointsIter((x,y), Camera_matrix, distCoeffs, None, Camera_matrix, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))
    pixel = np.array([[[undistorted_pixels[0][0][0], undistorted_pixels[0][0][1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pixel, H)

    #theta = np.deg2rad(theta)
    
    r1 = R.from_rotvec(np.pi *np.array([1,0,0]))
    

    r2 = R.from_rotvec(theta * np.array([0,0,-1]))

    combined = r2 * r1

    rotvec = combined.as_rotvec()

    return mapped, rotvec

class FabricPositionSubscriber(Node):
    def __init__(self):
        super().__init__('fabric_position_subscriber')

        # Pose publisher for RViz/MoveIt UI
        self.pose_publisher = self.create_publisher(PoseStamped, '/goal_pose', 10) #/rviz/moveit/move_group/

        # Subscription to fabric position topic
        self.subscription = self.create_subscription(
            Float32MultiArray,
            'fabric_position',
            self.callback,
            10
        )

    def callback(self, msg):
        #print(len(msg.data))
        if len(msg.data) != 9:
            self.get_logger().warn("Invalid fabric position data received.")
            return

        
      

        x, y, theta, confidence, x2, y2, theta2, confidence2, img_id = msg.data
        self.get_logger().info(f"Received position: x={x}, y={y}, theta={theta}")
        
        # Transform pixel to world coordinates
        transformed, angle = transform_pixels3(x, y, theta)
        transformed2, angle2 = transform_pixels3(x2,y2, theta2)
        target_x, target_y = float(transformed[0][0][0]), float(transformed[0][0][1])
        target_x2, target_y2 = float(transformed2[0][0][0]), float(transformed2[0][0][1])
        distance = np.sqrt(((target_x - target_x2)**2 + (target_y - target_y2)**2))
        distance = np.abs(distance)
        distance = distance*1000
        
        self.get_logger().info("Approach")
        self.get_logger().info(f"target pose: X={target_x*1000}, Y={target_y*1000}, rotation={angle[:2]}")
        self.get_logger().info(f"with initial error:{distance}")

        self.get_logger().info("End Point")
        self.get_logger().info(f"target pose: X={target_x2*1000}, Y={target_y2*1000}, rotation={angle2[:2]}")
        self.get_logger().info(f"Image ID: = {img_id}")
        data = [[x,y,theta, x2, y2, theta2, target_x*1000,target_y*1000,angle, target_x2*1000, target_y2*1000, angle2, distance, img_id]]
        csvwriter.writerows(data)
        csvfile.flush()
        
        
            
csvwriter = None
current_file = ''

def main(args=None):
    global csvwriter, csvfile
    rclpy.init(args=args)
    node = FabricPositionSubscriber()

    my_path = "/home/melina/workspace/Test_logs/"
    num_files = len(os.listdir(my_path))
    file_name = 'Test_log' + str(num_files+1)
    current_file = file_name
    
    csvfile = open(my_path + current_file, 'w', newline='')
    csvwriter = csv.writer(csvfile)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        csvfile.close()

if __name__ == '__main__':
    main()
