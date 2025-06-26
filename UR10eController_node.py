import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
import numpy as np
import cv2
import os
import csv


# Define the camera intrinsic parameters obtained from calibration
Camera_matrix = np.array([[2.94352929e+03, 0.00000000e+00, 9.98475579e+02],
                        [0.00000000e+00, 2.94681230e+03, 5.69576869e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Define the distortion coefficients obtained from calibration
distCoeffs = np.array([-2.20469671e-01, -1.84026226e-02, 1.48094161e-04, 6.83332051e-04, 2.03456320e+00])

# Function to check the quaternion representation of a rotation vector
def check_quaternion(ang1,ang2,ang3):
    np.array([ang1, ang2, ang3])
    rotation = R.from_rotvec(np.array([ang1, ang2, ang3]))
    quaternion = rotation.as_quat()
    return quaternion
# # Function to check all permutations of the angles and print the resulting quaternions and ZV vectors
def check_all(ang1, ang2, ang3):
    q1 = check_quaternion(ang1, ang2, ang3)
    q2 = check_quaternion(ang1, ang3, ang2)

    q3 = check_quaternion(ang2,ang1,ang3)
    q4 = check_quaternion(ang2,ang3,ang1)

    q5 = check_quaternion(ang3,ang1,ang2)
    q6 = check_quaternion(ang3,ang2,ang1)

    m1=R.from_quat(q1)
    m2=R.from_quat(q2)
    m3=R.from_quat(q3)
    m4=R.from_quat(q4)
    m5=R.from_quat(q5)
    m6=R.from_quat(q6)

    zv1= m1.apply([0, 0, 1])
    zv2= m2.apply([0, 0, 1])
    zv3= m3.apply([0, 0, 1])
    zv4= m4.apply([0, 0, 1])
    zv5= m5.apply([0, 0, 1])
    zv6= m6.apply([0, 0, 1])

    print("ZV1:", zv1)
    print("ZV2:", zv2)
    print("ZV3:", zv3)
    print("ZV4:", zv4)
    print("ZV5:", zv5)
    print("ZV6:", zv6)

    print("Quaternion 1:", q1)


# Transform pixel coordinates to world coordinates using undistortion and homography
def transform_pixels3(x, y, theta):
    # Convert pixel coordinates to undistorted points
    # (0,0), (0,1088), (2048,0), (2048,1088) undistorted image points through cv2.undistortPoints
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

    # Create a 2D array of the undistorted image points and world points
    world_points = np.array([[c1_img], [c2_img], [c3_img], [c4_img]])

    # Calculate the homography matrix from undistorted image points to world points
    H, _ = cv2.findHomography(undistorted_image_points, world_points)

    # Undistort the input pixel coordinates
    undistorted_pixels = cv2.undistortPointsIter((x,y), Camera_matrix, distCoeffs, None, Camera_matrix, (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, 0.03))
    pixel = np.array([[[undistorted_pixels[0][0][0], undistorted_pixels[0][0][1]]]], dtype=np.float32)

    # Map the undistorted pixel coordinates to world coordinates using the homography matrix
    mapped = cv2.perspectiveTransform(pixel, H)

    #theta = np.deg2rad(theta)
    
    # Create a rotation object from the rotation vector
    r1 = R.from_rotvec(np.pi *np.array([1,0,0]))
    
    # Create a rotation object from the rotation vector for the given angle
    r2 = R.from_rotvec(theta * np.array([0,0,-1]))

    # Combine the two rotations
    combined = r2 * r1

    rotvec = combined.as_rotvec()

    return mapped, rotvec

class FabricPositionSubscriber(Node):
    def __init__(self):
        super().__init__('fabric_position_subscriber')

        # Pose publisher for RViz/MoveIt UI
        self.pose_publisher = self.create_publisher(Float32MultiArray, '/pickup_place_pose/pos_command', 10) #/rviz/moveit/move_group/

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

        
      
        # Unpack the message data
        x, y, theta, confidence, x2, y2, theta2, confidence2, img_id = msg.data
        self.get_logger().info(f"Received position: x={x}, y={y}, theta={theta}")
        
        # Transform pixel to world coordinates
        transformed, angle = transform_pixels3(x, y, theta)
        transformed2, angle2 = transform_pixels3(x2,y2, theta2)
        
        # Calculate the quaternion representation of the rotation vectors
        quaternion1 = check_quaternion(angle[0], angle[1], 0)
        quaternion2 = check_quaternion(angle2[0], angle2[1], 0)

        # Prepare the data for CSV writing
        target_x, target_y = float(transformed[0][0][0]), float(transformed[0][0][1])
        target_x2, target_y2 = float(transformed2[0][0][0]), float(transformed2[0][0][1])
        distance = np.sqrt(((target_x - target_x2)**2 + (target_y - target_y2)**2))
        distance = np.abs(distance)
        distance = distance*1000
        
        # Log the received data
        self.get_logger().info("Approach")
        self.get_logger().info(f"target pose: X={target_x*1000}, Y={target_y*1000}, rotation={angle[:2]}")
        self.get_logger().info(f"with initial error:{distance}")

        self.get_logger().info("End Point")
        self.get_logger().info(f"target pose: X={target_x2*1000}, Y={target_y2*1000}, rotation={angle2[:2]}")
        self.get_logger().info(f"Image ID: = {img_id}")
        
        # Prepare the data for CSV writing and publishing
        data = [[x,y,theta, x2, y2, theta2, target_x*1000,target_y*1000,angle, target_x2*1000, target_y2*1000, angle2, distance, img_id]]
        move_msg =  Float32MultiArray()
        move_msg.data = [target_x, target_y, quaternion1[0], quaternion1[1], quaternion1[2], quaternion1[3], target_x2, target_y2, quaternion2[0], quaternion2[1], quaternion2[2], quaternion2[3]]
        csvwriter.writerows(data)
        csvfile.flush()
        self.pose_publisher.publish(move_msg)
        self.get_logger().info("Data sent")


csvwriter = None
current_file = ''

def main(args=None):
    check_all(2.086402,-2.348730, 0.000000)
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
