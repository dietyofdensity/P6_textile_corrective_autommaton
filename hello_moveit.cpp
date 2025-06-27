#include <memory>

#include <rclcpp/rclcpp.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Quaternion.h>
// include all services in ur_msgs 
#include <ur_msgs/srv/set_io.hpp>

double current_pose[] = {-0.334, -0.526, 0.542, 1.0, 0.0, 0.0, 0.0};
double startp[]= {0.571, -0.270, 0.24, 1.0, 0.0, 0.0, 0.0};
double endp[]= {0.571, -0.270, 0.24, 1.0, 0.0, 0.0, 0.0};
double home_p[]={0.400, -0.200, 0.505, 1.0, 0.0, 0.0, 0.0};
double static_offset[]={0.102, -0.004, 0.185, 0.0, 0.0, 0.0, 1.0};
bool act_now = false;
bool get_act_now(){
  return act_now;
}

double aproach_height = 0.404;
double pick_up_height = 0.288;

geometry_msgs::msg::Pose posefromArray(double *arr)
{
  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"p_f_r pose [%f, %f, %f]", arr[0], arr[1], arr[2]);
    geometry_msgs::msg::Pose pose;
    pose.position.x = arr[0];
    pose.position.y = arr[1];
    pose.position.z = arr[2];
    pose.orientation.x = arr[3];
    pose.orientation.y = arr[4];
    pose.orientation.z = arr[5];
    pose.orientation.w = arr[6];
    return pose;
} 
geometry_msgs::msg::Pose actual_start_pose()
{
  geometry_msgs::msg::Pose p1 = posefromArray(current_pose);
  tf2::Transform transform_ee;
  // make p1 into a transform
  return p1;
}
geometry_msgs::msg::Pose movIt_ee_pos(geometry_msgs::msg::Pose pose)
{

  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"m_ee_p pre transform pose [%f, %f, %f]", pose.position.x, pose.position.y, pose.position.z);
   RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"m_ee_p pre transform quaternion [%f, %f, %f, %f]", pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w);
  geometry_msgs::msg::Pose p1 = pose;
  // create a tf2 transform
  tf2::Transform transform_ee;

  //tf2::frame::Transform transform_ee;
  // make p1 into a transform
  transform_ee.setOrigin(tf2::Vector3(p1.position.x, p1.position.y, p1.position.z));
  transform_ee.setRotation(tf2::Quaternion(p1.orientation.x, p1.orientation.y, p1.orientation.z, p1.orientation.w));

  geometry_msgs::msg::Pose p2 = posefromArray(static_offset);
  tf2::Transform transform_static;
  transform_static.setOrigin(tf2::Vector3(-p2.position.x, -p2.position.y, -p2.position.z));
  transform_static.setRotation(tf2::Quaternion(p2.orientation.x, p2.orientation.y, p2.orientation.z, p2.orientation.w));
  // invert the static transform
  //transform_static = transform_static.inverse();
  // apply the static transform to the end effector pose
  
  transform_ee = transform_ee * transform_static;
  // convert the transform back to a pose msg

  geometry_msgs::msg::Pose pose_ee;
  tf2::toMsg(transform_ee, pose_ee);
  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"m_ee_p post transform pose [%f, %f, %f]", pose_ee.position.x, pose_ee.position.y, pose_ee.position.z);
  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"m_ee_p post transform quaternion [%f, %f, %f, %f]", pose_ee.orientation.x, pose_ee.orientation.y, pose_ee.orientation.z, pose_ee.orientation.w);

  return pose_ee;
}

void poseCallback(const geometry_msgs::msg::PoseStamped msg)
{
   //RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"Received current pose");
  
    geometry_msgs::msg::Point pos = msg.pose.position;
    geometry_msgs::msg::Quaternion quat = msg.pose.orientation;

    current_pose[0] = pos.x;
    current_pose[1] = pos.y;
    current_pose[2] = pos.z;

    current_pose[3] = quat.x;
    current_pose[4] = quat.y;
    current_pose[5] = quat.z;
    current_pose[6] = quat.w;    
     
}
void quaternion_config_test(double ang1,double ang2,double ang3){
   

}
void commandCallback(const std_msgs::msg::Float32MultiArray msg)
{
  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"Received command pose" );
  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"given command pre p1 [%f, %f, %f,%f]", msg.data[0], msg.data[1],msg.data[2],msg.data[3]);
  RCLCPP_INFO(rclcpp::get_logger("hello_moveit"),"given command pre p2 [%f, %f, %f,%f]", msg.data[4], msg.data[5],msg.data[6],msg.data[7]);
    // check if the message has 8 elements

    startp[0] = msg.data[0];
    startp[1] = msg.data[1];
    startp[2] = 0;
 


    startp[3] = msg.data[2];
    startp[4] = msg.data[3];
    startp[5] = msg.data[4];
    startp[6] = msg.data[5];


    endp[0] = msg.data[6];
    endp[1] = msg.data[7];
    endp[2] = 0;
  
    endp[3] = msg.data[8];
    endp[4] = msg.data[9];
    endp[5] = msg.data[10];
    endp[6] = msg.data[11];
     
    act_now = true;

}

void cart_move(moveit::planning_interface::MoveGroupInterface& move_group_interface, geometry_msgs::msg::Pose pose, double height)
{
  move_group_interface.setPoseReferenceFrame("base");
  

    std::vector<geometry_msgs::msg::Pose> waypoints;
      // If you want to use a cartesian path, you can set the start pose
    //move_group_interface.setStartStateToCurrentState();
    //waypoints.push_back(move_group_interface.getCurrentPose().pose);
    //waypoints.push_back(actual_start_pose());  // up

    pose.position.z = height;

    waypoints.push_back(pose);
 
    // If you want to use a cartesian path, you can set the start pose
    
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;
    double fraction = move_group_interface.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    //RCLCPP_INFO(logger, "Visualizing plan 4 (Cartesian path) (%.2f%% achieved)", fraction * 100.0);
    
    auto const [success, plan] = [&move_group_interface]{
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);}();

  if(success) {
  // This call blocks until the trajectory is finished
    auto result = move_group_interface.execute(trajectory);
    if (result == moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_INFO(rclcpp::get_logger("hello_moveit"), "Trajectory execution succeeded.");
    } else {
      RCLCPP_WARN(rclcpp::get_logger("hello_moveit"), "Trajectory execution failed.");
    }
  } else {
  RCLCPP_ERROR(rclcpp::get_logger("hello_moveit"), "Planning failed!");
}
  
}


void old_cart_move(moveit::planning_interface::MoveGroupInterface& move_group_interface, geometry_msgs::msg::Pose pose, double height)
{
  move_group_interface.setPoseReferenceFrame("base");
  auto const home_pose = []{
    geometry_msgs::msg::Pose msg;
    msg.orientation.x = 1.0;
    msg.orientation.y = 0.0;
    msg.orientation.z = 0.0;
    msg.orientation.w = 0.00082;
    msg.position.x = -0.404;
    msg.position.y = -0.400;
    msg.position.z = 0.504;
    return msg;
  }();
  auto const pick_up_pose = []{
    geometry_msgs::msg::Pose msg;
    msg.orientation.x = 1.0;
    msg.orientation.y = 0.0;
    msg.orientation.z = 0.0;
    msg.orientation.w = 0.00082;
    msg.position.x = -0.374;
    msg.position.y = -0.400;
    msg.position.z = 0.504;
    return msg;
  }();
  auto const place_down_pose = []{
    geometry_msgs::msg::Pose msg;
    msg.orientation.x = 1.0;
    msg.orientation.y = 0.0;
    msg.orientation.z = 0.0;
    msg.orientation.w = 0.00082;
    msg.position.x = -0.300;
    msg.position.y = -0.400;
    msg.position.z = 0.504;
    return msg;
  }();

    std::vector<geometry_msgs::msg::Pose> waypoints;
      // If you want to use a cartesian path, you can set the start pose
    //move_group_interface.setStartStateToCurrentState();
    //waypoints.push_back(move_group_interface.getCurrentPose().pose);
    //waypoints.push_back(actual_start_pose());  // up

    waypoints.push_back(home_pose);

    waypoints.push_back(pick_up_pose);  // down

    waypoints.push_back(place_down_pose);  // right

 


    // If you want to use a cartesian path, you can set the start pose
    

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.01;
    double fraction = move_group_interface.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    //RCLCPP_INFO(logger, "Visualizing plan 4 (Cartesian path) (%.2f%% achieved)", fraction * 100.0);
    auto const [success, plan] = [&move_group_interface]{
    moveit::planning_interface::MoveGroupInterface::Plan msg;
    auto const ok = static_cast<bool>(move_group_interface.plan(msg));
    return std::make_pair(ok, msg);
  }();

    if(success) {
     move_group_interface.execute(trajectory);
  } else {
    //RCLCPP_ERROR(logger, "Planing failed!");
  }
}


void open_gripper(moveit::planning_interface::MoveGroupInterface &move_group_interface)
{ 

}
void close_gripper(moveit::planning_interface::MoveGroupInterface &move_group_interface)
{ 
}
void pick_up_nav(moveit::planning_interface::MoveGroupInterface& move_group)
{
  // set the refrence frame of the pose target to the base_link
  move_group.setPoseReferenceFrame("base");
  
  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  geometry_msgs::msg::Quaternion msg_quat= tf2::toMsg(q);
  geometry_msgs::msg::Pose pick_pose;
  pick_pose.orientation = msg_quat;
  pick_pose.position.x = -0.374;
  pick_pose.position.y = -0.400;
  pick_pose.position.z = 0.274;

  move_group.setPoseTarget(pick_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto const outcome = static_cast<bool>(move_group.plan(plan));
  if(outcome) {
   // move_group.execute(plan);
  }
     

}
void move_to_home(moveit::planning_interface::MoveGroupInterface& move_group){

moveit::planning_interface::MoveGroupInterface::Plan my_plan;
bool success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
my_plan.trajectory_.joint_trajectory.header.frame_id = "base";

std::vector<double> joint_group_positions(6);
double degtorad= 3.14159265358979323846 / 180.0;

joint_group_positions[0] = -0*degtorad;  // radians
joint_group_positions[1] = -55.04*degtorad;
joint_group_positions[2] = -141.63*degtorad;
joint_group_positions[3] = -73.32*degtorad;
joint_group_positions[4] = 90*degtorad;
joint_group_positions[5] = -0*degtorad;


move_group.setJointValueTarget(joint_group_positions);
move_group.setMaxVelocityScalingFactor(1);
move_group.setMaxAccelerationScalingFactor(1); 




success = (move_group.plan(my_plan) == moveit::core::MoveItErrorCode::SUCCESS);
RCLCPP_ERROR(rclcpp::get_logger("hello_moveit"), "Visualizing plan 2 (joint space goal) %s", success ? "" : "FAILED");
if(success){
   move_group.execute(my_plan);
} 

}
void place_down_nav(moveit::planning_interface::MoveGroupInterface& move_group)
{
   


}
void move_to_pose_at_height(moveit::planning_interface::MoveGroupInterface& move_group, geometry_msgs::msg::Pose pose, double height)
{
  // set the refrence frame of the pose target to the base_link
  move_group.setPoseReferenceFrame("base");
  
  tf2::Quaternion q;
  q.setRPY(0, 0, 0);
  geometry_msgs::msg::Quaternion msg_quat= tf2::toMsg(q);
  geometry_msgs::msg::Pose target_pose = pose;
  target_pose.orientation = msg_quat;
  target_pose.position.z = height;

  move_group.setPoseTarget(target_pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  auto const outcome = static_cast<bool>(move_group.plan(plan));
  if(outcome) {
   // move_group.execute(plan);
  }
}


int main(int argc, char * argv[])
{
  // Initialize ROS and create the Node
  rclcpp::init(argc, argv);
  auto const node = std::make_shared<rclcpp::Node>(
    "hello_moveit",
    rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)
  );
  auto subscription_pose = node->create_subscription<geometry_msgs::msg::PoseStamped>(
        "/tcp_pose_broadcaster/pose", 10, poseCallback);
  auto subscription = node->create_subscription<std_msgs::msg::Float32MultiArray>(
        "/pickup_place_pose/pos_command", 10, commandCallback);
  // Create a ROS logger
  auto const logger = rclcpp::get_logger("hello_moveit");
  RCLCPP_ERROR(logger, "starting wait");

  rclcpp::sleep_for(std::chrono::seconds(2));
  RCLCPP_ERROR(logger, "wait complete");
  //rclcpp::sleep_for(time);
  // Next step goes here
  // Create the MoveIt MoveGroup Interface
  auto client = node->create_client<ur_msgs::srv::SetIO>("io_and_status_controller/set_io");
  
  // Wait for the service to be available
  if (!client->wait_for_service(std::chrono::seconds(2))) {
    RCLCPP_ERROR(node->get_logger(), "SetIO service not available!");
  }
  act_now=false;

  quaternion_config_test(2.086402,-2.348730, 0.0);
  quaternion_config_test(2.207189,-2.235603, 0.0);

  moveit::planning_interface::MoveGroupInterface m_g_i(node, "ur_manipulator");
  m_g_i.setPoseReferenceFrame("base");
  // make a while loop which runs until node shutdown

  geometry_msgs::msg::Pose home_pose = movIt_ee_pos(posefromArray(home_p));
  //cart_move(m_g_i,home_pose,0.504);
  move_to_home(m_g_i);


  while (rclcpp::ok()) {
    // wait for a command to be received using the commandCallback
    // if act_now is true, execute the cartesian move
    // wait for 300 ms
    rclcpp::spin_some(node); 

    rclcpp::sleep_for(std::chrono::milliseconds(1000));
    //
    RCLCPP_INFO(logger,"waited 1000 ms %s", get_act_now() ? "true" : "false");
    if(get_act_now()) {
     
      RCLCPP_INFO(logger, "Received new pose command, executing cartesian move");
      // establish pose targest and reqests
       geometry_msgs::msg::Pose start_pose = movIt_ee_pos(posefromArray(startp));
       RCLCPP_INFO(logger, "start pose was set to: %f, %f, %f", start_pose.position.x, start_pose.position.y, start_pose.position.z);
      geometry_msgs::msg::Pose end_pose = movIt_ee_pos(posefromArray(endp));
      RCLCPP_INFO(logger, "start pose was set to: %f, %f, %f", end_pose.position.x, end_pose.position.y, end_pose.position.z);
      geometry_msgs::msg::Pose home_pose = movIt_ee_pos(posefromArray(home_p));
      RCLCPP_INFO(logger, "start pose was set to: %f, %f, %f", home_pose.position.x, home_pose.position.y, home_pose.position.z);
      auto request_on = std::make_shared<ur_msgs::srv::SetIO::Request>();
      request_on->fun = 1;     // e.g., 1 for digital_out
      request_on->pin = 4;     // pin number
      request_on->state = 1; // 1.0 for ON, 0.0 for OFF

      auto request_off = std::make_shared<ur_msgs::srv::SetIO::Request>();
      request_off->fun = 1;     // e.g., 1 for digital_out
      request_off->pin = 4;     // pin number
      request_off->state = 0; // 1.0 for ON, 0.0 for OFF
      int slep_del1=1100;
      int slep_del2=200;

      // move from home to pickup pose
      cart_move(m_g_i, start_pose, aproach_height); // move to approach height
      rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      // move down to pick up
      cart_move(m_g_i, start_pose, pick_up_height); // move down to pick up height
       rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      // actuate gripper to pick up
      auto result_future_1 = client->async_send_request(request_on);
      if (rclcpp::spin_until_future_complete(node, result_future_1) == rclcpp::FutureReturnCode::SUCCESS) {
        auto response = result_future_1.get();
        if (response->success) {
          RCLCPP_INFO(node->get_logger(), "SetIO succeeded");
        } else {
          RCLCPP_WARN(node->get_logger(), "SetIO failed");
        }
      } else {
        RCLCPP_ERROR(node->get_logger(), "Service call failed");
      }
 

       rclcpp::sleep_for(std::chrono::milliseconds(slep_del1));
      // move up to pickup height
      cart_move(m_g_i, start_pose, aproach_height);
       rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      // move to place down pose
      cart_move(m_g_i, end_pose, aproach_height); // move to approach height
       rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      // move down to place down height
      cart_move(m_g_i, end_pose, pick_up_height); // move to approach height
       rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      // actuate gripper to place down
      auto result_future_2 = client->async_send_request(request_off);
      if (rclcpp::spin_until_future_complete(node, result_future_2) == rclcpp::FutureReturnCode::SUCCESS) {
        auto response = result_future_2.get();
        if (response->success) {
          RCLCPP_INFO(node->get_logger(), "SetIO succeeded");
        } else {
          RCLCPP_WARN(node->get_logger(), "SetIO failed");
        }
      } else {
        RCLCPP_ERROR(node->get_logger(), "Service call failed");
      }
       rclcpp::sleep_for(std::chrono::milliseconds(400));
      // move up to place down height
      cart_move(m_g_i, end_pose, aproach_height);
       rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      // move back to home pose
      move_to_home(m_g_i);
      //cart_move(m_g_i, home_pose, home_p[2]); // move to approach height
       rclcpp::sleep_for(std::chrono::milliseconds(slep_del2));
      act_now = false;
    }

  }
  
  // Shutdown ROS
  rclcpp::shutdown();
  return 0;
}