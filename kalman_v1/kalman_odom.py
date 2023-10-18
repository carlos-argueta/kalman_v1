# https://github.com/AkshayLaddha943/Arrow_SensorFusion_turtlebot3_ws

import rclpy
from rclpy.node import Node

# Messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Imu

import numpy as np

from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.common import Saver

import matplotlib.pyplot as plt

'''
Tha Kalman Filter Algorithm

Initialization

1. Initialize the state of the filter
2. Initialize our belief in the state

Predict

1. Use process model to predict state at the next time step
2. Adjust belief to account for the uncertainty in prediction    

Update

1. Get a measurement and associated belief about its accuracy
2. Compute residual between estimated state and measurement
3. Compute scaling factor based on whether the measurement
or prediction is more accurate
4. set state between the prediction and measurement based 
on scaling factor
5. update belief in the state based on how certain we are 
in the measurement
'''

class KalmanV1(Node):

    def __init__(self):
        
        super().__init__('kalman_v1')
        
        # Subscribe to the /odom topic
        self.odom_subscriber = self.create_subscription(
		 Odometry,
		 '/wheel_odom', #'/fake_odom',  #'/odom', '/wheel_odom'
		 self.odom_callback,
		 10)
        
        
        # Subscribe to the /cmd_vel topic
        self.cmd_vel_subscriber = self.create_subscription(
            Twist,
            '/cmd_vel_mux', #'/cmd_vel',
            self.cmd_vel_callback,
            10)
        
    
        self.string_subscriber = self.create_subscription(String, '/message',
														  self.message_callback, 10)
        
        # Variables for fake odometry
        self.fake_odom = Odometry()
        self.initial_odom = Odometry()
        self.fake_odom_set = False
        

        # Init the state vector X, with the initial position, heading, and velocities
        self.X = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) # x, y, theta, v_x, v_y, omega
       
        print(self.X, self.X.shape)

        
        self.P = np.eye(6) * 1000  # Initialize with high uncertainty
        

        # Init dt
        self.dt = 0.1  # Time step

        # Init the process model matrix F
        self.F = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0],   # x = x + v_x * dt  (dt = 1)    
                            [0.0, 1.0, 0.0, 0.0, self.dt, 0.0],   # y = y + v_y * dt  (dt = 1)
                            [0.0, 0.0, 1.0, 0.0, 0.0, self.dt],   # theta = theta + omega * dt  (dt = 1)
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # v_x = v_x
                            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # v_y = v_y
                            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # omega = omega
        
        
        # Variance of the process noise for each state variable
        q_x = 2 #0.1
        q_y = 2 #0.1
        q_theta = 1.57 #0.01
        q_vx = 0.25 #0.1
        q_vy = 0.25 #0.1
        q_omega = 0.3 # 0.01
     

        # Create the process noise covariance matrix
        self.Q = np.diag([q_x, q_y, q_theta, q_vx, q_vy, q_omega])
       
        # Init the control input matrix B
        self.B = np.array([
            [0, 0],
            [0, 0],
            [0, 0],
            [1, 0],
            [0, 0],
            [0, 1]
        ])

        # Define control vector u
        self.u = np.array([0.0, 0.0]) # linear velocity v, angular velocity w

      
        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # x = x
                            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # y = y
                            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])  # theta = theta
        
           
        # Init the measurement vector z
        self.z = np.array([[0.0], [0.0], [0.0]]) # x, y, theta
        
        # Init the measurement covariance matrix R
        self.R = np.diag([0.25, 0.25, 0.35]) # TODO: Tune this matrix, perhaps with the covariance from the odom message


        # Initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=6, dim_z=3)
        self.kf.x = self.X
        self.kf.P = self.P
        self.kf.F = self.F
        self.kf.Q = self.Q
        self.kf.B = self.B
        self.kf.H = self.H
        self.kf.R = self.R

        print(self.kf.x, self.kf.x.shape)

        # Init the saver to save the state of the filter
        self.s = Saver(self.kf)

        self.odom_count = 0

        self.message = ""

        self.last_time = None

        self.odom_msg = None
        

        

    def cmd_vel_callback(self, msg):
        self.get_logger().info('Cmd_vel: "%s" "%s"' % (msg.linear.x, msg.angular.z))
            
        # Update the control input vector
        self.u = np.array([[msg.linear.x], [msg.angular.z]])

        # Predict the state vector and the covariance
        #self.kf.predict(u=self.u)
        #self.kf.predict()


           

    def odom_callback(self, msg):

        self.odom_msg = msg

		# Calculate the fake odometry
        if not self.fake_odom_set:
            self.initial_odom = self.odom_msg
            self.fake_odom_set = True
        else:

            self.update_fake_odom()

		#self.get_logger().info("Odometry: %s"%self.odom)

		
        # Get the current time from the message header
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # convert to seconds

        # Calculate dt
        if self.last_time is not None:
            self.dt = current_time - self.last_time

        # Update the last time
        self.last_time = current_time

        # Use dt in the filter update
        self.kf.dt = self.dt
        self.kf.predict()

        self.odom_count += 1

        # Convert the quaternion to Euler angles
        _, _, self.theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        # New measurement received, update the measurement vector
        self.z = np.array([[self.fake_odom.pose.pose.position.x], [self.fake_odom.pose.pose.position.y], [self.theta]])

        # Update the state vector and covariance
        self.kf.update(self.z)

        # Save the state of the filter
        self.s.save()

        if self.odom_count % 20 == 0:
            self.get_logger().info('Odometry: "%s" "%s" "%s"' % (self.fake_odom.pose.pose.position.x, self.fake_odom.pose.pose.position.y, self.theta))
            self.get_logger().info('state:"%s" "%s" "%s" "%s"' % (self.kf.x.shape, self.kf.x[0], self.kf.x[1], self.kf.x[2]))

    def message_callback(self, msg):
        self.message = msg.data

        if self.message == "s":
            

            self.plot()

            self.scatter()

        self.get_logger().info(self.message)
                

    def euler_from_quaternion(self, quaternion):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quaternion = [x, y, z, w]
        """
        x = quaternion.x
        y = quaternion.y
        z = quaternion.z
        w = quaternion.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw
    
    def update_fake_odom(self):
		# Calculate the difference between the current and previous odometry readings
        delta_position_x = self.odom_msg.pose.pose.position.x - self.initial_odom.pose.pose.position.x
        delta_position_y = self.odom_msg.pose.pose.position.y - self.initial_odom.pose.pose.position.y
        delta_position_z = self.odom_msg.pose.pose.position.z - self.initial_odom.pose.pose.position.z

		# Create a new odometry message with the difference between the two odometry readings
        self.fake_odom = Odometry()
        self.fake_odom.header.stamp = self.get_clock().now().to_msg()
        self.fake_odom.child_frame_id = self.odom_msg.child_frame_id
        self.fake_odom.pose.pose.position.x = delta_position_x
        self.fake_odom.pose.pose.position.y = delta_position_y
        self.fake_odom.pose.pose.position.z = delta_position_z
        self.fake_odom.pose.pose.orientation = self.odom_msg.pose.pose.orientation


        #self.fake_odom_publisher.publish(self.fake_odom)
                
    def scatter(self):
        # Extract the saved states and measurements
        states = self.s.x
        measurements = self.s.z

        # Extract the estimated and measured x and y positions
        x_estimated = [state[0] for state in states]
        y_estimated = [state[1] for state in states]

        x_measured = [measurement[0] for measurement in measurements]
        y_measured = [measurement[1] for measurement in measurements]

        # Plot estimated and measured paths
        plt.figure()
        plt.scatter(x_estimated, y_estimated, label='Estimated path', s=5)
        plt.scatter(x_measured, y_measured, label='Measured path', s=5)
        plt.title('Estimated and Measured Paths')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.gca().set_aspect('equal', adjustable='box')  # maintain an equal aspect ratio
        plt.legend()
        plt.show()



    def plot(self):
        # Extract the saved states and measurements
        states = self.s.x
        measurements = self.s.z

        # Extract the estimated and measured x positions
        x_estimated = [state[0] for state in states]
        x_measured = [measurement[0] for measurement in measurements]

        # Create a time vector
        time = range(len(x_estimated))

        # Plot estimated and measured x position over time
        plt.figure()
        plt.plot(time, x_estimated, label='Estimated x position')
        plt.plot(time, x_measured, label='Measured x position')
        plt.title('Estimated and Measured X Position Over Time')
        plt.xlabel('Time')
        plt.ylabel('X Position')
        plt.legend()
        plt.show()




def main(args=None):
    rclpy.init(args=args)

    kf_node = KalmanV1()

    rclpy.spin(kf_node)

    kf_node.destroy_node()
    rclpy.shutdown()
    print("Node destroyed and shutdown")