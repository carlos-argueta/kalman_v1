# https://github.com/AkshayLaddha943/Arrow_SensorFusion_turtlebot3_ws

import rclpy
from rclpy.node import Node

# Messages
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String
from sensor_msgs.msg import Imu

import numpy as np
from math import degrees

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
        
        self.imu_subscriber = self.create_subscription(
            Imu,
            '/imu/data_raw', #'/imu',
            self.imu_callback,
            10)
        
        self.zed_imu_subscriber = self.create_subscription(
            Imu,
            '/zed/imu/data', #'/imu',
            self.zed_imu_callback,
            10)
        
        self.string_subscriber = self.create_subscription(String, '/message',
														  self.message_callback, 10)
    
        # Variables for fake odometry
        self.fake_odom = Odometry()
        self.initial_odom = Odometry()
        self.fake_odom_set = False

        # Variables for odometry
        self.odom_count = 0
        self.odom_msg = None

        # Variables for imu
        self.imu_msg = None
        self.initial_imu_msg = None
        self.initial_theta = None
        self.fake_theta_set = False
        self.fake_theta = 0.0
        self.fake_imu_theta = 0.0
        
        # Init the state vector X, with the initial position, heading, and velocities
        self.X = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) # x, y, theta, v_x, v_y, omega, a_x, a_y

        print(self.X, self.X.shape)

        self.P = np.eye(8) * 1000  # Initialize with high uncertainty
       
        # Init dt
        self.dt = 0.1  # Time step

        # Init the process model matrix F
        self.F = np.array([[1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 0.5*self.dt**2, 0.0],   # x = x + v_x * dt  + 0.5 * a_x * dt^2
                   [0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0, 0.5*self.dt**2],   # y = y + v_y * dt  + 0.5 * a_y * dt^2
                   [0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0, 0.0],   # theta = theta + omega * dt 
                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt, 0.0],   # v_x = v_x + a_x * dt
                   [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, self.dt],  # v_y = v_y + a_y * dt
                   [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # omega = omega
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # a_x = a_x
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])  # a_y = a_y

        
        # Variance of the process noise for each state variable
        q_x = 2 #0.1
        q_y = 2 #0.1
        q_theta = 1.57 #0.01
        q_vx = 0.25 #0.1
        q_vy = 0.25 #0.1
        q_omega = 0.3 # 0.01
        q_ax = 4.0 
        q_ay = 4.0

        # Create the process noise covariance matrix
        self.Q = np.diag([q_x, q_y, q_theta, q_vx, q_vy, q_omega, q_ax, q_ay])
   
        # Init the control input matrix B
        self.B = np.array([[0, 0],[0, 0],[0, 0],[1, 0],[0, 0],[0, 1]
        ])

        # Define control vector u
        self.u = np.array([0.0, 0.0]) # linear velocity v, angular velocity w

        self.H = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # x = x
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # y = y
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # theta = theta
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],   # omega = omega
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # a_x = a_x
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # a_y = a_y
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])  # theta = theta_imu
 
           
        # Init the measurement vector z
        self.z = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]) # x, y, theta, omega, a_x, a_y, theta_imu

        # Init the measurement covariance matrix R
        self.R = np.diag([0.25, 0.25, 0.35, 0.1, 0.0015387262937311438, 0.00153872629373114380, 0.1]) # TODO: Tune this matrix, perhaps with the covariance from the odom message

        # Initialize the Kalman filter
        self.kf = KalmanFilter(dim_x=8, dim_z=7)
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

        self.message = ""

        self.last_time = None

    

    def cmd_vel_callback(self, msg):
        self.get_logger().info('Cmd_vel: "%s" "%s"' % (msg.linear.x, msg.angular.z))
            
        # Update the control input vector
        self.u = np.array([[msg.linear.x], [msg.angular.z]])

        # Predict the state vector and the covariance
        #self.kf.predict(u=self.u)
        #self.kf.predict()

    def zed_imu_callback(self, msg):

        # Compute the yaw from the IMU quaternion
        _, _, self.zed_imu_theta = self.euler_from_quaternion(msg.orientation)

    def imu_callback(self, msg):

        self.imu_msg = msg

        # Wait until we get at least one odom message
        if not self.fake_odom_set:
            return
        
        # Extract the linear acceleration data from the IMU message
        a_x = msg.linear_acceleration.x
        a_y = msg.linear_acceleration.y

        # Extract the angular velocity data from the IMU message
        omega = msg.angular_velocity.z

        # Compute the yaw from the IMU quaternion
        _, _, self.imu_theta = self.euler_from_quaternion(msg.orientation)

        # Calculate the fake theta
        if not self.fake_theta_set:
            self.initial_imu_msg = self.imu_msg
            self.fake_theta_set = True

            _, _, self.initial_theta = self.euler_from_quaternion(self.initial_imu_msg.orientation)

        else:

            
            # Calculate the difference in yaw
            delta_theta = self.imu_theta - self.initial_theta

            # Unwrap the delta_yaw to avoid issues with angles wrapping around at 2*pi radians
            self.fake_imu_theta = self.normalize_angle(delta_theta)
            
        # Include them in the measurement vector `z`
        self.z = np.array([[self.fake_odom.pose.pose.position.x], [self.fake_odom.pose.pose.position.y], [self.fake_theta], [omega], [a_x], [a_y], [self.fake_imu_theta]])

        # You may also want to update the measurement noise covariance matrix R with the variance of the acceleration measurements
        # This is an optional step and will depend on the specifics of your application
        # For example, you might do something like this:
        #self.R[3][3] = np.var(a_x)
        #self.R[4][4] = np.var(a_y)    

        # Update the state vector and covariance
        self.kf.update(self.z)

        # Save the state of the filter
        self.s.save() 

        
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

        # Wait until we have received at least one IMU message
        if self.imu_msg is None:
            return 
        
        a_x = self.imu_msg.linear_acceleration.x
        a_y = self.imu_msg.linear_acceleration.y
        omega = self.imu_msg.angular_velocity.z
        
        # New measurement received, update the measurement vector
        self.z = np.array([[self.fake_odom.pose.pose.position.x], [self.fake_odom.pose.pose.position.y], [self.fake_theta], [omega], [a_x], [a_y], [self.fake_imu_theta]])

        
        # Update the state vector and covariance
        self.kf.update(self.z)

        # Save the state of the filter
        self.s.save()

        if self.odom_count % 20 == 0:
            self.get_logger().info('Odometry: "%s" "%s" "%s"' % (self.fake_odom.pose.pose.position.x, self.fake_odom.pose.pose.position.y, self.theta))
            self.get_logger().info('state:"%s" "%s" "%s" "%s"' % (self.kf.x.shape, self.kf.x[0], self.kf.x[1], self.kf.x[2]))
            self.get_logger().info("IMU theta: %s - %s"%(self.imu_theta, degrees(self.imu_theta)))
            self.get_logger().info("Fake IMU theta: %s - %s"%(self.fake_imu_theta, degrees(self.fake_imu_theta)))
            self.get_logger().info("Fake ODOM theta: %s - %s"%(self.fake_theta, degrees(self.fake_theta)))
            #self.get_logger().info("Zed IMU theta: %s - %s"%(self.zed_imu_theta, degrees(self.zed_imu_theta))) 

    def message_callback(self, msg):
        self.message = msg.data

        if self.message == "s":
            

            self.plot()

            self.scatter()

        self.get_logger().info(self.message)
                
    def normalize_angle(self, angle):
        """
        Convenience function to map an angle to the range [-pi,pi]
        """
        #return np.arctan2(np.sin(angle), np.cos(angle)
                
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

                          
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


        # Convert the current and initial orientations to Euler angles
        _, _, current_yaw = self.euler_from_quaternion(self.odom_msg.pose.pose.orientation)

        _, _, initial_yaw = self.euler_from_quaternion(self.initial_odom.pose.pose.orientation)

        # Calculate the difference in yaw
        delta_yaw = current_yaw - initial_yaw

        # Unwrap the delta_yaw to avoid issues with angles wrapping around at 2*pi radians
        self.fake_theta = self.normalize_angle(delta_yaw)

        # Convert the relative yaw back to a quaternion
        #x, y, z, w = quaternion_from_euler(0, 0, delta_yaw)

        #self.fake_odom.pose.pose.orientation.x = x
        #self.fake_odom.pose.pose.orientation.y = y
        #self.fake_odom.pose.pose.orientation.z = z
        #self.fake_odom.pose.pose.orientation.w = w

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
        plt.axis('equal')  # maintain an equal aspect ratio
        #plt.gca().set_aspect('equal', adjustable='box')  # maintain an equal aspect ratio
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