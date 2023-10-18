import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from std_msgs.msg import String

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

class ImuDataCollector(Node):
    def __init__(self):
        super().__init__('imu_data_collector')
        self.subscription = self.create_subscription(
            Imu,
            '/imu/data_raw',#'/zed/imu/data',  #'/imu',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        self.string_subscriber = self.create_subscription(String, '/message',
														  self.message_callback, 10)
        self.string_subscriber  # prevent unused variable warning

        self.linear_acc_x = []
        self.linear_acc_y = []
        self.angular_vel_z = []
        self.yaw = []

        self.started = False

    def listener_callback(self, msg):

        if self.started:
            self.linear_acc_x.append(msg.linear_acceleration.x)
            self.linear_acc_y.append(msg.linear_acceleration.y)
            self.angular_vel_z.append(msg.angular_velocity.z)

            # Convert quaternion to Euler angles
            _, _, yaw = self.euler_from_quaternion(msg.orientation)
            self.yaw.append(yaw)

    def message_callback(self, msg):
        self.message = msg.data

        if self.message == "o" or self.message == "f" or self.message == "b":

            self.linear_acc_x = []
            self.linear_acc_y = []
            self.angular_vel_z = []
            self.yaw = []
            
            self.started = True

        if self.message == "s":
            self.started = False
            self.plot_and_analyze()

        self.get_logger().info(self.message)

        self.message = ""

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
    

    def plot_and_analyze(self):
        time = range(len(self.linear_acc_x))

        plt.figure(figsize=(15, 10))
        plt.subplot(2, 2, 1)
        plt.plot(time, self.linear_acc_x)
        plt.title("Linear Acceleration X")
        plt.xlabel("Time")
        plt.ylabel("Acceleration")

        plt.subplot(2, 2, 2)
        plt.plot(time, self.linear_acc_y)
        plt.title("Linear Acceleration Y")
        plt.xlabel("Time")
        plt.ylabel("Acceleration")

        plt.subplot(2, 2, 3)
        plt.plot(time, self.angular_vel_z)
        plt.title("Angular Velocity Z")
        plt.xlabel("Time")
        plt.ylabel("Angular Velocity")

        plt.subplot(2, 2, 4)
        plt.plot(time, self.yaw)
        plt.title("Yaw")
        plt.xlabel("Time")
        plt.ylabel("Yaw")

        plt.tight_layout()
        plt.show()

        # Fit a normal distribution to the data
        mu, std = norm.fit(self.linear_acc_x)

        # Create a histogram
        plt.hist(self.linear_acc_x, bins=60, density=True, alpha=0.6, color='g')

        # Add title and labels
        plt.title('Histogram of X Acceleration Values')
        plt.xlabel('Acceleration (m/s²)')
        plt.ylabel('Frequency')

        # Plot the PDF (probability density function)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "X Acceleration Values - Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)


        # Show the plot
        plt.show()

        # Fit a normal distribution to the data
        mu, std = norm.fit(self.linear_acc_y)

        # Create a histogram
        plt.hist(self.linear_acc_y, bins=60, density=True, alpha=0.6, color='g')

        # Add title and labels
        plt.title('Histogram of Y Acceleration Values')
        plt.xlabel('Acceleration (m/s²)')
        plt.ylabel('Frequency')

        # Plot the PDF (probability density function)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Y Acceleration Values - Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        # Show the plot
        plt.show()

        # Fit a normal distribution to the data
        mu, std = norm.fit(self.angular_vel_z)

        # Create a histogram
        plt.hist(self.angular_vel_z, bins=60, density=True, alpha=0.6, color='g')

        # Add title and labels
        plt.title('Histogram of Angular Velocity Values')
        plt.xlabel('Yaw (rads/s)')
        plt.ylabel('Frequency')

        # Plot the PDF (probability density function)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Angular Velocity Values - Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        # Show the plot
        plt.show()

        # Fit a normal distribution to the data
        mu, std = norm.fit(self.yaw)

        # Create a histogram
        plt.hist(self.yaw, bins=60, density=True, alpha=0.6, color='g')

        # Add title and labels
        plt.title('Histogram of Yaw')
        plt.xlabel('Yaw (rads/s)')
        plt.ylabel('Frequency')

        # Plot the PDF (probability density function)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        title = "Yaw Values - Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title(title)

        # Show the plot
        plt.show()

        # Print stats
        datasets = [('Linear Acceleration X', self.linear_acc_x),
                    ('Linear Acceleration Y', self.linear_acc_y),
                    ('Angular Velocity Z', self.angular_vel_z),
                    ('Yaw', self.yaw)]
        for title, data in datasets:
            print(f"{title} stats:")
            print(f"Min: {np.min(data)}")
            print(f"Max: {np.max(data)}")
            print(f"Average: {np.mean(data)}")
            print()

def main(args=None):
    rclpy.init(args=args)

    imu_data_collector = ImuDataCollector()

    try:
        rclpy.spin(imu_data_collector)
    except KeyboardInterrupt:
        #imu_data_collector.plot_and_analyze()
        print("Bye")
    finally:
        imu_data_collector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
