from setuptools import setup

package_name = 'kalman_v1'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='carlos',
    maintainer_email='carlos.argueta@soulhackerslabs.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kalman_test = kalman_v1.kalman_test:main',
            'imu_test = kalman_v1.imu_test:main',
            'kalman_odom = kalman_v1.kalman_odom:main',
            'kalman_odom_imu = kalman_v1.kalman_odom_imu:main',
            'kalman_odom_imu_laser = kalman_v1.kalman_odom_imu_laser:main',
        ],
    },
)
