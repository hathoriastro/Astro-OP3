import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import math

class ImuYawListener(Node):
    def __init__(self):
        super().__init__('imu_yaw_listener')
        # Buat subscriber untuk topik IMU
        self.subscription = self.create_subscription(
            Imu,
            '/robotis_op3/imu',
            self.imu_callback,
            10)
        self.get_logger().info("IMU Yaw Listener started, subscribing to /robotis_op3/imu")

    def imu_callback(self, msg: Imu):
        # Ambil quaternion dari pesan IMU
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z
        w = msg.orientation.w

        # Konversi quaternion ke Euler (yaw, pitch, roll)
        # Rumus standar konversi quaternion ke yaw (Z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        # Konversi radian ke derajat
        yaw_deg = math.degrees(yaw)

        # Cetak nilai yaw di terminal
        print(f"Yaw (deg): {yaw_deg:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = ImuYawListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()