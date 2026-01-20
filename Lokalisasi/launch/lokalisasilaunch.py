from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='Lokalisasi',        # Nama paket Anda
            executable='astro',     # Nama skrip atau executable yang ingin dijalankan
            name='lokalisasi_node',      # Nama node, bisa Anda tentukan sesuai keinginan
            output='screen'             # Menampilkan log di terminal
        )
    ])
