import rclpy
from sensor_msgs.msg import JointState
import json
from pathlib import Path

out = Path('tmp/ros2_joint_echo.json')
records = []

rclpy.init(args=None)
node = rclpy.create_node('joint_echo_capture')

def cb(msg: JointState):
    records.append({
        'name': list(msg.name[:8]),
        'position': [float(v) for v in list(msg.position[:8])],
        'count_names': len(msg.name),
        'count_positions': len(msg.position),
    })
    if len(records) >= 3:
        out.write_text(json.dumps(records, indent=2), encoding='utf-8')
        rclpy.shutdown()

node.create_subscription(JointState, '/g1/cmd/joint_commands', cb, 20)
start = node.get_clock().now()
while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.1)
out.write_text(json.dumps(records, indent=2), encoding='utf-8')
node.destroy_node()
