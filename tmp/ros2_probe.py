import argparse
import json
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class Probe(Node):
    def __init__(self):
        super().__init__('g1_probe')
        self.cmd_count = 0
        self.state_count = 0
        self.cmd_last = None
        self.state_first = None
        self.state_last = None
        self.state_changed = False
        self.create_subscription(JointState, '/g1/cmd/joint_commands', self._on_cmd, 50)
        self.create_subscription(JointState, '/g1/joint_states', self._on_state, 50)

    def _on_cmd(self, msg: JointState):
        self.cmd_count += 1
        self.cmd_last = list(msg.position)

    def _on_state(self, msg: JointState):
        self.state_count += 1
        pos = list(msg.position)
        if self.state_first is None:
            self.state_first = pos
        self.state_last = pos
        if self.state_first is not None and len(self.state_first) == len(pos):
            for a, b in zip(self.state_first, pos):
                if abs(float(a) - float(b)) > 1e-5:
                    self.state_changed = True
                    break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=8.0)
    parser.add_argument('--output', type=str, default='tmp/ros2_probe_result.json')
    args = parser.parse_args()

    rclpy.init(args=None)
    node = Probe()
    end = time.time() + args.duration
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    result = {
        'cmd_count': node.cmd_count,
        'state_count': node.state_count,
        'state_changed': bool(node.state_changed),
        'cmd_last_len': 0 if node.cmd_last is None else len(node.cmd_last),
        'state_first_len': 0 if node.state_first is None else len(node.state_first),
        'state_last_len': 0 if node.state_last is None else len(node.state_last),
    }
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f)
    print(json.dumps(result))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
