import json
import time

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage


class TFProbe(Node):
    def __init__(self):
        super().__init__('tf_probe')
        self.count = 0
        self.first = None
        self.last = None
        self.changed = False
        self.create_subscription(TFMessage, '/tf', self._on_tf, 50)

    def _on_tf(self, msg: TFMessage):
        for t in msg.transforms:
            child = str(t.child_frame_id)
            if not child.endswith('base_link'):
                continue
            pos = (
                float(t.transform.translation.x),
                float(t.transform.translation.y),
                float(t.transform.translation.z),
            )
            self.count += 1
            if self.first is None:
                self.first = pos
            self.last = pos
            if self.first is not None:
                if any(abs(a - b) > 1e-4 for a, b in zip(self.first, pos)):
                    self.changed = True


def main():
    rclpy.init(args=None)
    node = TFProbe()
    end = time.time() + 14.0
    while rclpy.ok() and time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    result = {
        'tf_count': node.count,
        'first': node.first,
        'last': node.last,
        'changed': node.changed,
    }
    print(json.dumps(result))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
