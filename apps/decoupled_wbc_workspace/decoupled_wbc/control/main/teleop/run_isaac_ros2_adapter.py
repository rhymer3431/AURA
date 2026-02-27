import rclpy
from rclpy.executors import MultiThreadedExecutor
import tyro

from decoupled_wbc.control.utils.isaac_ros_adapter import (
    InternalCommandToIsaacBridge,
    IsaacAdapterConfig,
    IsaacToInternalStateBridge,
)


def main(config: IsaacAdapterConfig):
    if config.run_mode not in {"both", "state", "command"}:
        raise ValueError("run_mode must be one of: both, state, command")

    rclpy.init(args=None)
    nodes = []
    try:
        if config.run_mode in {"both", "state"}:
            nodes.append(IsaacToInternalStateBridge(config))
        if config.run_mode in {"both", "command"}:
            nodes.append(InternalCommandToIsaacBridge(config))

        executor = MultiThreadedExecutor(num_threads=4)
        for node in nodes:
            executor.add_node(node)

        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        for node in nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    config = tyro.cli(IsaacAdapterConfig)
    main(config)
