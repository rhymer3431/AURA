from __future__ import annotations

import logging


def setup_joint_and_tf_graph(namespace: str, robot_prim_path: str) -> None:
    try:
        import omni.graph.core as og  # type: ignore
        import usdrt.Sdf  # type: ignore
    except Exception as exc:
        logging.warning("Could not import Isaac omnigraph modules for joint/tf graph setup: %s", exc)
        return

    graph_path = "/G1ROS2Bridge"
    cmd_topic = f"/{namespace}/cmd/joint_commands"
    joint_state_topic = f"/{namespace}/joint_states"

    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
                ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
                ("ArticulationController", "isaacsim.core.nodes.IsaacArticulationController"),
                ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
                ("Context.outputs:context", "PublishClock.inputs:context"),
                ("Context.outputs:context", "PublishJointState.inputs:context"),
                ("Context.outputs:context", "SubscribeJointState.inputs:context"),
                ("Context.outputs:context", "PublishTF.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
                ("ReadSimTime.outputs:simulationTime", "PublishTF.inputs:timeStamp"),
                ("SubscribeJointState.outputs:jointNames", "ArticulationController.inputs:jointNames"),
                ("SubscribeJointState.outputs:positionCommand", "ArticulationController.inputs:positionCommand"),
                ("SubscribeJointState.outputs:velocityCommand", "ArticulationController.inputs:velocityCommand"),
                ("SubscribeJointState.outputs:effortCommand", "ArticulationController.inputs:effortCommand"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "/clock"),
                ("PublishJointState.inputs:topicName", joint_state_topic),
                ("SubscribeJointState.inputs:topicName", cmd_topic),
                ("PublishTF.inputs:topicName", "/tf"),
                ("PublishJointState.inputs:targetPrim", [usdrt.Sdf.Path(robot_prim_path)]),
                ("PublishTF.inputs:targetPrims", [usdrt.Sdf.Path(robot_prim_path)]),
                ("ArticulationController.inputs:robotPath", robot_prim_path),
            ],
        },
    )

    logging.info(
        "ROS2 bridge graph ready: robot=%s, cmd_topic=%s, joint_state_topic=%s",
        robot_prim_path,
        cmd_topic,
        joint_state_topic,
    )


def setup_camera_graph(namespace: str, camera_prim_path: str) -> None:
    try:
        import omni.graph.core as og  # type: ignore
        import usdrt.Sdf  # type: ignore
    except Exception as exc:
        logging.warning("Could not import Isaac omnigraph modules for camera graph setup: %s", exc)
        return

    graph_path = "/G1ROSCamera"
    color_topic = f"/{namespace}/camera/color/image_raw"
    depth_topic = f"/{namespace}/camera/depth/image_raw"
    camera_info_topic = f"/{namespace}/camera/color/camera_info"

    keys = og.Controller.Keys
    ros_camera_graph, _, _, _ = og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperRgb.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperInfo.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut", "CameraHelperDepth.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperRgb.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperInfo.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "CameraHelperDepth.inputs:renderProductPath"),
            ],
            keys.SET_VALUES: [
                # Keep GUI viewport interactive by using a dedicated offscreen render product.
                ("CreateRenderProduct.inputs:cameraPrim", [usdrt.Sdf.Path(camera_prim_path)]),
                ("CreateRenderProduct.inputs:width", 640),
                ("CreateRenderProduct.inputs:height", 480),
                ("CameraHelperRgb.inputs:frameId", f"{namespace}/camera_color_optical_frame"),
                ("CameraHelperRgb.inputs:topicName", color_topic),
                ("CameraHelperRgb.inputs:type", "rgb"),
                ("CameraHelperInfo.inputs:frameId", f"{namespace}/camera_color_optical_frame"),
                ("CameraHelperInfo.inputs:topicName", camera_info_topic),
                ("CameraHelperDepth.inputs:frameId", f"{namespace}/camera_depth_optical_frame"),
                ("CameraHelperDepth.inputs:topicName", depth_topic),
                ("CameraHelperDepth.inputs:type", "depth"),
            ],
        },
    )
    og.Controller.evaluate_sync(ros_camera_graph)
    logging.info(
        "ROS2 camera graph ready: camera=%s, topics=[%s, %s, %s]",
        camera_prim_path,
        color_topic,
        depth_topic,
        camera_info_topic,
    )
