from setuptools import setup

package_name = "autonomy_stack"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/autonomy_stack"]),
        ("share/autonomy_stack", ["package.xml"]),
        (
            "share/autonomy_stack/launch",
            [
                "launch/autonomy.launch.py",
                "launch/nav2_orbslam.launch.py",
            ],
        ),
        (
            "share/autonomy_stack/config",
            [
                "config/autonomy.yaml",
                "config/nav2_orbslam.yaml",
                "config/frontier_explorer.yaml",
            ],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="mangoo",
    maintainer_email="mangoo@localhost",
    description="Minimal ROS 2 autonomy stack for Habitat-Sim RGB-D pipelines.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigator_node = autonomy_stack.navigator_node:main",
            "yoloe_semantic_node = autonomy_stack.yoloe_semantic_node:main",
            "semantic_fusion_node = autonomy_stack.semantic_fusion_node:main",
            "sparse_map_occupancy_node = autonomy_stack.sparse_map_occupancy_node:main",
            "pose_tf_bridge_node = autonomy_stack.pose_tf_bridge_node:main",
            "frontier_explorer_node = autonomy_stack.frontier_explorer_node:main",
        ],
    },
)
