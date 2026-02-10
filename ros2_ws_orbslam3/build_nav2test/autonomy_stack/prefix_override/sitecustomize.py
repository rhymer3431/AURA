import sys
if sys.prefix == '/home/mangoo/miniconda3':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mangoo/project/ros2_ws_orbslam3/install_nav2test/autonomy_stack'
