@echo off
setlocal

set "ISAAC_ROOT=%AURA_ISAAC_SIM_ROOT%"
if "%ISAAC_ROOT%"=="" (
    set "ISAAC_ROOT=C:\isaac-sim"
)

if not exist "%ISAAC_ROOT%\kit\python\python.exe" (
    echo Isaac Sim console python not found: "%ISAAC_ROOT%\kit\python\python.exe"
    endlocal
    exit /B 1
)

if exist "%ISAAC_ROOT%\setup_python_env.bat" (
    call "%ISAAC_ROOT%\setup_python_env.bat"
)

set SCRIPT_DIR=%ISAAC_ROOT%\
set NO_ROS_ENV=false

for %%a in (%*) do (
    if "%%a"=="--no-ros-env" (
        set NO_ROS_ENV=true
        goto :continue
    )
)

:continue
if "%NO_ROS_ENV%"=="false" if exist "%ISAAC_ROOT%\setup_ros_env.bat" (
    call "%ISAAC_ROOT%\setup_ros_env.bat"
    set "ROS2_BRIDGE_LIB_PATH=%ISAAC_ROOT%\exts\isaacsim.ros2.bridge\%ROS_DISTRO%\lib"
    if exist "%ROS2_BRIDGE_LIB_PATH%" (
        set "PATH=%PATH%;%ROS2_BRIDGE_LIB_PATH%"
    )
)

set "CARB_APP_PATH=%ISAAC_ROOT%\kit"
set "ISAAC_PATH=%ISAAC_ROOT%\"
set "EXP_PATH=%ISAAC_ROOT%\apps"

call "%ISAAC_ROOT%\kit\python\python.exe" %*
set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /B %EXIT_CODE%
