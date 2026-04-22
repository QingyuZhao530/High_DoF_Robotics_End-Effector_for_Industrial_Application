#!/bin/bash
# chmod +x ~/myagv_ros/src/myagv_navigation/test/run_inspect_test_no_jetson.sh
#
# Launches the corridor inspection patrol WITHOUT Jetson:
#   - Pure move_base navigation (global planner handles curved corridors)
#   - Interactive mode (default): use RViz 2D Nav Goal to set targets
#   - Waypoint mode: pass --waypoints <yaml_file>
#
# Usage:
#   ./run_inspect_test_no_jetson.sh                     # interactive mode
#   ./run_inspect_test_no_jetson.sh --waypoints wp.yaml # fixed waypoints
#   ./run_inspect_test_no_jetson.sh jetson              # connect to Jetson roscore

source /opt/ros/noetic/setup.bash
source ~/myagv_ros/devel/setup.bash

# ── network mode ─────────────────────────────────────────────────────────
if [[ "$1" == "jetson" || "$2" == "jetson" ]]; then
    export ROS_MASTER_URI=http://192.168.1.10:11311
    export ROS_IP=192.168.1.10
    echo "Mode: AGV + Jetson (ROS_IP=192.168.1.10)"
else
    export ROS_MASTER_URI=http://localhost:11311
    unset ROS_IP
    echo "Mode: AGV alone (localhost)"
fi

# ── waypoints file ────────────────────────────────────────────────────────
WAYPOINT_YAML=""
INTERACTIVE=true
for arg in "$@"; do
    if [[ "$arg" == "--waypoints" ]]; then
        NEXT_IS_WP=1
    elif [[ -n "$NEXT_IS_WP" ]]; then
        WAYPOINT_YAML="$arg"
        INTERACTIVE=false
        NEXT_IS_WP=""
    fi
done

LOG_DIR=~/inspect_no_jetson_logs
mkdir -p "$LOG_DIR"

MAP_DIR=~/myagv_ros/src/myagv_navigation/map
MAP_YAML="${MAP_DIR}/map.yaml"

echo "======================================"
echo "   Starting Inspect (No Jetson)"
echo "======================================"

# ── pre-flight check ──────────────────────────────────────────────────────
if [ ! -f "$MAP_YAML" ]; then
    echo "ERROR: Map not found at $MAP_YAML"
    echo "Run nav_test_ver1 or nav_test_ver2 first to build a map."
    exit 1
fi

# ===== 0. roscore =====
echo "[0/5] Starting roscore..."
roscore > "$LOG_DIR/roscore.log" 2>&1 &
PID_ROSCORE=$!

for i in $(seq 1 15); do
    if rostopic list > /dev/null 2>&1; then
        echo "       roscore running (${i}s)"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "ERROR: roscore did not start."
        kill $PID_ROSCORE 2>/dev/null
        exit 1
    fi
    sleep 1
done

# ===== 1. YDLidar =====
echo "[1/5] Starting YDLidar..."
cd ~/myagv_ros/src/myagv_odometry/scripts
./start_ydlidar.sh > "$LOG_DIR/ydlidar.log" 2>&1 &
PID_LIDAR=$!
cd ~
sleep 5

# ===== 2. AGV base =====
echo "[2/5] Starting AGV base..."
roslaunch myagv_odometry myagv_active.launch > "$LOG_DIR/agv_base.log" 2>&1 &
PID_BASE=$!
sleep 5

# ===== 3. Navigation (map_server + AMCL + move_base) =====
echo "[3/5] Starting navigation (map_server + AMCL + move_base)..."
roslaunch myagv_navigation navigation_active.launch map:=$MAP_YAML \
    > "$LOG_DIR/navigation.log" 2>&1 &
PID_NAV=$!
sleep 8

# ===== 4. (optional) extra move_base if not included above =====
# roslaunch myagv_navigation exploration_nav.launch > "$LOG_DIR/move_base.log" 2>&1 &
# PID_MOVEBASE=$!
# sleep 5

# ===== 5. inspect_test_no_jetson.py =====
echo "[5/5] Starting inspect_test_no_jetson.py..."
rosparam set /inspect_no_jetson/goal_timeout    120.0
rosparam set /inspect_no_jetson/pause_secs        2.0
rosparam set /inspect_no_jetson/max_retries         2
rosparam set /inspect_no_jetson/arrival_dist      0.30
rosparam set /inspect_no_jetson/return_to_start   true

if [ "$INTERACTIVE" = true ]; then
    rosparam set /inspect_no_jetson/interactive true
    echo "  Mode: INTERACTIVE (use RViz 2D Nav Goal)"
else
    rosparam set /inspect_no_jetson/interactive     false
    rosparam set /inspect_no_jetson/waypoint_yaml   "$WAYPOINT_YAML"
    echo "  Mode: WAYPOINT  ($WAYPOINT_YAML)"
fi

python3 ~/myagv_ros/src/myagv_navigation/test/inspect_test_no_jetson.py \
    > "$LOG_DIR/inspect_no_jetson.log" 2>&1 &
PID_INSPECT=$!

echo ""
echo "======================================"
echo "   Environment Ready"
echo "======================================"
echo ""
if [ "$INTERACTIVE" = true ]; then
    echo "STEP 1 --> In RViz: '2D Pose Estimate' — set robot's initial position."
    echo "STEP 2 --> In RViz: '2D Nav Goal'      — click any target point."
    echo "           The robot navigates via move_base (handles curves correctly)."
    echo "           Repeat STEP 2 for more targets."
else
    echo "STEP 1 --> In RViz: '2D Pose Estimate' — set robot's initial position."
    echo "           The robot will then navigate to all waypoints automatically."
fi
echo ""
echo "Logs: $LOG_DIR/"
echo "  tail -f $LOG_DIR/inspect_no_jetson.log"
echo ""
echo "Press q to stop robot and shut everything down."
echo "======================================"

# ── wait for q ────────────────────────────────────────────────────────────
while true; do
    read -n 1 key
    if [[ $key == "q" ]]; then
        echo ""
        echo "Stopping robot..."
        kill $PID_INSPECT 2>/dev/null || true
        sleep 0.5
        timeout 3 rostopic pub -1 /cmd_vel geometry_msgs/Twist \
            '{linear: {x: 0.0}, angular: {z: 0.0}}' 2>/dev/null || true
        sleep 1
        echo "Shutting down all processes..."
        kill $PID_NAV $PID_BASE $PID_LIDAR 2>/dev/null || true
        sleep 1
        kill $PID_ROSCORE 2>/dev/null || true
        killall -9 rosmaster roscore roslaunch 2>/dev/null || true
        echo "System stopped."
        break
    fi
done