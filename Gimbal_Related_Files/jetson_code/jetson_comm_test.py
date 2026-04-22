#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jetson_comm_test.py
ROS multi-machine communication test between AGV and Jetson.

This script can run on EITHER machine. It performs:
  1. Connectivity check: verifies ROS master is reachable.
  2. Topic check: confirms /map, /tf, /cmd_vel are publishing.
  3. Map sharing test:
     - Subscribes to /map and saves the latest OccupancyGrid as
       map.pgm + map.yaml to a specified directory.
     - This lets the Jetson obtain the map without needing scp.
  4. Defect candidate publish test:
     - Publishes a dummy PoseStamped on /defect_candidates to verify
       that nav_test_ver2.py (on the AGV) can receive it.

Usage on Jetson
---------------
  # Make sure ROS env vars are set:
  #   export ROS_MASTER_URI=http://<AGV_IP>:11311
  #   export ROS_IP=<JETSON_IP>
  python3 jetson_comm_test.py

  # Or with arguments:
  python3 jetson_comm_test.py --save-map --map-dir /home/jetson/maps
  python3 jetson_comm_test.py --publish-dummy-defect
  python3 jetson_comm_test.py --all

Usage on AGV (for self-test)
----------------------------
  python3 jetson_comm_test.py --all
"""

import argparse
import math
import os
import sys
import time

import numpy as np

try:
    import rospy
    import tf
    from nav_msgs.msg      import OccupancyGrid
    from geometry_msgs.msg import PoseStamped
    from std_msgs.msg      import String
except ImportError:
    print("ERROR: ROS Python packages not found.")
    print("Make sure you have sourced /opt/ros/noetic/setup.bash")
    sys.exit(1)


class CommTest:

    def __init__(self, map_dir, save_map, publish_defect):
        self.map_dir        = map_dir
        self.save_map       = save_map
        self.publish_defect = publish_defect

        self.map_msg  = None
        self.tf_ok    = False

    # ─────────────────────── 1. Master connectivity ────────────────────
    def check_master(self):
        print("\n[1] Checking ROS Master connectivity...")
        master_uri = os.environ.get("ROS_MASTER_URI", "not set")
        ros_ip     = os.environ.get("ROS_IP", "not set")
        print("    ROS_MASTER_URI = %s" % master_uri)
        print("    ROS_IP         = %s" % ros_ip)

        try:
            rospy.init_node("jetson_comm_test", anonymous=True)
            print("    [OK] Connected to ROS master.")
            return True
        except Exception as e:
            print("    [FAIL] Cannot connect to ROS master: %s" % e)
            return False

    # ─────────────────────── 2. Topic availability ─────────────────────
    def check_topics(self):
        print("\n[2] Checking key topics...")
        required = ["/map", "/tf", "/cmd_vel", "/scan"]
        published = [t for (t, _) in rospy.get_published_topics()]

        all_ok = True
        for topic in required:
            found = topic in published
            status = "OK" if found else "NOT FOUND"
            print("    %-20s [%s]" % (topic, status))
            if not found:
                all_ok = False

        if all_ok:
            print("    All required topics are active.")
        else:
            print("    WARNING: Some topics are missing. "
                  "Make sure SLAM / lidar / base are running.")
        return all_ok

    # ─────────────────────── 3. TF check ───────────────────────────────
    def check_tf(self):
        print("\n[3] Checking TF (map -> base_footprint)...")
        listener = tf.TransformListener()
        for i in range(30):  # wait up to 3 seconds
            try:
                listener.waitForTransform(
                    "map", "base_footprint", rospy.Time(0), rospy.Duration(0.1))
                (trans, _) = listener.lookupTransform(
                    "map", "base_footprint", rospy.Time(0))
                print("    [OK] Robot position: (%.3f, %.3f, %.3f)" %
                      (trans[0], trans[1], trans[2]))
                self.tf_ok = True
                return True
            except Exception:
                pass
            rospy.sleep(0.1)

        print("    [FAIL] TF map->base_footprint not available after 3 s.")
        print("    Hint: run 'rostopic echo /tf' on Jetson to check if TF data flows.")
        return False

    # ─────────────────────── 4. Map subscription + save ────────────────
    def _map_cb(self, msg):
        self.map_msg = msg

    def check_and_save_map(self):
        print("\n[4] Subscribing to /map...")
        sub = rospy.Subscriber("/map", OccupancyGrid, self._map_cb, queue_size=1)

        for i in range(50):  # wait up to 5 seconds
            if self.map_msg is not None:
                break
            rospy.sleep(0.1)

        sub.unregister()

        if self.map_msg is None:
            print("    [FAIL] No /map message received after 5 s.")
            print("    Is SLAM running on the AGV?")
            return False

        info = self.map_msg.info
        print("    [OK] Map received: %d x %d, resolution=%.3f m/px" %
              (info.width, info.height, info.resolution))
        print("    Origin: (%.3f, %.3f)" %
              (info.origin.position.x, info.origin.position.y))

        if not self.save_map:
            print("    (Use --save-map to write map.pgm + map.yaml)")
            return True

        # ── save PGM ──────────────────────────────────────────────────
        os.makedirs(self.map_dir, exist_ok=True)
        pgm_path  = os.path.join(self.map_dir, "map.pgm")
        yaml_path = os.path.join(self.map_dir, "map.yaml")

        data = np.array(self.map_msg.data, dtype=np.int8)
        h, w = info.height, info.width
        grid = data.reshape((h, w))

        # OccupancyGrid: -1=unknown, 0=free, 100=occupied
        # PGM (map_server convention): 254=free, 0=occupied, 205=unknown
        pgm = np.full((h, w), 205, dtype=np.uint8)
        pgm[grid == 0]   = 254
        pgm[grid >= 50]  = 0
        # Values 1-49 → proportional grey
        mid = (grid > 0) & (grid < 50)
        pgm[mid] = (254 - (grid[mid].astype(np.float32) / 50.0 * 254)).astype(np.uint8)

        # PGM is stored top-row-first; OccupancyGrid origin is bottom-left
        pgm = np.flipud(pgm)

        # Write binary PGM (P5)
        with open(pgm_path, "wb") as f:
            header = "P5\n%d %d\n255\n" % (w, h)
            f.write(header.encode("ascii"))
            f.write(pgm.tobytes())

        print("    Saved: %s" % pgm_path)

        # ── save YAML ─────────────────────────────────────────────────
        yaml_content = (
            "image: map.pgm\n"
            "resolution: %.6f\n"
            "origin: [%.6f, %.6f, 0.0]\n"
            "negate: 0\n"
            "occupied_thresh: 0.65\n"
            "free_thresh: 0.196\n"
        ) % (info.resolution,
             info.origin.position.x,
             info.origin.position.y)

        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        print("    Saved: %s" % yaml_path)
        return True

    # ─────────────────────── 5. Publish dummy defect ───────────────────
    def test_defect_publish(self):
        print("\n[5] Publishing dummy defect candidate on /defect_candidates...")

        pub = rospy.Publisher("/defect_candidates", PoseStamped, queue_size=1)
        rospy.sleep(0.5)  # let publisher register

        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp    = rospy.Time.now()
        msg.pose.position.x = 1.0
        msg.pose.position.y = 0.5
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0

        pub.publish(msg)
        print("    [OK] Published dummy defect at (1.0, 0.5)")
        print("    If nav_test_ver2.py is running, you should see:")
        print("      'nav_explore: defect candidate #N at (1.00, 0.50)'")

        # Also test /inspect_trigger reception
        trigger_pub = rospy.Publisher("/inspect_trigger", String, queue_size=1)
        rospy.sleep(0.3)
        trigger_pub.publish(String(data="test_trigger"))
        print("    [OK] Published test message on /inspect_trigger")

        return True

    # ─────────────────────── summary ───────────────────────────────────
    def run(self):
        results = {}

        ok = self.check_master()
        results["master"] = ok
        if not ok:
            print("\n=== SUMMARY: Cannot reach ROS master. Aborting. ===")
            return results

        results["topics"] = self.check_topics()
        results["tf"]     = self.check_tf()
        results["map"]    = self.check_and_save_map()

        if self.publish_defect:
            results["defect_pub"] = self.test_defect_publish()

        # ── summary ───────────────────────────────────────────────────
        print("\n======================================")
        print("   Communication Test Summary")
        print("======================================")
        for name, ok in results.items():
            print("  %-15s [%s]" % (name, "PASS" if ok else "FAIL"))

        all_pass = all(results.values())
        if all_pass:
            print("\nAll tests passed. AGV <-> Jetson ROS communication is working.")
        else:
            print("\nSome tests failed. Check the output above for details.")
            print("\nCommon fixes:")
            print("  1. Ensure AGV and Jetson are on the same network (ping test)")
            print("  2. Set ROS_MASTER_URI and ROS_IP correctly on both machines")
            print("  3. Make sure SLAM and base nodes are running on the AGV")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Test ROS communication between AGV and Jetson")
    parser.add_argument("--save-map", action="store_true",
                        help="Save the /map topic as map.pgm + map.yaml")
    parser.add_argument("--map-dir", default=os.path.expanduser("~/maps"),
                        help="Directory to save map files (default: ~/maps)")
    parser.add_argument("--publish-dummy-defect", action="store_true",
                        help="Publish a dummy defect candidate for testing")
    parser.add_argument("--all", action="store_true",
                        help="Run all tests including save-map and dummy defect")
    args = parser.parse_args()

    if args.all:
        args.save_map = True
        args.publish_dummy_defect = True

    tester = CommTest(
        map_dir=args.map_dir,
        save_map=args.save_map,
        publish_defect=args.publish_dummy_defect,
    )
    tester.run()


if __name__ == "__main__":
    main()