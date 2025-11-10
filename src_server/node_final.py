#!/usr/bin/env python3

import os, sys, threading, queue, subprocess

# ROS
ros_python_path = "/opt/ros/noetic/lib/python3/dist-packages"
if ros_python_path not in sys.path:
    sys.path.insert(0, ros_python_path)

import rospy
from std_msgs.msg import String, Bool

# External Files
import config

class FinalNode:
    def __init__(self):
        # ROS setup
        os.environ['ROS_MASTER_URI'] = config.SERVER_ROS_MASTER_URI
        os.environ['ROS_IP'] = config.SERVER_ROS_IP
        
        rospy.init_node('final_node', anonymous=True)
        rospy.loginfo(f"🌐 SERVER_ROS_MASTER_URI: {config.SERVER_ROS_MASTER_URI}")
        rospy.loginfo(f"🌐 SERVER_ROS_IP: {config.SERVER_ROS_IP}")

        rospy.Subscriber('/tts_request', String, self.tts_callback)

        # TTS
        self.tts_state_pub = rospy.Publisher('/tts_playing', Bool, queue_size=1)
        rospy.sleep(0.5)
        self.q = queue.Queue()
        self.tts_thread = threading.Thread(target=self.tts_worker, daemon=True)
        self.tts_thread.start()

        rospy.loginfo("Final Node initialized")

    def tts_callback(self, msg):
        text = (msg.data or "").strip()
        if text:
            self.q.put(text)
            rospy.loginfo(f"Text added to queue: {text}")

    def tts_worker(self):
        while not rospy.is_shutdown():
            try:
                text = self.q.get(timeout=1.0)
            except queue.Empty:
                continue

            self.tts_state_pub.publish(Bool(data=True))
            try:
                rospy.loginfo(f"Speaking: {text}")
                subprocess.run(["espeak-ng", "-s", "150", "-v", "en-us", text], check=True)
                rospy.loginfo("TTS completed")
            except Exception as e:
                rospy.logerr(f"TTS unexpected error: {e}")
            finally:
                rospy.sleep(0.5)
                self.tts_state_pub.publish(Bool(data=False))
                rospy.loginfo("TTS state set to False")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        final_node = FinalNode()
        final_node.run()
    except rospy.ROSInterruptException:
        pass