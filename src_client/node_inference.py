#!/usr/bin/env python3

import os, sys, time

# ROS
ros_python_path = "/opt/ros/noetic/lib/python3/dist-packages"
if ros_python_path not in sys.path:
    sys.path.insert(0, ros_python_path)

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# Virtual Environment
from PIL import Image
import numpy as np
import cv2

# External Files
import config
from utils import build_prompt, visualize_image, decode_image, save_image
from tools import execute_tool, init_tool_publishers
from message import MsgBuffer

class InferenceNode:
    def __init__(self):
        from vllm import LLM, SamplingParams
        from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
        from lang_sam import LangSAM

        # ROS setup
        os.environ['ROS_MASTER_URI'] = config.SERVER_ROS_MASTER_URI
        os.environ['ROS_IP'] = config.CLIENT_ROS_IP
        
        rospy.init_node('inference_node', anonymous=True)
        init_tool_publishers()
        rospy.loginfo(f"🌐 SERVER_ROS_MASTER_URI: {config.SERVER_ROS_MASTER_URI}")
        rospy.loginfo(f"🌐 CLIENT_ROS_IP: {config.CLIENT_ROS_IP}")

        # Publishers
        self.tts_pub = rospy.Publisher('/tts_request', String, queue_size=10)
        rospy.sleep(0.5)

        # Initialize VLM
        self.sampling_params = SamplingParams(**config.SAMPLING_PARAMS)
        system_prompt = build_prompt(config.ROLE_PATH, config.TOOL_PATH)
        self.conversation = MsgBuffer(system_prompt, max_groups=3)
        
        self.vlm = LLM(
            model=config.VLM_MODEL_PATH,
            generation_config="vllm",
            gpu_memory_utilization=config.VLM_GPU_UTIL,
            max_model_len=config.VLM_MAX_LENGTH,
            limit_mm_per_prompt=config.VLM_LIMIT_INPUT
        )
        
        config.DETECTOR = LangSAM(
            sam_ckpt_path=config.SAM2_PATH,
            gdino_model_ckpt_path=config.GDINO_PATH,
            gdino_processor_ckpt_path=config.GDINO_PATH,
            sam_type=config.SAM2_TYPE
        )
        self.parser = Hermes2ProToolParser(tokenizer=self.vlm.get_tokenizer())

        # Subscribers
        rospy.Subscriber('/audio_text', String, self.stt_callback)
        # rospy.Subscriber('/camera_image', CompressedImage, self.cam_image_callback)   # use when you don't use sim_image
        rospy.Subscriber('/mujoco_ros_interface/cam_L/image/compressed', CompressedImage, self.sim_image_callback)

        # Image
        self.latest_image = None
        
        rospy.loginfo("Inference Node initialized")
    
    def sim_image_callback(self, msg):
        try:
            self.latest_image = decode_image(msg)
        except Exception as e:
            rospy.logerr(f"Sim Image callback error: {e}")

    def cam_image_callback(self, msg):
        try:
            self.latest_image = decode_image(msg)
        except Exception as e:
            rospy.logerr(f"Cam Image callback error: {e}")

    def stt_callback(self, msg):
        try:
            t1 = time.time()
            user_input = msg.data
            if not user_input:
                return
            
            rospy.loginfo(f"Processing text: {user_input}")
            
            # Load image
            if self.latest_image is not None:
                self.conversation.start_user(user_input, self.latest_image)
                # save_image(self.latest_image)
            else:
                rospy.logwarn("No valid image available.")
                return
            
            # Generate response
            output = self.vlm.chat(
                messages=self.conversation.to_messages(),
                sampling_params=self.sampling_params,
                use_tqdm=True
            )
            t2 = time.time()
            answer = output[0].outputs[0].text.strip()
            self.conversation.append_assistant(answer)
            info = self.parser.extract_tool_calls(answer, request=None)
                                   
            # Handle tool calls
            if info.tools_called:
                self.handle_tool_calls(info.tool_calls)
            else:
                rospy.loginfo(self.conversation)
                rospy.loginfo(f"Response Time: {t2 - t1:.2f} seconds")

                # Send to TTS
                tts_msg = String()
                tts_msg.data = answer
                self.tts_pub.publish(tts_msg)
                time.sleep(0.5)
                visualize_image(self.latest_image)
            
        except Exception as e:
            rospy.logerr(f"Inference Error: {e}")
    
    def handle_tool_calls(self, tool_calls):
        try:
            current_tool_calls = tool_calls

            while current_tool_calls:
                # Update image for tool execution
                if self.latest_image is not None:
                    self.conversation.update_image(self.latest_image)
                    # visualize_image(self.latest_image)
                    # save_image(self.latest_image)
                else:
                    rospy.logwarn("No valid image available for tool execution.")
                    return
                
                # Execute tools
                tool_result = execute_tool(current_tool_calls, image=self.latest_image)
                self.conversation.append_tool(tool_result)
                if config.CURRENT_IMAGE is not None:
                    self.conversation.update_image(config.CURRENT_IMAGE)
                    config.CURRENT_IMAGE = None
                    # visualize_image(config.CURRENT_IMAGE)
                    # save_image(config.CURRENT_IMAGE)
                
                # Generate post-tool response
                post_output = self.vlm.chat(
                    messages=self.conversation.to_messages(),
                    sampling_params=self.sampling_params,
                    use_tqdm=True
                )
            
                post_answer = post_output[0].outputs[0].text.strip()
                self.conversation.append_assistant(post_answer)
                info = self.parser.extract_tool_calls(post_answer, request=None)
                current_tool_calls = info.tool_calls if info else None

                rospy.loginfo(self.conversation)
                
                # Send to TTS
                tts_msg = String()
                tts_msg.data = post_answer
                self.tts_pub.publish(tts_msg)
            
        except Exception as e:
            rospy.logerr(f"Tool execution error: {e}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        inference_node = InferenceNode()
        inference_node.run()
    except rospy.ROSInterruptException:
        pass