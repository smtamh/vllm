#!/usr/bin/env python3

import os, sys
from datetime import datetime

# ROS
ros_python_path = "/opt/ros/noetic/lib/python3/dist-packages"
if ros_python_path not in sys.path:
    sys.path.insert(0, ros_python_path)

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

# Virtual Environment
from PIL import Image as PILImage
from vllm import LLM, SamplingParams
from vllm.entrypoints.openai.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
import numpy as np
import cv2

# External Files
import config
from utils import build_prompt
from tools import execute_tool
from message import MsgBuffer

class InferenceNode:
    def __init__(self):
        # ROS setup
        os.environ['ROS_MASTER_URI'] = config.SERVER_ROS_MASTER_URI
        os.environ['ROS_IP'] = config.CLIENT_ROS_IP
        
        rospy.init_node('inference_node', anonymous=True)
        rospy.loginfo(f"🌐 SERVER_ROS_MASTER_URI: {config.SERVER_ROS_MASTER_URI}")
        rospy.loginfo(f"🌐 SERVER_ROS_IP: {config.CLIENT_ROS_IP}")

        # Publishers
        self.tts_pub = rospy.Publisher('/tts_request', String, queue_size=10)

        # Initialize VLM
        self.sampling_params = SamplingParams(**config.SAMPLING_PARAMS)
        system_prompt = build_prompt(config.ROLE_PATH, config.TOOL_PATH)
        self.conversation = MsgBuffer(system_prompt, max_groups=5)
        
        self.vlm = LLM(
            model=config.VLM_MODEL_PATH,
            generation_config="vllm",
            gpu_memory_utilization=config.VLM_GPU_UTIL,
            max_model_len=config.VLM_MAX_LENGTH,
            limit_mm_per_prompt=config.VLM_LIMIT_INPUT
        )
        
        self.parser = Hermes2ProToolParser(tokenizer=self.vlm.get_tokenizer())

        # Subscribers
        rospy.Subscriber('/audio_text', String, self.stt_callback)
        rospy.Subscriber('/camera_image', CompressedImage, self.image_callback)

        # Image
        self.latest_image = None
        
        rospy.loginfo("Inference Node initialized")
    
    def image_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        except Exception as e:
            rospy.logerr(f"Image callback error: {e}")

    def stt_callback(self, msg):
        try:
            user_input = msg.data
            if not user_input:
                return
            
            rospy.loginfo(f"Processing text: {user_input}")
            
            # Load image
            if self.latest_image is not None:
                self.conversation.start_user(user_input, self.latest_image)

                # # Save image
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # image_path = os.path.join(config.IMAGE_DIR, f"image_{timestamp}.jpg")
                # self.latest_image.save(image_path)
                # rospy.loginfo(f"Image saved to {image_path}")
            else:
                rospy.logwarn("No valid image available.")
                return
            
            # Generate response
            output = self.vlm.chat(
                messages=self.conversation.to_messages(),
                sampling_params=self.sampling_params,
                use_tqdm=True
            )
            
            answer = output[0].outputs[0].text.strip()
            self.conversation.append_assistant(answer)
            info = self.parser.extract_tool_calls(answer, request=None)
                                   
            # Handle tool calls
            if info.tools_called:
                self.handle_tool_calls(info.tool_calls)
            else:
                rospy.loginfo(self.conversation)

                # Send to TTS
                tts_msg = String()
                tts_msg.data = answer
                self.tts_pub.publish(tts_msg)

            # Visualize (optional)
            self.visualize_image()
            
        except Exception as e:
            rospy.logerr(f"Inference Error: {e}")
    
    def handle_tool_calls(self, tool_calls):
        try:
            # Update image for tool execution
            if self.latest_image is not None:
                self.conversation.update_image(self.latest_image)
                
                # # Save image
                # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # image_path = os.path.join(config.IMAGE_DIR, f"image_{timestamp}.jpg")
                # self.latest_image.save(image_path)
                # rospy.loginfo(f"Image saved to {image_path}")
            else:
                rospy.logwarn("No valid image available for tool execution.")
                return
            
            # Execute tools
            tool_result = execute_tool(tool_calls, image=self.latest_image)
            self.conversation.append_tool(tool_result)
            
            # Generate post-tool response
            post_output = self.vlm.chat(
                messages=self.conversation.to_messages(),
                sampling_params=self.sampling_params,
                use_tqdm=True
            )
            
            post_answer = post_output[0].outputs[0].text.strip()
            self.conversation.append_assistant(post_answer)
            
            rospy.loginfo(self.conversation)
            
            # Send to TTS
            tts_msg = String()
            tts_msg.data = post_answer
            self.tts_pub.publish(tts_msg)

            self.visualize_image()
            
        except Exception as e:
            rospy.logerr(f"Tool execution error: {e}")
    
    def visualize_image(self):
        try:
            cv_image = cv2.cvtColor(np.array(self.latest_image), cv2.COLOR_RGB2BGR)
            cv2.imshow("Input Image", cv_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        except Exception as e:
            rospy.logerr(f"Visualization error: {e}")
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        inference_node = InferenceNode()
        inference_node.run()
    except rospy.ROSInterruptException:
        pass