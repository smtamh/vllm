#!/usr/bin/env python3

import os, sys, threading, json, queue

# ROS
ros_python_path = "/opt/ros/noetic/lib/python3/dist-packages"
if ros_python_path not in sys.path:
    sys.path.insert(0, ros_python_path)

import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import CompressedImage

# Virtual Environment
import cv2
import numpy as np
from vosk import Model, KaldiRecognizer
import sounddevice as sd

# External Files
import config

class StartNode:
    def __init__(self):
        # ROS setup
        os.environ['ROS_MASTER_URI'] = config.SERVER_ROS_MASTER_URI
        os.environ['ROS_IP'] = config.SERVER_ROS_IP
        
        rospy.init_node('start_node', anonymous=True)
        rospy.loginfo(f"🌐 SERVER_ROS_MASTER_URI: {config.SERVER_ROS_MASTER_URI}")
        rospy.loginfo(f"🌐 SERVER_ROS_IP: {config.SERVER_ROS_IP}")

        # Publishers
        self.text_queue = queue.Queue() # Queue for finalized STT sentences / typed input
        self.text_pub = rospy.Publisher('/audio_text', String, queue_size=10)
        self.image_pub = rospy.Publisher('/camera_image', CompressedImage, queue_size=10)
        
        # Image
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Error: Could not open camera.")
            rospy.signal_shutdown("Camera not available")
            return
        
        # STT
        self.stt_model = Model(config.STT_PATH)
        self.stop_event = threading.Event()
        self.stt_thread = threading.Thread(target=self.stt_worker, daemon=True)

        # TTS
        self.tts_playing = threading.Event()
        rospy.Subscriber('/tts_playing', Bool, self.tts_state_callback)
        
        # Type input
        self.typing_thread = threading.Thread(target=self.typing_worker, daemon=True)
        
        # Input mode: STT or typing
        if config.USE_TYPING:
            self.typing_thread.start()
        else:
            self.stt_thread.start()

        rospy.on_shutdown(self.cleanup)
        rospy.loginfo("Start Node initialized")
    
    def capture_image_and_pub(self):
        ret, frame = self.cap.read()
        if ret:
            msg = CompressedImage()
            msg.header.stamp = rospy.Time.now()
            msg.format = "jpeg"
            msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tobytes()
            self.image_pub.publish(msg)
        return ret

    def tts_state_callback(self, msg):
        if msg.data:
            self.tts_playing.set()
            # TTS started -> clear any previously finalized sentences from the text queue
            while True:
                try: self.text_queue.get_nowait()
                except queue.Empty: break
        else:
            self.tts_playing.clear()

    def stt_worker(self, samplerate=16000, blocksize=8000):
        # record audio in a separate thread and put the data into an audio queue 
        q = queue.Queue()
        def _callback(indata, frames, time, status):
            if status:
                print(status, flush=True)
            # TTS playing -> do not enqueue any audio data
            if self.tts_playing.is_set():
                return
            q.put(indata.copy().tobytes())

        with sd.InputStream(callback=_callback, channels=1,
                            samplerate=samplerate, blocksize=blocksize, dtype='int16'):
            
            rec = KaldiRecognizer(self.stt_model, samplerate)
            rec.SetWords(True)
            
            rospy.loginfo("STT worker started, listening...")
            while not self.stop_event.is_set():
                # TTS playing -> clear any previously unfinalized audio data from the audio queue and reset recognizer
                if self.tts_playing.is_set():
                    while True:
                        try: q.get_nowait()
                        except queue.Empty: break
                    
                    rec = KaldiRecognizer(self.stt_model, samplerate)
                    rec.SetWords(True)
                    rospy.sleep(0.01)
                    continue

                # process the audio data: if silence(endpoint) is detected, finalize and enqueue the sentence into the text queue
                data = q.get()
                if rec.AcceptWaveform(data):                   
                    res = json.loads(rec.Result())
                    text = res.get('text', '').strip()
                    if text:
                        self.text_queue.put(text)
                        rospy.loginfo(f"Recognized: {text}")

            # process any remaining audio
            final_res = json.loads(rec.FinalResult())
            final_text = final_res.get('text', '').strip()
            if final_text:
                self.text_queue.put(final_text)
                rospy.loginfo(f"Final Recognized: {final_text}")

    def typing_worker(self):
        while not rospy.is_shutdown():
            try:
                text = input()
                if text.strip():
                    self.text_queue.put(text.strip())
                    rospy.loginfo(f"Typed input: {text.strip()}")
            except (EOFError, KeyboardInterrupt):
                rospy.signal_shutdown("Input closed")
                break

    def publish_text_from_queue(self):
        # TTS playing -> prevent publishing previously finalized sentences from the text queue
        # no TTS -> can be ignored
        if self.tts_playing.is_set():
            return
        try:
            text = self.text_queue.get_nowait()
        except queue.Empty:
            return
        self.text_pub.publish(String(data=text))
        rospy.loginfo(f"Text published: {text}")

    def run(self):
        rate = rospy.Rate(10)               # 10 Hz
        try:
            while not rospy.is_shutdown():
                self.capture_image_and_pub()
                self.publish_text_from_queue()
                rate.sleep()
        except KeyboardInterrupt:
            rospy.loginfo("KeyboardInterrupt detected, shutting down...")
            self.cleanup()
            raise

    def cleanup(self):
            try:
                if hasattr(self, 'stop_event'):
                    self.stop_event.set()
                if hasattr(self, 'stt_thread') and self.stt_thread.is_alive():
                    self.stt_thread.join(timeout=1.0)

                if self.cap.isOpened():
                    self.cap.release()
                cv2.destroyAllWindows()
                rospy.loginfo("Resources cleaned up successfully")
            except Exception as e:
                rospy.logerr(f"Error during cleanup: {e}")

if __name__ == '__main__':
    try:
        node = StartNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Terminate STT thread
        try:
            node.stop_event.set()
            if hasattr(node, 'stt_thread'):
                node.stt_thread.join(timeout=1.0)
        except Exception:
            pass
        
        # Release camera
        try:
            if node.cap.isOpened():
                node.cap.release()
            cv2.destroyAllWindows()
        except Exception:
            pass