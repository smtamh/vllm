import os, json, cv2, random, time, subprocess
import numpy as np
import matplotlib.pyplot as plt

# ROS
import rospy
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from tocabi_msgs.msg import positionCommand, TaskCommand
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Pose

# YOLO
os.environ["YOLO_OFFLINE"] = "true"
from ultralytics import YOLO

# EXTERNAL FILES
import config
from utils import center_crop, visualize_image, decode_image
# add your tool functions here

def init_tool_publishers():
    global mode_pub, hand_open_pub, task_command_pub, joint_state_pub, tts_pub, rrt_pub, target_pub, grasped_pub
    mode_pub = rospy.Publisher('/tocabi/act/mode', Int32, queue_size=10)
    hand_open_pub = rospy.Publisher('/mujoco_ros_interface/hand_open', Int32, queue_size=10)
    task_command_pub = rospy.Publisher('/tocabi/taskcommand', TaskCommand, queue_size=10)
    joint_state_pub = rospy.Publisher('/tocabi/positioncommand', positionCommand, queue_size=10)
    tts_pub = rospy.Publisher('/tts_request', String, queue_size=10)
    rrt_pub = rospy.Publisher('/tocabi/srmt/start_rrt', Bool, queue_size=1)
    target_pub = rospy.Publisher('/target_pose', Pose, queue_size=1)
    grasped_pub = rospy.Publisher('/tocabi/obj_grasped', Bool, queue_size=1)
    rospy.loginfo(rospy.get_published_topics())
    rospy.sleep(0.5)

def publish_to_sim():
    env = os.environ.copy()
    env["ROS_MASTER_URI"] = config.CLIENT_ROS_MASTER_URI
    env["ROS_IP"] = config.CLIENT_ROS_IP
    return env

def rock_paper_scissors(image=None):
    tts_msg  = String()
    tts_msg.data = "Rock, Paper, Scissors!"
    tts_pub.publish(tts_msg)
    time.sleep(0.5)

    # if you use simulator in the client side, change the ROS environment
    # env = publish_to_sim()

    # execute your choice
    choice = random.choice(["rock", "paper", "scissors"])

    ## implement your choice execution logic
    msg = Int32()
    if  choice == "rock":
        msg.data = 1
    elif choice == "paper":
        msg.data = 0
    elif choice == "scissors":
        msg.data = 2

    # # if you use simulator in the client side
    # subprocess.Popen([
    #     "rostopic", "pub", "-1",
    #     "/mujoco_ros_interface/hand_open", "std_msgs/Int32", f"data: {msg.data}"
    # ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # time.sleep(1)

    # if you use simulator in the server side == default (Also for the real robot)
    hand_open_pub.publish(msg)
    time.sleep(2)

    # get opponent's choice
    latest_image = decode_image(rospy.wait_for_message('/camera_image', CompressedImage, timeout=5.0))
    config.CURRENT_IMAGE = latest_image
    
    # check opponent's choice using YOLO
    model = YOLO(config.YOLO_PATH)
    image_cropped = center_crop(latest_image, 320)
    results = model.predict(source=image_cropped, conf=0.25)

    opponent_choice = "none"
    opponent_box = None
    
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            scores  = r.boxes.conf.tolist()
            classes = r.boxes.cls.tolist()

            max_idx = scores.index(max(scores))
            opponent_cls = int(classes[max_idx])
            opponent_box = r.boxes.xyxy.tolist()[max_idx]  # [x1, y1, x2, y2]

            if opponent_cls == 0:
                opponent_choice = "paper"
            elif opponent_cls == 1:
                opponent_choice = "rock"
            elif opponent_cls == 2:
                opponent_choice = "scissors"
            else:
                opponent_choice = "unknown"

    if opponent_choice == "paper":
        if choice == "rock":
            winner = "user"
        elif choice == "paper":
            winner = "tie"
        else:
            winner = "assistant"
    elif opponent_choice == "rock":
        if choice == "rock":
            winner = "tie"
        elif choice == "paper":
            winner = "assistant"
        else:
            winner = "user"
    elif opponent_choice == "scissors":
        if choice == "rock":
            winner = "assistant"
        elif choice == "paper":
            winner = "user"
        else:
            winner = "tie"
    else:
        winner = "unknown"

    # visualize
    if opponent_box is not None:
        annotated = image_cropped.copy()
        x1, y1, x2, y2 = map(int, opponent_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, opponent_choice, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLO Result", annotated)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    else:
        visualize_image(image_cropped)

    result = {"executed": True, "assistant_move": choice, "user_move": opponent_choice, "winner": winner}
    return result

def wave_hand(image=None):
    return {"executed": True}

def shake_hand(image=None):
    return {"executed": True}

def detect_object(object, image=None):
    
    # LangSAM
    detection = config.DETECTOR.predict([image], [object])
    res = detection[0]
    boxes = res['boxes']
    labels = res['labels']
    scores = res['scores']

    if object in labels:
        idxs = [i for i, lab in enumerate(labels) if lab == object]
        bboxes = [boxes[i].tolist() for i in idxs]
        result = {
            "object": object,
            "found": True,
            "bbox": bboxes
        }
    else:
        result = {
            "object": object,
            "found": False,
            "bbox": []
        }

    # visualize
    if result["found"]:
        img_np = np.array(image).copy()
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        for i, bbox in enumerate(result["bbox"]):
            x1, y1, x2, y2 = map(int, bbox)
            score = scores[idxs[i]]
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_bgr, f"{object} ({score:.2f})", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detection Result", img_bgr)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    else:
        visualize_image(image)

    return result

def move_hand_to(x, y, z=None, image=None):
    # target range: x in [0.4, 0.9], y in [-0.55, 0.55]
    target = [x, y, z if z is not None else 1.025]
    
    target_msg = Pose()
    target_msg.position.x = target[0]
    target_msg.position.y = target[1]
    target_msg.position.z = target[2]

    target_msg.orientation.x = 0.0
    target_msg.orientation.y = 0.0
    target_msg.orientation.z = 0.0
    target_msg.orientation.w = 1.0
    target_pub.publish(target_msg)
    time.sleep(1)
    
    rrt_msg = Bool()
    rrt_msg.data = True
    rrt_pub.publish(rrt_msg)
    time.sleep(1)
    return {"executed": True, "moved_to": target}

def grasp(object_name=None, image=None):
    # msg = Int32()
    # msg.data = 1
    # if object_name is None:
    #     ## implement your grasping logic here
    #     time.sleep(2)
    #     return {"executed": True, "grasped": None}
    # detection = detect_object(object_name, image=image)
    # if detection["found"]:
    #     ## implement your grasping logic here
    #     hand_open_pub.publish(msg)
    #     time.sleep(2)
    #     return {"executed": True, "grasped": detection}
    # else:
    #     return {"executed": False}

    if object_name is not None:
        detection = detect_object(object_name, image=image)
        if detection["found"]:
            rrt_msg = Bool()
            rrt_msg.data = True
            rrt_pub.publish(rrt_msg)
            time.sleep(1)

        try:
            msg = rospy.wait_for_message('/tocabi/srmt/end_rrt', Bool, timeout=10.0)
        except rospy.ROSException:
            return {"executed": False}
        
        if msg.data is True:
            hand_msg = Int32()
            hand_msg.data = 1
            hand_open_pub.publish(hand_msg)
            time.sleep(1)

            grasp_msg = Bool()
            grasp_msg.data = True
            grasped_pub.publish(grasp_msg)
            time.sleep(1)

    else:
        hand_msg = Int32()
        hand_msg.data = 1
        hand_open_pub.publish(hand_msg)
        time.sleep(1)

    return {"executed": True, "grasped": object_name}

def stretch(image=None):
    hand_msg = Int32()
    hand_msg.data = 0
    hand_open_pub.publish(hand_msg)
    time.sleep(1)

    grasp_msg = Bool()
    grasp_msg.data = False
    grasped_pub.publish(grasp_msg)
    time.sleep(1)
    return {"executed": True}

################################################################

# add your tool functions in FUNC_REGISTRY
FUNC_REGISTRY = {
    "play_rock_paper_scissors": rock_paper_scissors,
    "wave_hand": wave_hand,
    "shake_hand": shake_hand,
    "move_hand_to": move_hand_to,
    "grasp": grasp,
    "stretch_hand": stretch
}

# DO NOT CHANGE THIS: main function to execute tool calls
def execute_tool(tool_calls, image=None):
    
    results = []
    for tc in tool_calls:
        func_name = tc.function.name
        raw_args  = tc.function.arguments
        func      = FUNC_REGISTRY.get(func_name)
        kwargs    = json.loads(raw_args)

        out = func(**kwargs, image=image)
        
        results.append({
            "tool_call_id": getattr(tc, "id", None),
            "function": func_name,
            "arguments": kwargs,
            "result": out
        })

        time.sleep(3)

    return results
