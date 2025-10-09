import os, json, cv2, random
import numpy as np
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import CompressedImage

# YOLO
os.environ["YOLO_OFFLINE"] = "true"
from ultralytics import YOLO

# EXTERNAL FILES
import config
from utils import center_crop, visualize_image, decode_image
# add your tool functions here

def rock_paper_scissors(image=None):
    # execute your choice
    choice = random.choice(["rock", "paper", "scissors"])
    ## TODO: implement your choice execution logic

    latest_image = decode_image(rospy.wait_for_message('/camera_image', CompressedImage, timeout=1.0))
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

def move_hand_to(object_name=None, image=None):
    detection = detect_object(object_name, image=image)
    if detection["found"]:
        ## TODO: implement your hand movement logic here
        return {"executed": True, "moved_to": object_name, "bbox": detection["bbox"]}
    else:
        return {"executed": False}

def grasp(object_name=None, image=None):
    if object_name is None:
        return {"executed": True, "grasped": None}
    detection = detect_object(object_name, image=image)
    if detection["found"]:
        ## TODO: implement your grasping logic here
        return {"executed": True, "grasped": detection}
    else:
        return {"executed": False}

def stretch(image=None):
    return {"executed": True}

################################################################

# add your tool functions in FUNC_REGISTRY
FUNC_REGISTRY = {
    "play_rock_paper_scissors": rock_paper_scissors,
    "wave_hand": wave_hand,
    "shake_hand": shake_hand,
    "move_hand_to": move_hand_to,
    "grasp": grasp,
    "stretch": stretch
}

# DO NOT CHANGE THIS: main function to execute tool calls
def execute_tool(tool_calls, image=None):
    
    results = []
    for tc in tool_calls:
        func_name = tc.function.name
        raw_args  = tc.function.arguments
        kwargs    = json.loads(raw_args)

        func = FUNC_REGISTRY.get(func_name)
        out = func(**kwargs, image=image)
        results.append({
            "tool_call_id": getattr(tc, "id", None),
            "function": func_name,
            "arguments": kwargs,
            "result": out
        })

    return results
