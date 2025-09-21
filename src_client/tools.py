import os, json, cv2, random
os.environ["YOLO_OFFLINE"] = "true"

from ultralytics import YOLO

import config
from utils import center_crop
# add your tool functions here

def rock_paper_scissors(image=None):
    # execute your choice
    # TODO: implement your choice execution logic

    choice = random.choice(["rock", "paper", "scissors"])
    # check opponent's choice using YOLO

    model = YOLO(config.YOLO_PATH)
    image_cropped = center_crop(image, 320)
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

    # visualize
    if opponent_box is not None:
        annotated = image_cropped.copy()
        x1, y1, x2, y2 = map(int, opponent_box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, opponent_choice, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("YOLO Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    result = {"assistant_move": choice, "user_move": opponent_choice}
    return result

def say_hello(image=None):
    return {"message": "Hello!"}

################################################################

# add your tool functions in FUNC_REGISTRY
FUNC_REGISTRY = {
    "rock_paper_scissors": rock_paper_scissors,
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
            "executed": True,
            "result": out
        })

    return results
