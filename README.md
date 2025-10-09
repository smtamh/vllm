## README

You can use **two versions** of this project:

---

### 1. Single-Demo
- **Environment:** Ubuntu 22.04.# LTS  
- **Description:**  
  - Does not use ROS.  
  - Uses local image files instead of live camera input.  
  - User can type input text instead of using a microphone.  
- **Purpose:**  
  - Quickly test VLM responses from given prompts (`role.txt`, `tools.json`).  
  - You may modify the prompts to improve VLM responses.  

---

### 2. Server + Client
- **Environment:** Ubuntu 20.04.6 LTS  
- **Description:**  
  - Uses ROS1.  
  - Input comes from both microphone (speech) and camera (image).  

**Process Flow:**  
1. Microphone records speech, camera captures images — both are sent to the server.  
2. The STT model on the server converts audio to text. ROS publishes both the captured image and the transcribed text.  
3. The client subscribes to image and text topics, then sends them as inputs to the VLM.  
4. The VLM produces two types of outputs:  
   - **Text response**  
   - **Tool call** → The tool is executed, and the VLM generates a refined text response based on the tool’s result.  
5. ROS publishes the final text response from the VLM.  
6. The TTS model on the server subscribes to the VLM’s text response and produces speech output.  

---

### Documentation
For installation and setup details, refer to:  
- `CLIENT.md`  
- `SERVER.md`  
- `SINGLE_DEMO.md`  

### Models

You should download the following models before running:

- **Download from Huggingface**  
  Run the following command inside `src_client/` folder:  
  ```bash
  python download_model.py
  ```
- **YOLO**  
  Pretrained YOLO is already included in the `models/` folder.

- **VOSK**  
  Download from https://alphacephei.com/vosk/models.  
  Save inside the `models/` folder and unzip.

- **SAM 2**  
  Download from https://github.com/facebookresearch/sam2.  
  Save inside the `models/` folder.
