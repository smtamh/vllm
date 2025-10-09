## Installation
### System Packages
    sudo apt install libgtk2.0-dev pkg-config

### Python Packages
Install packages in `python 3.10` virtual environment in `Ubuntu 20.04.6 LTS`:
    
    uv pip install rospkg vllm ultralytics

You should also set third party packages.

    git submodule update --init --recursive
    cd vllm/third_party/flashinfer
    uv pip install -e . --no-build-isolation -v

    cd vllm/third_party/lang-segment-anything
    uv pip install -e . --no-build-isolation -v

To enable image visualization, replace `opencv-python-headless` with `opencv-python`:

    uv pip uninstall opencv-python-headless
    uv pip install --force-reinstall --no-cache-dir opencv-python   
    
<br>

## SSH Setting
Configure SSH for remote server access:

    nano ~/.ssh/config

```
Host SERVER_NAME
    HostName SERVER_IP
    User SERVER_USER          
    Port 22                 # SSH Port (original value)
```

<br>

## Inference
Check server IP and client IP first:

    hostname -I

Update `config.py - # Network Settings` in the `src_client/` and `src_server/` folder, then run:

    micromamba activate client
    python node_inference.py

### Inference Server
Run inference server remotely via SSH:

#### 1st terminal:

    ssh SERVER_NAME
    roscore

#### 2nd terminal:

    ssh SERVER_NAME
    micromamba activate server
    python node_start.py

#### 3rd terminal:

    ssh SERVER_NAME
    micromamba activate server
    python node_final.py

### Switch Input from STT to Keyboard Input
Update `config.py - # Input Mode` in the `src_client/` and `src_server/` folder.