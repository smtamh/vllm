## Installation
### System Packages
    sudo apt install libportaudio2 libportaudiocpp0 portaudio19-dev ffmpeg espeak-ng v4l-utils

### Python Packages
Install packages in `python 3.8` virtual environment in `Ubuntu 20.04.6 LTS`:
    
    uv pip install rospkg opencv-python vosk sounddevice

<br>

## SSH Setting
Install system package for SSH server:

    sudo apt install openssh-server

You can check your SSH server status:

    sudo systemctl status ssh

Then, enable and allow SSH server.

    sudo systemctl start ssh
    sudo systemctl enable ssh
    sudo ufw allow ssh

<br>

## Camera Check
You can check cameras using `camera_test.py` in your virtual environment.

    python camera_test.py

Or, simply run a few Linux commands.

    v4l2-ctl --list-devices             # list devices and their /dev/video paths
    ffplay VIDEO_PATH                   # preview video (e.g., ffplay /dev/video0)

<br>

## Inference
Check `CLIENT.md` in the `src_client/` folder.