#!/usr/bin/env python3
"""
Camera Test
"""

import cv2
import sys

def test_camera():
    print("Start Camera Test...")
    
    # initialize camera (default to camera index 0)
    cap = cv2.VideoCapture(2)
    
    if not cap.isOpened():
        print("❌ Unable to open camera.")
        print("Trying other camera indices...")
        
        # Try other camera indices (1, 2, 3)
        for i in range(1, 4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✅ Camera {i} opened successfully.")
                break
        else:
            print("❌ Unable to find a working camera.")
            return False
    else:
        print("✅ Camera 0 opened successfully.")

    # Check camera resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Camera Information:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")

    # Capture a few images for testing
    print("\nImage capture test in progress...")

    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame {i+1}/5 captured successfully (size: {frame.shape})")
        else:
            print(f"❌ Frame {i+1}/5 capture failed")
            cap.release()
            return False

    # # Image save test
    # ret, frame = cap.read()
    # if ret:
    #     cv2.imwrite('/home/smtamh/dyros/vllm/test_image.jpg', frame)
    #     print("✅ Test image saved as test_image.jpg")

    # Real-time preview (optional)
    print("\nStarting real-time preview. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read failed")
            break
            
        cv2.imshow('Camera Test', frame)

        # press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera test completed!")
    return True

def list_cameras():
    """Check for available cameras"""
    print("Searching for available cameras...")
    
    available_cameras = []
    for i in range(10): 
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    
    if available_cameras:
        print(f"✅ Available cameras: {available_cameras}")
    else:
        print("❌ No available cameras found.")

    return available_cameras

if __name__ == "__main__":
    print("=== Camera Test Program ===\n")

    # Check for available cameras
    cameras = list_cameras()
    
    if cameras:
        print(f"\nTesting with default camera (#{cameras[0]}).")
        test_camera()
    else:
        print("\nPlease check if a camera is connected.")
        sys.exit(1)