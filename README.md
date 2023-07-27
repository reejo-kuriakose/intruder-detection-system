# Intruder Detection System with Face Recognition

## Overview
The Intruder Detection System with Face Recognition is an intelligent system designed to detect and recognize intruders using facial recognition technology. It leverages the power of YOLOface for real-time face detection and FaceNet for accurate face recognition. The system is integrated with Firebase, enabling seamless communication between the system and a mobile app, which receives notifications and captures images of detected intruders.

## Features
- Real-time face detection using YOLOface: The system utilizes the YOLOface algorithm, a variant of YOLO (You Only Look Once), to perform fast and efficient face detection in real-time.
- Face recognition with FaceNet: FaceNet, a deep learning model, is employed to extract high-dimensional features from faces and compare them to a database of known faces for accurate recognition.
- Firebase integration: The system seamlessly connects to Firebase, a powerful cloud-based platform, allowing for efficient data transfer and synchronization between the intruder detection system and the mobile app.
- Mobile app with notifications: The accompanying mobile app receives notifications whenever an intruder is detected. Users can easily monitor the system's activity and take appropriate action.
- Image capture and storage: When an intruder is detected, the system captures an image of the intruder, which is then securely stored in Firebase. This allows users to review the captured images and add the face of the intruder to the known faces if desired.

## Acknowledgements
- YOLOface: https://pypi.org/project/yoloface
- FaceNet: https://github.com/davidsandberg/facenet
