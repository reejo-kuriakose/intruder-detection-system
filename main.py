# main.py
from faceDetection import MPFaceDetection
from initPhase import facenet,engine

if __name__ == '__main__':
    # save first face crop as anchor, otherwise don't use
    # while not facenet.detect_save_faces(engine.process_webcam(return_frame=True), output_dir="faces/Known"):
    #     continue

    engine.run()
