from yoloface import face_analysis
import yoloface
import numpy as np
import typing

class YOLOFace:
    def __init__(self,
        face: yoloface.face_analysis):
         self.face = face

    def tlbr(self, frame: np.ndarray, box: typing.List, return_tlbr: bool = False) -> np.ndarray:
        detections = []
        for detection in box:
            x, y, w, h = detection
            if h <= w*2.1:
                detections.append([y, x, y + w, x + h])
            # print(detections)
        return np.array(detections)

    def __call__(self, frame: np.ndarray, return_tlbr: bool = False) -> np.ndarray:
            # face=face_analysis() 
            _,box,conf=self.face.face_detection(frame_arr=frame,frame_status=False,model='tiny')
            # output_frame=face.show_output(frame,box,frame_status=True)
            if return_tlbr:
                if conf:
                    # print(self.tlbr(frame, box))
                    return self.tlbr(frame, box)
                return []