import cv2
import stow
import typing
import numpy as np
import onnxruntime as ort
import response as rsp
import threading
import initPhase
import engine

class FaceNet:
    """FaceNet class object, which can be used for simplified face recognition
    """
    def __init__(
        self, 
        detector: object,
        onnx_model_path: str = "models/faceNet.onnx", 
        anchors: typing.Union[str, dict] = 'faces/Known',
        unknownanchors: typing.Union[str, dict] = 'faces/Unknown',
        force_cpu: bool = False,
        threshold: float = 0.45,
        color: tuple = (255, 255, 255),
        thickness: int = 2,
        ) -> None:
        """Object for face recognition
        Params:
            detector: (object) - detector object to detect faces in image
            onnx_model_path: (str) - path to onnx model
            force_cpu: (bool) - if True, onnx model will be run on CPU
            anchors: (str or dict) - path to directory with faces or dictionary with anchor names as keys and anchor encodings as values
            threshold: (float) - threshold for face recognition
            color: (tuple) - color of bounding box and text
            thickness: (int) - thickness of bounding box and text
        """
        if not stow.exists(onnx_model_path):
            raise Exception(f"Model doesn't exists in {onnx_model_path}")

        self.detector = detector
        self.threshold = threshold
        self.color = color
        self.thickness = thickness

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        providers = providers if ort.get_device() == "GPU" and not force_cpu else providers[::-1]

        self.ort_sess = ort.InferenceSession(onnx_model_path, providers=providers)

        self.input_shape = self.ort_sess._inputs_meta[0].shape[1:3]
        
        self.anchors = self.load_anchors(anchors) if isinstance(anchors, str) else anchors
        if len(stow.ls('faces/Unknown'))!=0:
            self.unknownanchors = self.load_anchors(unknownanchors) if isinstance(unknownanchors, str) else unknownanchors
            # print(unknownanchors)
        # else:
        #     self.unknownanchors ={}
        self.index_counter = 0

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """Normalize imaunknownanchorsge

        Args:
            img: (np.ndarray) - image to be normalized

        Returns:
            img: (np.ndarray) - normalized image
        """
        mean, std = img.mean(), img.std()
        return (img - mean) / std

    def l2_normalize(self, x: np.ndarray, axis: int = -1, epsilon: float = 1e-10) -> np.ndarray:
        """l2 normalization function

        Args:
            x: (np.ndarray) - input array
            axis: (int) - axis to normalize
            epsilon: (float) - epsilon to avoid division by zero

        Returns:
            x: (np.ndarray) - normalized array
        """
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
        return output

    def detect_save_faces(self, image: np.ndarray, output_dir: str = "faces/Unknown"):
        """Detect faces in given image and save them to output_dir

        Args:
            image: (np.ndarray) - image navato be processed
            output_dir: (str) - directory where faces will be saved

        Returns:
            bool: (bool) - True if faces were detected and saved
        """
        face_crops = [image[t:b, l:r] for t, l, b, r in self.detector(image, return_tlbr=True)]
        # print(np.asarray(face_crops))
        # face_encode = self.encode(np.asarray(face_crops))
        # print(face_encode[0])

        if face_crops == []: 
            return {False,-1}
        # print(face_crops)
        face_unknown_distance = 0
        face_known_distance = 0
        stow.mkdir(output_dir)
        # if len(stow.ls('faces/Unknown')) != 0:
        #     self.unknownanchors = self.load_anchors('faces/Unknown')
        #     if len(list(self.anchors.values())) > 0:
        #         face_unknown_distance = self.cosine_distance(face_encode,list(self.unknownanchors.values()))
        #         # print(face_unknown_distance)
        #         face_known_distance = self.cosine_distance(face_encode,list(self.anchors.values()))
        #         # print(face_known_distance)
        for index,crop in enumerate(face_crops):
            face_encode = self.encode(face_crops[index])
            if len(stow.ls('faces/Unknown')) != 0:
                self.anchors = self.load_anchors('faces/Known')
                self.unknownanchors = self.load_anchors('faces/Unknown')
                # print("Known anchor:",self.anchors)
                # print("Unknown Anchor:",self.unknownanchors)
                if len(list(self.unknownanchors.values())) > 0:
                    face_unknown_distance = self.cosine_distance(face_encode,list(self.unknownanchors.values()))
                    print("unknown:",np.max(face_unknown_distance))
                    face_known_distance = self.cosine_distance(face_encode,list(self.anchors.values()))
                    print("known:",np.max(face_known_distance))
                    # print(np.max(face_known_distance))
            if np.max(face_unknown_distance) < self.threshold and np.max(face_known_distance) < self.threshold:
                output_path = stow.join(output_dir, f"Unknown_{str(self.index_counter)}.png")
                cv2.imwrite(output_path, crop)
                print("Crop saved to:", output_path)
                self.index_counter+=1
            # if len(stow.ls('faces/Unknown'))!=0:
            #     self.unknownanchors = self.load_anchors('output_dir')
                # print(type(self.unknownanchors))
            print(self.index_counter)
        return True

    def load_anchors(self, faces_path: str):
        """Generate anchors for given faces path

        Args:
            faces_path: (str) - path to directory with faces

        Returns:
            anchors: (dict) - dictionary with anchor names as keys and anchor encodings as values
        """
        anchors = {}
        if not stow.exists(faces_path):
            return {}
        # print(faces_path)
        # print (stow.ls(faces_path))
        for face_path in stow.ls(faces_path):
            anchors[stow.basename(face_path)] = self.encode(cv2.imread(face_path.path))
        return anchors

    def encode(self, face_image: np.ndarray) -> np.ndarray:
        """Encode face image with FaceNet model

        Args 
            face_image: (np.ndarray) - face image to be encoded
            
        Returns:
            face_encoding: (np.ndarray) - face encoding
        """
        
        face = self.normalize(face_image)
        if face.size == 0:
            return []
        # print(face)
        face = cv2.resize(face, self.input_shape).astype(np.float32)

        encode = self.ort_sess.run(None, {self.ort_sess._inputs_meta[0].name: np.expand_dims(face, axis=0)})[0][0]
        normalized_encode = self.l2_normalize(encode)

        return normalized_encode

    def cosine_distance(self, a: np.ndarray, b: typing.Union[np.ndarray, list]) -> np.ndarray:
        """Cosine distance between wectors a and b

        Args:
            a: (np.ndarray) - first vector
            b: (np.ndarray) - second list of vectors

        Returns:
            distance: (float) - cosine distance
        """
        if len(a) == 0 or len(b) == 0:
            return 0.0
        
        if isinstance(a, list):
            a = np.array(a)

        if isinstance(b, list):
            b = np.array(b)

        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    def draw(self, image: np.ndarray, face_crops: dict):
        """Draw face crops on image

        Args:
            image: (np.ndarray) - image to be drawn on
            face_crops: (dict) - dictionary with face crops as values and face names as keys

        Returns:
            image: (np.ndarray) - image with drawn face crops
        """
        for value in face_crops.values():
            t, l, b, r = value["tlbr"]
            cv2.rectangle(image, (l, t), (r, b), self.color, self.thickness)
            cv2.putText(image, stow.name(value['name']), (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.color, self.thickness)
            # print(stow.name(value['name']))
        return image

    def __call__(self, frame: np.ndarray, flag: int = 1) -> np.ndarray:
        """Face recognition pipeline

        Args:
            frame: (np.ndarray) - image to be processed

        Returns:
            frame: (np.ndarray) - image with drawn face recognition results
        """
        tokens = ["fzvy4MEVQjK5O-vCNDuKoO:APA91bH1wBibwOSNETEde4o0LUc9d1crOlr0QLP2pIIV3XLr6W4megproTXZugZlN2ov-AE-C010K-ccbirt-upwf9QaRsV8OkpYLjJ9cxYP-uYLKHVprfcsL4nXFYyipvUKUH6G97hB"]
        face_crops = {index: {"name": "Intruder", "tlbr": tlbr} for index, tlbr in enumerate(self.detector(frame, return_tlbr=True))}
        for key, value in face_crops.items():
            t, l, b, r = value["tlbr"]
            face_encoding = self.encode(frame[t:b, l:r])
            if len(face_encoding) > 0 and len(list(self.anchors.values())) > 0:
                distances = self.cosine_distance(face_encoding, list(self.anchors.values()))
                # print(self.anchors)
                unknown_distances = 0
                if len(stow.ls('faces/Unknown')) != 0:
                    self.unknownanchors = self.load_anchors('faces/Unknown')
                    unknown_distances = self.cosine_distance(face_encoding,list(self.unknownanchors.values()))
                    # print(self.unknownanchors)
                # distances = self.cosine_distance(face_encoding, list(self.anchors.values()))
                print("Known:",distances)
                print("UnKnown:", unknown_distances)
                if np.max(distances) > self.threshold:
                    face_crops[key]["name"] = list(self.anchors.keys())[np.argmax(distances)]
                    log_thread = threading.Thread(target=initPhase.sendLog, args=(face_crops[key]['name'].replace(".png", ""),))
                    log_thread.start()
                    # rsp.send_notification = True
                elif np.max(distances) < self.threshold:
                    if np.max(unknown_distances) < self.threshold:
                        notification_thread = threading.Thread(target=rsp.sendPush, args=("Intruder Detected", "Intruder Detected has been detected", tokens))
                        log_thread = threading.Thread(target=initPhase.sendLog, args=("Unknown",))
                        # print(unknown_distances)
                        capture = initPhase.captureFace()
                        initPhase.addNewFace(capture)
                        notification_thread.start()
                        log_thread.start()
                        # rsp.send_notification = False
                # print(distances)
 
        frame = self.draw(frame, face_crops)

        return frame