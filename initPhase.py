# from firebase_admin import firestore,storage
# from yoloFace import YOLOFace
# from faceNet.faceNet import FaceNet
# from yoloface import face_analysis
# from utils import FPSmetric
# from engine import Engine
# import threading
# import datetime

# global facenet
# global engine
# global capture
# db = firestore.client()
# bucket = storage.bucket()
# yoloface = face_analysis()
# facenet = FaceNet(
#     detector = YOLOFace(face = yoloface),
#     onnx_model_path = "models/faceNet.onnx", 
#     anchors = "faces/Known",
#     force_cpu = False,
# )
# engine = Engine(webcam_id=1, show=True, custom_objects=[facenet, FPSmetric()])
# def addNewFace():
#     image = engine.return_frame()
#     capture = facenet.detect_save_faces(image, output_dir="faces/Unknown")
#     print(capture)
#     capture_list = list(capture)
#     while not capture_list[1] and capture_list[0]==-1:
#         new_capture = facenet.detect_save_faces(engine.return_frame(), output_dir="faces/Unknown")
#         capture_list[0] = list(new_capture)[0]
#         continue
#     if capture_list[0]!=-1:
#         upload_thread = threading.Thread(target=uploadImage, args=(capture_list[0],))
#         upload_thread.start()
#     # print(capture_list[1])

# def uploadImage(index: str):
#     blob = bucket.blob(f'Unknown_{index}.png')
#     with open(f'faces/Unknown/Unknown_{index}.png','rb') as f:
# 	    blob.upload_from_file(f)      
    
#     link = blob.public_url
    
#     data = {'Name': f'Unknown_{index}',
#             'image_url': link,
#             }
#     doc_ref = db.collection('Unknown').document(f'{index}')
#     doc_ref.set(data)

# def sendLog(name: str, status):
#     doc_ref = db.collection('logs').document(status)
#     doc = doc_ref.get()
#     time_threshhold = datetime.timedelta(minutes=5)
#     if doc.exists:
#         print('Sending Logs')
#         last_entry_time = doc.to_dict()["timestamp"].replace(tzinfo=None)
#         time_diff = datetime.datetime.now() - last_entry_time
#         if time_diff > time_threshhold:
#              doc_ref.set({
#                 "name": name,
#                 "timestamp": firestore.SERVER_TIMESTAMP
#             })
#     else:
#         doc_ref.set({
#                 "name": name,
#                 "timestamp": firestore.SERVER_TIMESTAMP
#             })
#         print('Log send successfully')

from firebase_admin import firestore, storage
from yoloFace import YOLOFace
from faceNet.faceNet import FaceNet
from yoloface import face_analysis
from utils import FPSmetric
from engine import Engine
import threading
import datetime

global facenet
global engine
global capture
db = firestore.client()
bucket = storage.bucket()
yoloface = face_analysis()
facenet = FaceNet(
    detector=YOLOFace(face=yoloface),
    onnx_model_path="models/faceNet.onnx",
    anchors="faces/Known",
    force_cpu=False,
)
engine = Engine(webcam_id=0, show=True, custom_objects=[facenet, FPSmetric()])

previous_entry = {'name': None, 'timestamp': None}
entry_list = []


def addNewFace():
    image = engine.return_frame()
    capture = facenet.detect_save_faces(image, output_dir="faces/Unknown")
    # print(capture)
    # capture_list = list(capture)
    # while not capture:
    #     capture = facenet.detect_save_faces(engine.return_frame(), output_dir="faces/Unknown")
    #     # capture_list[0] = list(new_capture)[0]
    #     continue
    if capture == True:
        upload_thread = threading.Thread(target=uploadImage, args=(facenet.index_counter-1,))
        upload_thread.start()
    # print(capture_list[1])


def uploadImage(index: str):
    blob = bucket.blob(f'Unknown_{index}.png')
    with open(f'faces/Unknown/Unknown_{index}.png', 'rb') as f:
        blob.upload_from_file(f)

    link = blob.public_url

    data = {'Name': f'Unknown_{index}',
            'image_url': link,
            }
    doc_ref = db.collection('Unknown').document(f'{index}')
    doc_ref.set(data)


def sendLog(name: str):
    global previous_entry
    global entry_list
    
    if previous_entry['name'] != name:
        entry_list.append({'name': name, 'timestamp': datetime.datetime.now()})
        previous_entry = {'name': name, 'timestamp': datetime.datetime.now()}
        pushEntryToFirebase(name, datetime.datetime.now())
    else:
        time_threshold = datetime.timedelta(minutes=5)
        time_diff = datetime.datetime.now() - previous_entry['timestamp']
        if time_diff > time_threshold:
            entry_list.append({'name': name, 'timestamp': datetime.datetime.now()})
            previous_entry = {'name': name, 'timestamp': datetime.datetime.now()}
            pushEntryToFirebase(name, datetime.datetime.now())

def pushEntryToFirebase(name: str, timestamp: datetime.datetime):
    doc_ref = db.collection('logs').document()
    doc_ref.set({
        'name': name,
        'timestamp': firestore.SERVER_TIMESTAMP
    })