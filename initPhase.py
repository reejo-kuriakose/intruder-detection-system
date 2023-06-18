from firebase_admin import firestore, storage
from yoloFace import YOLOFace
from faceNet.faceNet import FaceNet
from yoloface import face_analysis
from utils import FPSmetric
from engine import Engine
import threading
import datetime
import requests
import os
import stow

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
lock = threading.Lock()


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
    lock.acquire()
    try:
        blob = bucket.blob(f'Unknown_{index}.png')
        with open(f'faces/Unknown/Unknown_{index}.png', 'rb') as f:
            blob.upload_from_file(f, content_type='image/png')

        # link = blob.public_url
        firebase_bucket_name = bucket.name
        firebase_object_name = blob.name
        link = f'https://firebasestorage.googleapis.com/v0/b/{firebase_bucket_name}/o/{firebase_object_name}?alt=media'

        data = {'Name': f'Unknown_{index}',
                'image_url': link,
                'timestamp': firestore.SERVER_TIMESTAMP
                }
        doc_ref = db.collection('Unknown').document(f'{index}')
        doc_ref.set(data)
    finally:
        lock.release()


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

# Function to download file from Cloud Storage
def download_file(file_url, file_name):
    response = requests.get(file_url)
    with open(f'faces/Known/{file_name}', 'wb') as file:
        file.write(response.content)

# Function to handle new document added event
def handle_document_added(snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'ADDED':
            # Get the document data
            document_data = change.document.to_dict()
            
            # Extract the name and image URL
            name = document_data['Name']
            image_url = document_data['image_url']
            print(image_url)
            
            # Download the file from Cloud Storage
            file_name = f'{name}.png'
            download_file(image_url, file_name)
            
            print(f'File downloaded: {file_name}')
            
            # Break the loop after processing the first added entry
            break
    if len(stow.ls('faces/Known')) != 0:
        facenet.anchors = facenet.load_anchors('faces/Known')

def handle_document_removed(snapshot, changes, read_time):
    for change in changes:
        if change.type.name == 'REMOVED':
            # Get the document data
            document_data = change.document.to_dict()
            
            # Extract the name
            name = document_data['Name']
            
            # Delete the file from the "faces/Unknown" folder
            file_name = f'{name}.png'
            file_path = f'faces/Unknown/{file_name}'
            
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f'File deleted: {file_name}')
            else:
                print(f'File not found: {file_name}')

    # if len(stow.ls('faces/Unknown')) != 0:
    #     facenet.unknownanchors = facenet.load_anchors('faces/Unknown')
    # else:
    #     facenet.unknownanchors = {}

trigger_ref = db.collection('Known')
trigger = trigger_ref.on_snapshot(handle_document_added)

unknown_trigger_ref = db.collection('Unknown')
unknown_trigger = unknown_trigger_ref.on_snapshot(handle_document_removed)

