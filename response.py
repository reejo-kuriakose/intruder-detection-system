import firebase_admin
from firebase_admin import credentials, messaging
# import initPhase

global cred
global database
send_notification = True
cred = credentials.Certificate("/home/reejo/arch/serviceAccountKey.json")
database = firebase_admin.initialize_app(cred,{'storageBucket': 'dyfintro.appspot.com'})
def sendPush(title, msg, registration_token, dataObject=None):
    # See documentation on defining a message payload.
    message = messaging.MulticastMessage(
        notification=messaging.Notification(
            title=title,
            body=msg
        ),
        data=dataObject,
        tokens=registration_token
    )
    response = messaging.send_multicast(message)
    # print('Successfully sent message:', response)


   