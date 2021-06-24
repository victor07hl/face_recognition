import os, uuid
import cv2
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__

connect_str= os.getenv('AZURE_STORAGE_CONNECTION_STRING')
#Creating blobserviceclient
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

#reading img
nameimg = 'gatesn.jpg'
img = cv2.imread('main_user/imgs/train/gates.jpg')

#creating blobclient
container_name = "maindata"
blob_client = blob_service_client.get_blob_client(container = container_name,blob=nameimg)

#uploading img from path----------
#converting img to blob
with open("main_user/imgs/train/gates.jpg", "rb") as image:
    f = image.read()
    b = bytearray(f)
    print (type(b))

#blob_client.upload_blob(b)


#convert cv2 to bytearray
imgb = cv2.imencode('.jpg',img)
print(imgb)
blob_client.upload_blob(b)

#print blobs from container

container = ContainerClient.from_connection_string(connect_str, container_name=container_name)
blob_list = container.list_blobs()
for blob in blob_list:
    print(blob.name + '\n')