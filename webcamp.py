import cv2
import numpy as np
import face_recognition
import os 


path_imgs = 'imgs/train'
names_imgs = []
images=[]
list_img= os.listdir(path_imgs)
print(list_img)
print('entering in for')
for img_name in list_img:
	imgadd = cv2.imread(f'{path_imgs}/{img_name}')
	imgadd = cv2.cvtColor(imgadd, cv2.COLOR_BGR2RGB)
	images.append(imgadd)
	names_imgs.append(os.path.splitext(img_name)[0])
	print(os.path.splitext(img_name)[0])

#Creating encoding functions
def findEncodings(images):
	encode_list = []
	for img in images:
		encode = face_recognition.face_encodings(img)[0]
		encode_list.append(encode)
	return encode_list


encodelistknow = findEncodings(images)
print(f'encoding {len(encodelistknow)} images')