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

#capturing frames with video cam
cap = cv2.VideoCapture(0)

while True:
	success, img = cap.read()
	imgs = cv2.resize(img,(0,0),None,0.8,0.8)
	#imgs = cv2.cvtColor(imgs,cv2.COLOR_BGR2RGB)

	facescurFrame = face_recognition.face_locations(imgs)
	encodecurFrame = face_recognition.face_encodings(imgs,facescurFrame)


	for encodeFace,faceLoc in zip(encodecurFrame,facescurFrame):
		matches = face_recognition.compare_faces(encodelistknow,encodeFace)
		faceDis = face_recognition.face_distance(encodelistknow,encodeFace)
		print(faceDis,matches)
		matchIndex = np.argmin(faceDis)
		mindis=min(faceDis)

		if matches[matchIndex]:
			name = names_imgs[matchIndex].upper()
			print(name)
		else:
			name = 'unknow'
			print(name)

		cv2.rectangle(imgs,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
		cv2.putText(imgs,f'{name} {round(mindis,2)}',(faceLoc[3],faceLoc[0]),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
		cv2.imshow('current',imgs)
	c  = cv2.waitKey(1)
	if c == 27:
		break



