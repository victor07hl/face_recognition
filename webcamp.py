import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime
from encoding import * 
import time

user="main_user"

ti = datetime.now()
names_imgs, encodelistknow = decoding()
print(type(encodelistknow))

tf = datetime.now()
print((tf-ti).seconds,'seconds')

#capturing frames with video cam
cap = cv2.VideoCapture(0)

while True:
	success, img = cap.read()
	imgs = cv2.resize(img,(0,0),None,0.8,0.8)
	#print('-----size:',imgs.shape)

	facescurFrame = face_recognition.face_locations(imgs)
	encodecurFrame = face_recognition.face_encodings(imgs,facescurFrame)
	#print(len(facescurFrame),len(encodecurFrame))


	for encodeFace,faceLoc in zip(encodecurFrame,facescurFrame):
		matches = face_recognition.compare_faces(encodelistknow,encodeFace,0.58)
		faceDis = face_recognition.face_distance(encodelistknow,encodeFace)
		#print(faceDis,matches)
		matchIndex = np.argmin(faceDis)
		mindis=min(faceDis)

		if matches[matchIndex]:
			name = names_imgs[matchIndex].upper()
			colabel = (0,130,0)
			cv2.rectangle(imgs,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),1)
			cv2.putText(imgs,f'{name} {round(mindis,2)}',(faceLoc[3],faceLoc[0]),cv2.FONT_HERSHEY_COMPLEX,1,colabel,1)
			#print(name)
		else:
			#time for give name to unknown img face
			tiun = datetime.now()
			tistr = tiun.strftime("%d%m%Y_%H%M%S")

			#label for unkonwn img
			name = 'unknown'
			pathimg = f'{user}/unknownfaces/{tistr}.jpg' #path for save unknown img
			colabel = (10,15,137) # BGR to red

            #set rectangle location and label 
			cv2.rectangle(imgs,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),1)
			cv2.putText(imgs,f'{name} {round(mindis,2)}',(faceLoc[3],faceLoc[0]),cv2.FONT_HERSHEY_COMPLEX,1,colabel,1)
			
			#write unknown face 
			cv2.imwrite(pathimg,imgs)

			tfun = datetime.now()
			print(f" time saving unknown img {(tfun-tiun).microseconds}")

		
		#display in screen full img with rectangles location and labels
		cv2.imshow('current',imgs)
		
	c  = cv2.waitKey(1)
	if c == 27:
		break



