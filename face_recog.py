import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('imgs/elon_musk.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgtest = face_recognition.load_image_file('imgs/gates.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)


#face locations 
faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]

#drawing rectangles
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

cv2.imshow('elon_musk',imgElon)
cv2.imshow('test',imgtest)

#Results comparison
results = face_recognition.compare_faces([encodeElon],encodetest)
print(results)

while True:
	c = cv2.waitKey(1)
	if c == 27:
		cv2.destroyAllWindows()
		break