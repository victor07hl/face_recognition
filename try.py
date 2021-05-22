import pandas
import numpy
import cv2
import face_recognitions

print('helloworld')
img = cv2.imread('/home/victor/Pictures/tree_linux.png')
cv2.imshow('image',img)

while True:
	c = cv2.waitKey(1)
	if c == 27:
		cv2.destroyAllWindows()
		break

#import cv2

'''cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()'''
