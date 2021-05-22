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
	#images.append(cv2.imread(f'{path_imgs}/{img_name}'))
	#names_imgs.append(os.path.splitext(img_name[0]))
	print(os.path.splitext(img_name)[0])