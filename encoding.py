import cv2
import numpy as np
import face_recognition
import os 
from datetime import datetime
import pandas as pd

def encoding(path_imgs = 'imgs/train',name_outcsv = 'encode_imgs', user = 'main_user' ):
	ti = datetime.now()
	names_imgs = []
	img_encods = []
	list_img= os.listdir(path_imgs)
	#print(list_img)
	for img_name in list_img:
		img_read = cv2.imread(f'{path_imgs}/{img_name}')
		img_encod = list(face_recognition.face_encodings(img_read)[0])
		img_encods.append(img_encod)
		names_imgs.append(os.path.splitext(img_name)[0])


	data = list(zip(names_imgs,img_encods))
	df_encod = pd.DataFrame(data, columns = ['name','encoding'],index = names_imgs)
	try:
		os.mkdir(user) #this line will be removed when use cloud
	except FileExistsError:
		print(f'folder {user} already created')
	df_encod.to_csv(user+'/'+name_outcsv,sep=',',header=True)
	tf = datetime.now()
	print((tf-ti).microseconds)
	#print(df_encod)
	return df_encod



def decoding(user = 'main_user'):
	df = pd.read_csv(user+'/encode_imgs', index_col = 'Unnamed: 0')
	names = list(df['name'])
	encodeListKnow = []
	for name in names:
		encod = df.loc[name]['encoding']
		encodfloat = [float(cod) for cod in encod[1:len(encod)-1].split(', ')]
		encodeListKnow.append(encodfloat)

	print(f"users registered : {', '.join(names)}")
	return names,encodeListKnow


if __name__ == '__main__':
	name,encod=decoding()
	print(encod[0][0])


