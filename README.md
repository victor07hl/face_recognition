# face_recognition

Hi, I'm @victor07hl
I'm interested in artificial intelligence and Machine Learning 
I'm currently learning how deploy face recognition aplication in python

Project Description

This project is about facial recognition. The main idea is check if the person in front
of camera is previusly registered, if not the program save the unknow person image.
On the master branch are the files.py for save unkonwn imgs on local machine.
The other branches are created for each cloud either azure, aws, gcp.

Libreries:
face_recognition : $pip install face-recognition
opencv : $pip install opencv-python
numpy : $pip install numpy
pandas : $pip install pandas

Also you can install this libraries and run the files on your Anaconda enviroment

Folder project
 subfolder descriptions:
    -'main_user' : Contain 2 subfolders and 1 file named encode_imgs
    -'main_user/imgs/train' : This folder has all imgs for user that you want
                             to register. The img name must be the name of user.
    -'main_user/unknonwnfaces' : This folder has all imgs for unknown persons. The 
                                program put this imgs automatically on this folder
 
 files description:
    -'main_user/encode_imgs': this csv file contain the information about encode of each
                              img on 'main_user/imgs/train/' folder
    -'encoding.py': This script contain 2 functions encoding() for save encoding 
                    of imgs on 'main_user/encode_imgs' csv file; and decoding() for 
                    get encoding of imgs in a list with users name and compare with
                    current img on webcamp.py file.
    -'webcamp.py' : On this script is all for program run. This start webcam and take
                    pictures for check user.
    -'face_recog.py' : This file is for probe face_recognition library
                         

