import cv2 as cv
import numpy as np
import face_recognition as fr
import os
from datetime import datetime

path = 'imageatt'
images = []
classname= []
my_list = os.listdir(path)
# print(my_list)

for candidate in my_list:
    current_img = cv.imread(f'{path}/{candidate}')
    images.append(current_img)
    classname.append(os.path.splitext(candidate)[0])

print(classname)

def find_encodings(images):
    encode_list = []
    for img in images:
        img= cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encoding = fr.face_encodings(img)[0]
        encode_list.append(encoding)
    return encode_list
def mark_attendance(name):
    with open('Attendance.csv','r+') as f:
        my_data_list =f.readlines()
        name_list=[]
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            date=now.strftime('%m/%d/%Y').replace('/0','/')
            time = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{date},{time}')
            print("Done,",name)



encode_list_known=find_encodings(images)
print("Encoding Completed,starting webcam.....")

cam =cv.VideoCapture(0)

while True:
    isTrue,img=cam.read()
    img_rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    face_cur_frame = fr.face_locations(img_rgb)
    encoding_cur_frame = fr.face_encodings(img_rgb,face_cur_frame)
    for encode_face,face_loc in zip(encoding_cur_frame,face_cur_frame):
        matches = fr.compare_faces(encode_list_known, encode_face)
        face_dis = fr.face_distance(encode_list_known, encode_face)
        # print(face_dis)
        match_index = np.argmin(face_dis)
        if matches[match_index]>0.15:
            name=classname[match_index].upper()
            mark_attendance(name)
        else:
            name='Unknown'
        # print(name)
        y1,x2,y2,x1 = face_loc
        cv.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv.rectangle(img,(x1,y2-30),(x2,y2),(0,255,0),cv.FILLED)
        cv.putText(img,name,(x1+6,y2-6),cv.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),2)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break


    cv.imshow('webcam',img)
cam.release()
cv.distroyAllWindows()