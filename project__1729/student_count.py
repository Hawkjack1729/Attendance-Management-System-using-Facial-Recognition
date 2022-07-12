import cv2 as cv
import numpy as np
import face_recognition as fr
import os
from tkinter import * 
from tkinter.ttk import *
root = Tk()
height = root.winfo_screenheight()
width = root.winfo_screenwidth()

def change_res(width,height):
    cam.set(3,width)
    cam.set(4,height)
face_cascade = cv.CascadeClassifier('cascade.xml')
root = Tk()
height = root.winfo_screenheight()
width = root.winfo_screenwidth()
cam = cv.VideoCapture(0)
change_res(width, height)
# make_1080p()
while True:
    isTrue,frame=cam.read()
    img_gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img_gray,1.5,4)
    i=0
    for (x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.rectangle(frame,(x,y+10),(x,y),(0,255,0),-1)
        i+=1
        cv.putText(frame,'Student'+ str(i),(x+6,y-6),cv.FONT_HERSHEY_COMPLEX,0.6,(255,255,255),2)
    cv2.namedWindow("Webcam", cv.WINDOW_FULLSCREEN)
    cv.imshow('Webcam',frame)
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
print("Total Student present: ",i)
cam.release()