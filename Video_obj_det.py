import cv2 as cv
import matplotlib.pyplot as plt


# Loading the model 

configFile = r'C:\Users\LAB1\Desktop\Object\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozenModel = r'C:\Users\LAB1\Desktop\Object\frozen_inference_graph.pb'
model = cv.dnn_DetectionModel(frozenModel,configFile)

# Creating the class file

classFile = r'C:\Users\LAB1\Desktop\Object\coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames, len(classNames)) #classNames ==> Contains all the 80 classes from the coco dataset

#  Setting up the configuration of the model

model.setInputSize(320,320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

# # For an image

img = cv.imread(r'C:\Users\LAB1\Desktop\Object\main.jpg')
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(rgb);


ClassIndex, confidence, bbox = model.detect(rgb,confThreshold=0.5)

font_scale = 1.1
font = cv.FONT_HERSHEY_COMPLEX
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv.rectangle(img,boxes,(255,0,0),2)
    cv.putText(img,classNames[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color = (0,255,0),thickness = 3)
cv.imshow('output',img);
cv.waitKey(0)

