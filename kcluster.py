
import numpy as np
import cv2

# Load the pre-trained Haar Cascade classifiers for cat features
cat_head_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
cat_legs_cascade = cv2.CascadeClassifier('haarcascade_catlegs.xml')
cat_tail_cascade = cv2.CascadeClassifier('haarcascade_cattail.xml')

# Load the image and convert it to grayscale
image = cv2.imread('./input_data/egyptian_mau.png') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


heads = cat_head_cascade.detectMultiScale(gray, 1.3, 5)
legs = cat_legs_cascade.detectMultiScale(gray, 1.3, 5)
tails = cat_tail_cascade.detectMultiScale(gray, 1.3, 5)

# Draw rectangles around the detected features
for (x,y,w,h) in heads:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
for (x,y,w,h) in legs:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
for (x,y,w,h) in tails:
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)

# Display the result
cv2.imshow('Cat Features', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
