import colorthief
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from keras.applications.resnet import ResNet50, preprocess_input
from keras.utils import img_to_array,load_img
# Load the pre-trained VGG16 model
resnet_model = ResNet50(weights='imagenet', include_top=False)

# Load the input image
img_path = 'input_data/colin1.jpg'
img = cv2.imread(img_path)

# Resize the image to a fixed size
img_size = (224, 224)
resized_img = cv2.resize(img, img_size)

# Preprocess the image for VGG16 input
x = img_to_array(resized_img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Load the pre-trained YOLOv3 model for object detection
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
    
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Get the bounding boxes of the cat in the image using YOLOv3
blob = cv2.dnn.blobFromImage(resized_img, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            # class_id 16 is for 'cat' in YOLOv3
            center_x = int(detection[0] * img_size[0])
            center_y = int(detection[1] * img_size[1])
            width = int(detection[2] * img_size[0])
            height = int(detection[3] * img_size[1])
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])

# Apply Non-Maximum Suppression to remove duplicate detections
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

# Draw the final detections on the image
colors = np.random.uniform(0, 255, size=(len(class_ids), 3))
for i in indices:
    box = boxes[i]
    left = box[0]
    top = box[1]
    width = box[2]
    height = box[3]
    color = colors[i]
    cv2.rectangle(resized_img, (left, top), (left+width, top+height), color, 2)
    cv2.putText(resized_img, str(class_ids[i]), (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if len(boxes) > 0:
 for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cat_img = resized_img[top:top+height, left:left+width]
        cv2.imshow("extracted cat",cat_img)
        cv2.waitKey(0)
        cv2.imwrite('extracted.png', cat_img)


#EXTRACTING 3 MAIN COLORS
ct = colorthief.ColorThief("extracted.png")
dominant_color = ct.get_color(quality=1)
print(dominant_color)

#plt.imshow([[dominant_color]])
#plt.show()

pallete = ct.get_palette(color_count=3)
plt.imshow([[pallete[i] for i in range(3)]])
plt.show()

os.remove("extracted.png")
