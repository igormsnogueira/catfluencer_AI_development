import cv2
import numpy as np

# Load YOLOv3 model
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

# Load sample image
img_path = 'thin_cat.jpg'
img = cv2.imread(img_path)

# Prepare input blob for YOLOv3
blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)

# Set input blob for the network
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

# Forward pass through the network
outs = net.forward(output_layers)
# Post-process the detections
conf_threshold = 0.5
nms_threshold = 0.4

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > conf_threshold:
            center_x = int(detection[0] * img.shape[1])
            center_y = int(detection[1] * img.shape[0])
            width = int(detection[2] * img.shape[1])
            height = int(detection[3] * img.shape[0])
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
    cv2.rectangle(img, (left, top), (left+width, top+height), color, 2)
    cv2.putText(img, str(class_ids[i]), (left, top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the output image
cv2.imshow('Output', img)
cv2.waitKey(0)
cv2.destroyAllWindows()