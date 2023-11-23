import cv2
import numpy as np
#from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.resnet import ResNet50, preprocess_input
from keras.utils import img_to_array,load_img
# Load the pre-trained VGG16 model
#vgg16_model = VGG16(weights='imagenet', include_top=False)
resnet_model = ResNet50(weights='imagenet', include_top=False)

# Load the input image
img_path = 'siamese_cat2.jpg'
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
    cat_img = resized_img[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
    cv2.imshow('Cropped Cat Image', cat_img)
    cv2.waitKey(0)

    x = img_to_array(resized_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #cat_features = vgg16_model.predict(x)
    cat_features = resnet_model.predict(x)
    # Calculate the similarity between the cat image and each cartoon
    cartoon_features = []
    for i in range(0,5):
        cartoon_img_path = "demoCat_00" + str(i+1) + '.png'
        cartoon_img = load_img(cartoon_img_path, target_size=img_size)
        x = img_to_array(cartoon_img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        cartoon_features.append(resnet_model.predict(x))

    # Compute the cosine similarity between the cat and each cartoon
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = []
    for cartoon_feature in cartoon_features:
        similarity = cosine_similarity(cat_features.reshape(1,-1), cartoon_feature.reshape(1,-1))[0][0]
        similarities.append(similarity)
    print(similarities)
    # Choose the cartoon that has the highest similarity with the cat
    max_similarity_index = similarities.index(max(similarities))
    chosen_cartoon_path = "demoCat_00" + str(max_similarity_index+1) + '.png'
    chosen_cartoon = cv2.imread(chosen_cartoon_path)
    cv2.imshow('Chosen Cartoon', chosen_cartoon)
    cv2.waitKey(0)
