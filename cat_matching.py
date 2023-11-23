import numpy as np
import cv2
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.preprocessing import image
kera = tf.keras
from keras.utils import img_to_array,load_img
# Load pre-trained VGG16 model
base_model = tf.keras.applications.VGG16(weights='imagenet')
# Remove the last fully connected layer
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Load all 100 cat drawings and extract their features
drawings_features = []
for i in range(1,6):
    img_path = f"testing_images/demoCat_00{i}.png"
    img = load_img(img_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model.predict(x)
    drawings_features.append(features)

# Load the user-provided cat image and extract its features
user_img_path = "testing_images/white_cat.jpg"
user_img = load_img(user_img_path, target_size=(224, 224))
x = img_to_array(user_img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
user_features = model.predict(x)

# Compare the user-provided cat image with each drawing and find the most similar one
min_distance = float('inf')
most_similar_drawing_idx = None
for i in range(1,6):
    distance = np.linalg.norm(drawings_features[i-1] - user_features)
    print(drawings_features[i-1])
    print (distance)
    if distance < min_distance:
        min_distance = distance
        most_similar_drawing_idx = i

# Load and display the most similar drawing
most_similar_drawing_path = f"testing_images/demoCat_00{most_similar_drawing_idx}.png"
most_similar_drawing = cv2.imread(most_similar_drawing_path)
cv2.imshow("Most similar drawing", most_similar_drawing)
cv2.waitKey(0)