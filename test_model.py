import pandas as pd
import os
import numpy as np
import tensorflow as tf
import sys
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageFile
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.utils import img_to_array,load_img
ImageFile.LOAD_TRUNCATED_IMAGES = True #must keep it, otherwise some images will trigger error when training, as they are truncated


class_names = ['abyssinian', 'american bobtail', 'american bobtail ', 'american bobtail shorthair', 'american curl longhair', 'american curl shorthair', 'american wirehair', 'australian mist', 'balinese', 'bengal', 'bengal longhair', 'birman', 'bombay', 'british longhair', 'british shorthair', 'burmese', 'burmilla', 'burmilla longhair', 'chartreux', 'chausie', 'cornish rex', 'cymric', 'devon rex', 'domestic longhair', 'domestic shorthair', 'egyptian mau', 'exotic shorthair', 'havana brown', 'himalayan', 'japanese bobtail', 'japanese bobtail longhair', 'khao manee', 'korat', 'kurilian bobtail', 'kurilian bobtail longhair', 'laperm', 'laperm shorthair', 'lykoi', 'maine coon', 'manx', 'mekong bobtail', 'minuet', 'minuet longhair', 'munchkin', 'munchkin longhair', 'nebelung', 'norwegian forest', 'ocicat', 'oriental longhair', 'oriental shorthair', 'persian', 'peterbald', 'pixiebob', 'pixiebob longhair', 'ragamuffin', 'ragdoll', 'russian blue', 'snowshoe', 'savannah', 'scottish fold', 'scottish fold longhair', 'scottish straight', 'scottish straight longhair', 'selkirk rex', 'selkirk rex longhair', 'siamese', 'siberian', 'singapura', 'somali', 'sphynx', 'tonkinese', 'toyger', 'turkish angora', 'turkish van']
#class_names = ['0','1','2','3','-1']

#USING THE MODEL TO PREDICT
model = tf.keras.models.load_model('cat_model_classifier_breed_improved.h5')
'''
uploaded_image_path = "./thin_cat.jpg"
# Open the image using Pillow
#image = Image.open(image_path)
#used for v1 and v3
#uploaded_image = load_img(uploaded_image_path, target_size=(224, 224))
#used for v4
uploaded_image = load_img(uploaded_image_path, target_size=(331, 331))
# Convert the image to a numpy array
x = img_to_array(uploaded_image)

# Reshape the array to add a dimension for the batch size
x = np.expand_dims(x, axis=0)

# Preprocess the input image data
x = preprocess_input(x)
predictions = model.predict(x,verbose=0)
prediction_index = np.argmax(predictions[0]) #use numpy function called argmax, to get the index of the higher value from an array, in this case get the label with higher probability
output_str = f"The breed predicted for the uploaded cat is :<b> {class_names[prediction_index]} </b>\n\n" #printing the predicted label
output_str += f"Probabilities:\n"
class_probabilities = {}
print(output_str)
'''
def classify_image(image_path):
    img = load_img(image_path, target_size=(331, 331))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    prediction = model.predict(img,verbose=0)
    prediction_index = np.argmax(prediction[0])
    return class_names[prediction_index]  # Return the predicted class (breed)

# Directory where the images are located
image_directory = './raca_gatos'

# Load the CSV file containing image information
csv_file = 'test10.csv'
df = pd.read_csv(csv_file)

# Iterate through each row in the CSV file and classify the images
for index, row in df.iterrows():
    image_id = row['id']  # Assuming 'id' is the column containing image names with extensions
    image_path = os.path.join(image_directory, image_id)
    
    try:
        breed = classify_image(image_path)
        df.at[index, 'breed'] = breed
        print(f"Classified {image_id} as {breed}")
    except Exception as e:
        print(f"Error classifying {image_id}: {str(e)}")

# Save the updated DataFrame back to the CSV file
df.to_csv(csv_file, index=False)