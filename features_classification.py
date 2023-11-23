#import pandas as pd
import sys
import time
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
keras = tf.keras
from multiprocessing import Pool
from PIL import  ImageFile
from keras.applications.mobilenet_v2 import  preprocess_input
from keras.utils import img_to_array,load_img
ImageFile.LOAD_TRUNCATED_IMAGES = True #must keep it, otherwise some images will trigger error when training, as they are truncated


def predict(model_selected, class_names, x, label):
    predictions = model_selected(x)
    prediction_index = np.argmax(predictions[0])
    predicted_class = class_names[prediction_index]
    return predicted_class


def process_prediction(args):
    model_path, class_names, label, x = args
    #uploaded_image_path = "./cat_photos/zgmqniavky.jpg"
    x = load_img(x, target_size=(331, 331))
    # Convert the image to a numpy array
    x = img_to_array(x)
    # Reshape the array to add a dimension for the batch size
    x = np.expand_dims(x, axis=0)
    # Preprocess the input image data
    x = preprocess_input(x)
    model = tf.keras.models.load_model(model_path)
    return predict(model, class_names, x, label)

def main():
    prediction_tasks = [
        ("cat_model_classifier_breed_improved.h5", ['abyssinian', 'american bobtail', 'american bobtail shorthair', 'american curl longhair', 'american curl shorthair', 'american wirehair', 'australian mist', 'balinese', 'bambino', 'bengal', 'bengal longhair', 'birman', 'bombay', 'british longhair', 'british shorthair', 'burmese', 'burmilla', 'burmilla longhair', 'chartreux', 'chausie', 'cornish rex', 'cymric', 'devon rex', 'domestic longhair', 'domestic shorthair', 'egyptian mau', 'exotic shorthair', 'havana brown', 'himalayan', 'japanese bobtail', 'japanese bobtail longhair', 'khao manee', 'korat', 'kurilian bobtail', 'kurilian bobtail longhair', 'laperm', 'laperm shorthair', 'lykoi', 'maine coon', 'manx', 'mekong bobtail', 'minuet', 'minuet longhair', 'munchkin', 'munchkin longhair', 'nebelung', 'norwegian forest', 'ocicat', 'oriental longhair', 'oriental shorthair', 'persian', 'peterbald', 'pixiebob', 'pixiebob longhair', 'ragamuffin', 'ragdoll', 'russian blue', 'savannah', 'scottish fold', 'scottish fold longhair', 'scottish straight', 'scottish straight longhair', 'selkirk rex', 'selkirk rex longhair', 'siamese', 'siberian', 'singapura', 'snowshoe', 'somali', 'sphynx', 'tonkinese', 'toyger', 'turkish angora', 'turkish van'], "breed"),
        ("cat_model_classifier_chonky.h5", ['magro', 'médio', 'gordo', 'muito gordo'], "chonky"),
        ("cat_model_classifier_height.h5", ['pequeno', 'médio', 'grande'], "height"),
        ("cat_model_classifier_fur.h5", ['sem pelo', 'pelo com falha', 'médio', 'pelo longo'], "fur"),
        ("cat_model_classifier_is_kitten.h5", ['não', 'sim'], "is_kitten"),
        ("cat_model_classifier_tail_length.h5", ['sem cauda', 'meia cauda', 'cauda completa'], "tail_length"),
        ("cat_model_classifier_face_smush.h5", ['não', 'sim'], "face_smush"),
        ("cat_model_classifier_tail_fur.h5", ['sem pelo', 'pelo curto', 'pelo longo', 'sem cauda'], "tail_fur"),
        ("cat_model_classifier_eye_colour_right.h5", ['olho faltando', 'azul', 'verde', 'avelã', 'verde esmeralda', 'amarelo', 'marrom', 'laranja'], "eye_colour_right"),
        ("cat_model_classifier_eye_colour_left.h5", ['olho faltando', 'azul', 'verde', 'avelã', 'verde esmeralda', 'amarelo', 'marrom', 'laranja'], "eye_colour_left"),
        ("cat_model_classifier_right_ear_size.h5", ['orelha faltando', 'curvada pra frente', 'curvada pra trás', 'pequena', 'média', 'grande', 'média com pelos', 'grande com pelos'], "right_ear_size"),
        ("cat_model_classifier_left_ear_size.h5", ['orelha faltando', 'curvada pra frente', 'curvada pra trás', 'pequena', 'média', 'grande', 'média com pelos', 'grande com pelos'], "left_ear_size"),
        ("cat_model_classifier_fur_type.h5", ['Solid', 'Colorpoint', 'Tabby', 'Bicolor', 'Tricolor(Calico)', 'Tortoiseshell', 'Ticked Coat'], "fur_type")
    ]

    image_folder = "exception_cats2"
    image_paths = [os.path.join(image_folder, filename) for filename in os.listdir(image_folder) if filename.endswith(('.jpg', '.png', '.jpeg','.PNG','.JPEG','.JPG','.GIF','.gif'))]
    print(image_paths)
    with Pool() as pool:
        results = []        
        # Iterate through each image
        for x in image_paths:
            print("oi")
            image_predictions = {"id": os.path.basename(x)}  # Use the filename as the "id"

            # Create a list of arguments for each model and image combination
            task_args = [(model_path, class_names, label, x) for model_path, class_names, label in prediction_tasks]
            
            # Use the multiprocessing pool to process predictions for the current image
            predictions = pool.map(process_prediction, task_args)
            
            for i, (model_path, _, label, _) in enumerate(task_args):
                image_predictions[label] = predictions[i]
            
            results.append(image_predictions)
            df = pd.DataFrame(results)
            df.to_csv("predictions2.csv", index=False)

       
if __name__ == "__main__":
    main()
