import os
import shutil
import random
import string
import pandas as pd
import csv 
from google_images_download import google_images_download

parent_dir = "./downloads/" # replace with the actual path to the parent directory

response = google_images_download.googleimagesdownload()
arguments = {"keywords":"manx cat","limit":100,"print_urls":True}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function

def random_string():
    letters = string.ascii_lowercase
    return  ''.join(random.choice(letters) for i in range(15))

def rename_and_update_csv(folder_path):
    csv_file = "extra_cats.csv"
    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg', '.gif','.JPG','.webp')):  # Add more extensions if needed
            old_filepath = os.path.join(folder_path, filename)

            # Generate a new random filename while keeping the extension
            extension = os.path.splitext(filename)[1]
            new_filename = random_string() + extension.lower()
            new_filepath = os.path.join(folder_path, new_filename)

            # Rename the file
            os.rename(old_filepath, new_filepath)

            # Add a new row to the CSV file
            csv_file = os.path.join('', csv_file)
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([new_filename, 5])


#rename_and_update_csv(parent_dir)