import pandas as pd
import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import ImageFile
import warnings
import sys
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True #must keep it, otherwise some images will trigger error when training, as they are truncated

#class_names = ['Abyssinian', 'American Bobtail', 'American Curl', 'American Shorthair', 'American Wirehair', 'Applehead Siamese', 'Balinese', 'Bengal', 'Birman', 'Bombay', 'British Shorthair', 'Burmese', 'Burmilla', 'Calico', 'Chartreux', 'Chausie', 'Chinchilla', 'Cornish Rex', 'Cymric', 'Devon Rex', 'Dilute Calico', 'Dilute Tortoiseshell', 'Domestic Long Hair', 'Domestic Medium Hair', 'Domestic Short Hair', 'Egyptian Mau', 'Exotic Shorthair', 'Extra-Toes Cat - Hemingway Polydactyl', 'Havana', 'Himalayan', 'Japanese Bobtail', 'Javanese', 'Korat', 'LaPerm', 'Maine Coon', 'Manx', 'Munchkin', 'Nebelung', 'Norwegian Forest Cat', 'Ocicat', 'Oriental Long Hair', 'Oriental Short Hair', 'Oriental Tabby', 'Persian', 'Pixiebob', 'Ragamuffin', 'Ragdoll', 'Russian Blue', 'Scottish Fold', 'Selkirk Rex', 'Siamese', 'Siberian', 'Silver', 'Singapura', 'Snowshoe', 'Somali', 'Sphynx - Hairless Cat', 'Tabby', 'Tiger', 'Tonkinese', 'Torbie', 'Tortoiseshell', 'Turkish Angora', 'Turkish Van', 'Tuxedo']
#class_names = ['abyssinian', 'american bobtail', 'american bobtail ', 'american bobtail shorthair', 'american curl longhair', 'american curl shorthair', 'american longhair', 'american shorthair', 'american wirehair', 'australian mist', 'balinese', 'bengal', 'bengal longhair', 'birman', 'bombay', 'british longhair', 'british shorthair', 'burmese', 'burmilla', 'burmilla longhair', 'chartreux', 'chausie', 'cornish rex', 'cymric', 'devon rex', 'domestic longhair', 'domestic shorthair', 'donskoy', 'egyptian mau', 'european shorthair', 'exotic shorthair', 'havana brown', 'himalayan', 'japanese bobtail', 'japanese bobtail longhair', 'khao manee', 'korat', 'kurilian bobtail', 'kurilian bobtail longhair', 'laperm', 'laperm shorthair', 'lykoi', 'maine coon', 'manx', 'mekong bobtail', 'minuet', 'minuet longhair', 'munchkin', 'munchkin longhair', 'nebelung', 'norwegian forest', 'ocicat', 'oriental longhair', 'oriental shorthair', 'persian', 'peterbald', 'pixiebob', 'pixiebob longhair', 'ragamuffin', 'ragdoll', 'russian blue', 'snowshoe', 'savannah', 'scottish fold', 'scottish fold longhair', 'scottish straight', 'scottish straight longhair', 'selkirk rex', 'selkirk rex longhair', 'siamese', 'siberian', 'singapura', 'somali', 'sphynx', 'thai', 'tonkinese', 'toyger', 'turkish angora', 'turkish van']
class_names = ['abyssinian', 'american bobtail', 'american bobtail shorthair', 'american curl longhair', 'american curl shorthair', 'american wirehair', 'australian mist', 'balinese', 'bambino', 'bengal', 'bengal longhair', 'birman', 'bombay', 'british longhair', 'british shorthair', 'burmese', 'burmilla', 'burmilla longhair', 'chartreux', 'chausie', 'cornish rex', 'cymric', 'devon rex', 'domestic longhair', 'domestic shorthair', 'egyptian mau', 'exotic shorthair', 'havana brown', 'himalayan', 'japanese bobtail', 'japanese bobtail longhair', 'khao manee', 'korat', 'kurilian bobtail', 'kurilian bobtail longhair', 'laperm', 'laperm shorthair', 'lykoi', 'maine coon', 'manx', 'mekong bobtail', 'minuet', 'minuet longhair', 'munchkin', 'munchkin longhair', 'nebelung', 'norwegian forest', 'ocicat', 'oriental longhair', 'oriental shorthair', 'persian', 'peterbald', 'pixiebob', 'pixiebob longhair', 'ragamuffin', 'ragdoll', 'russian blue', 'savannah', 'scottish fold', 'scottish fold longhair', 'scottish straight', 'scottish straight longhair', 'selkirk rex', 'selkirk rex longhair', 'siamese', 'siberian', 'singapura', 'snowshoe', 'somali', 'sphynx', 'tonkinese', 'toyger', 'turkish angora', 'turkish van']

#class_names = ['0','1','2','3','4','5','6','7']
print(f"The quantity of element in class_names: {len(class_names)}")
data_dir = './cats_oficial/'
csv_file = 'cats_training data_PHASEIII.csv'
batch_size = 32
number_of_epochs = 25


data_df = pd.read_csv(csv_file, dtype={'chonky': str})
data_df = data_df.dropna(subset=['chonky'])
data_df['chonky'] = data_df['chonky'].astype(str)
filtered_data = data_df[data_df['chonky'].isin(['', '-1', '-2','3']) == False]

# Select only the 'id' and 'chonky' columns
selected_columns = ['id', 'chonky']
filtered_data = filtered_data[selected_columns]
data_df = filtered_data
data_df = data_df.sample(frac=1, random_state=42)

# Create an empty DataFrame to store the result
result_df = pd.DataFrame(columns=selected_columns)

# Iterate over unique 'chonky' values and select 100 rows for each
unique_chonky_values = filtered_data['chonky'].unique()
for chonky_value in unique_chonky_values:
    chonky_subset = filtered_data[filtered_data['chonky'] == chonky_value].head(840)
    result_df = pd.concat([result_df, chonky_subset])

# Shuffle the resulting DataFrame
data_df = result_df.sample(frac=1, random_state=42)



print("number of useful rows: ",data_df.shape[0])
"""

data_df['id'] = data_df['id'].apply(lambda x:str(int(x)) + '.jpg')
data_df['breed'] = data_df['breed'].astype(str)

def get_index(string):
    return class_names.index(string)

data_df['breed_id'] = data_df['breed'].apply(get_index)
data_df['breed_id'] = data_df['breed_id'].astype(str)

shuffled_data = data_df.sample(frac=1).reset_index(drop=True)
shuffled_data.to_csv(csv_file, index=False)
data_df = pd.read_csv(csv_file)

"""
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values between 0 and 1
    validation_split=0.2,  # Split the data into training and validation sets
    rotation_range=0,  # Random rotation between -20 and +20 degrees
    width_shift_range=0.1,  # Randomly shift the width by 0.1
    height_shift_range=0.1,  # Randomly shift the breed by 0.1
    shear_range=0,  # Apply shear transformation with intensity 0.2
    zoom_range=0.2,  # Randomly zoom by 20%
    horizontal_flip=False
)


# Create the data generator for the training set
train_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=data_dir,
    x_col='id',
    y_col='chonky',
    color_mode='rgb',
    target_size=(331,331), # Resize images to 224x224 pixels
    batch_size=batch_size, # Use batches of 32 images
    class_mode='categorical',
    subset='training', # Use the training subset of the data
    shuffle=True,
    seed=42
)

# Create the data generator for the validation set
val_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=data_dir,
    x_col='id',
    y_col='chonky',
    color_mode='rgb',
    target_size=(331,331),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation', # Use the validation subset of the data
    shuffle=True,
    seed=42
)

a = train_generator.class_indices
class_names = list(a.keys())  # storing class/breed names in a list
print(class_names)

base_model = tf.keras.applications.InceptionResNetV2(
    include_top=False,
    weights='imagenet',
    input_shape=(331,331,3)
)
base_model.trainable = False #freezing the layers of the base model, as the weights and biases of it is already trained a lot, we just want to train the layers we add after, with our training data


model = tf.keras.Sequential([
    base_model,  
    tf.keras.layers.BatchNormalization(renorm=True),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

early = tf.keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=val_generator.n//val_generator.batch_size

history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    epochs=number_of_epochs,
    validation_steps=STEP_SIZE_VALID,
    validation_data=val_generator,
    callbacks=[early]
)

test_loss, test_accuracy = model.evaluate(val_generator,steps=STEP_SIZE_VALID)
print("Test accuracy:", test_accuracy)

model.save("cat_model_classifier_chonky_test.h5")
