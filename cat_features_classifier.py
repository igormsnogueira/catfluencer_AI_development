import pandas as pd
import tensorflow as tf
keras = tf.keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from PIL import ImageFile
import warnings
warnings.filterwarnings('ignore')
ImageFile.LOAD_TRUNCATED_IMAGES = True #must keep it, otherwise some images will trigger error when training, as they are truncated



data_dir = './images/'
csv_file = 'cats.csv'
batch_size = 32
number_of_epochs = 25


data_df = pd.read_csv(csv_file)
data_df.dropna(subset=['age', 'coat'], inplace=True)
data_df = data_df[~data_df['age'].str.strip().eq('')]
data_df = data_df[~data_df['coat'].str.strip().eq('')]



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
    rescale=1./255, # Normalize pixel values between 0 and 1
    validation_split=0.2, # Split the data into training and validation sets
    horizontal_flip = True
)


# Create the data generator for the training set
train_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=data_dir,
    x_col='id',
    y_col=['age','coat'],
    color_mode='rgb',
    target_size=(331,331), # Resize images to 224x224 pixels
    batch_size=batch_size, # Use batches of 32 images
    class_mode='multi_output',
    subset='training', # Use the training subset of the data
    shuffle=True,
    seed=42
)

# Create the data generator for the validation set
val_generator = datagen.flow_from_dataframe(
    dataframe=data_df,
    directory=data_dir,
    x_col='id',
    y_col=['age','coat'],
    color_mode='rgb',
    target_size=(331,331),
    batch_size=batch_size,
    class_mode='multi_output',
    subset='validation', # Use the validation subset of the data
    shuffle=True,
    seed=42
)




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
    tf.keras.layers.Dense(2, activation='sigmoid')
])

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])

early = tf.keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True
)

STEP_SIZE_TRAIN=train_generator.samples
STEP_SIZE_VALID=val_generator.samples

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

model.save("cat_features_classifier.h5")