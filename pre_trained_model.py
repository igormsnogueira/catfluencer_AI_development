import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar() #hide the progressbar that shows on console with the importing information

#we will load a dataset containing images of dogs and cats, to try to classify them to whether they are a dog or a cat
# split the data manually into 80% training, 10% testing(from 80% to 90%), 10% validation( from 90% to 100%)
(raw_train, raw_validation, raw_test), metadata = tfds.load( #loading a dataset from tensorflow datasets
    'cats_vs_dogs', #name of the dataset
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'], #spliting the data
    with_info=True, 
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that gets the integer value associated to a label and converts it to its string/categorical representation
# display 2 images from the dataset
for image, label in raw_train.take(5): #taking first five images from the train dataset and displaying it , with its correspondent label
  plt.figure()
  plt.imshow(image)
  plt.title(get_label_name(label))

plt.show()

#data pre-processing
IMG_SIZE = 160 # All images will be resized to 160x160 as they have different sizes. Always better to convert the larger ones to a smaller size

def format_example(image, label): # creating a function to get an image and its label and returns an image that is reshaped to IMG_SIZE
  image = tf.cast(image, tf.float32) #convert all pixels of the image to be a float number, in case we have pixels that are integers on the images
  image = (image/127.5) - 1 #max pixel value is 255, when we divide the pixel size by 127.5 the highest value will be 2 (255/127.5) and the lowest will be 0(0/127.5) , then we reduce 1, so it will be between -1 and 1, so all pixel values will be between -1 and 1 this is better to be used on the neural network 
  image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE)) #resizing the image to the desired size
  return image, label #returning the resized image and its label


#now apply this function to all images in the 3 datasets
train = raw_train.map(format_example) #format_example is the function we created above, map function returns the image after being applied to the format_example function
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

BATCH_SIZE = 32 #batch size, we feed 32 images per time into the neural network
SHUFFLE_BUFFER_SIZE = 1000 #it will pick 1000 images, shuffle it , and the next batch will be composed by images from those shuffled 1000 images

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) #shuffling and creating the batches of 32 each for training data
validation_batches = validation.batch(BATCH_SIZE) #creating batches of 32 each of validation data
test_batches = test.batch(BATCH_SIZE) #creating batches of 32 each for testing data

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3) #defining the shape of the input images, in this case 160x160 with 3 color channels 

# Creating and configuring the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, #define the input image shape
    include_top=False, #We don't want the top layer/classifier (output layer) as we will create our own output layer to classify data, we just want to use the pre-trained model to extract features
    weights='imagenet' #using predetermined weights from imagenet (Googles dataset).
)

base_model.trainable = False #freezing the layers of the base model, as the weights and biases of it is already trained a lot, we just want to train the layers we add after, with our training data

#defining our sequential model using the base model as the initial layers, and then adding our layers to classify the images
model = tf.keras.Sequential([
  base_model,#setting the base model as the first layers of this model
  tf.keras.layers.GlobalAveragePooling2D(), #first layer after the base model, it is a pooling layer to downsample the feature maps and retain important features, here we are using global average, which means each group of pixels is represented on the output feature map as the average of them, it will also flatten the data into a single dimension
  keras.layers.Dense(1) #output layer, it is a dense layer with a single neuron to classify if the image is a dog or cat
])

#compiling the model
base_learning_rate = 0.0001 #how much to modify the biases and weights on training process (modify 0.0001 meaning a small change when updating it, because we don't want significant updates as we are already using a pre-trained model before it )
model.compile(
  optimizer=tf.keras.optimizers.legacy.RMSprop(lr=base_learning_rate), #defining the optimizer function, we will use the RMS, so we need to provide a base learning rate to it
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), #defining the loss layer to calculate the error/loss of each output. using BINARY crossentropy because we just have 2 possible labels/classes
  metrics=['accuracy'] #defining the meaningful metric, in this case we are interested in the accuracy for each possible label/output
)

initial_epochs = 3
validation_steps = 20

##training the model providing the train data, setting number of epochs to 3 and using the validation data JUST as reference to the model
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy'] #get the accuracy array and print it
print(acc)

#evaluating the model using the validation data with 20 validation steps, which means, it will evaluate 20 batches of input data before stopping to evaluate, it won't evaluate all the data
loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps) 
print(accuracy0)

#We can save our trained model to be used later whenever we want
model.save("dogs_vs_cats.h5")

#then when we want to load it on our code to be used, we simply do: 
new_model = tf.keras.models.load_model('dogs_vs_cats.h5')