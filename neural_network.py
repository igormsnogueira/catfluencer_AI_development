import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist  # load a dataset about cloathing/fashion from keras with 60.000 images for training and 10.000 images for testing
#in this dataset each label (possible output) is an integer between 0 and 9. Where:
# 0 is t-shirt/top, 1 is trouser , 2 is pullover, 3 is dress, 4 is coat, 5 is sandal, 6 is shirt, 7 is sneaker, 8 is bag and 9 is ankle boot 
#the images are in grayscale, so each pixel has a ton of gray between 0 and 255, where 0 is black and 255 is white.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split the dataset into tetsing and training

print(train_images.shape) #outputs how many data/images you have on train_images and the dimension of those images. Ex: (60.000,28,28) means 60.000 images with 28x28 size
print(train_images[0,23,23]) #outputs the pixel 23x23 of the first image (index 0). Each pixel is a number between 0 and 255. 0 is black and 255 is white
print(train_labels[:10]) #print the first 10 labels(outputs) of the training dataset

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']#defining the name associated to each output integer

#let's see one image from this dataset...
plt.figure() #using matplotlib to define a new figure
plt.imshow(train_images[1]) #choosing what image to show in the figure. In this case,we will show the second image of the training data (index 1)
plt.colorbar() #tell matplotlib to show a colorbar on the image showing color tones from the image and their numeric value in rgb scale (0 to 255)
plt.grid(False) #disable the grids that shows on top of the image
plt.show() #display/open the image on your computer

#pre-processing data, which means applying some transformations to data before feeding it into the model
#for images we usually convert the pixels values (0 to 255) into values between 0 and 1, we do it by diviging each pixel value by 255
#we normally also convert the images color to grayscale , but in this case it is already in grayscale.
#usually you always put your data values into 0 to 1 range
train_images = train_images / 255.0
test_images = test_images / 255.0

#A sequential model is a linear stack of layers where each layer is connected to the previous one.

model = keras.Sequential([ #using keras to create the architecture of a sequential neural network model. 
    keras.layers.Flatten(input_shape=(28, 28)),  # adding an input layer (1) , Flatten allows us to provide a multi dimensional input and flatten it (convert it to single dimension, creating 1 neuron per value in the multi dimensional input), this case the shape of input is 28x28 pixels, so it will create 784 neurons
    keras.layers.Dense(128, activation='relu'),  # adding a hidden layer (2), Dense is a layer with N neurons all connected to all neurons of the first layer, it also applies a linear transformation(the weighted sum) to the input data from neurons and then apply an activation function on it. In this case we are using the relu function in this layer and 128 nodes (it is a random quantity, usually a little lower than the input layer)
    keras.layers.Dense(10, activation='softmax') # adding an output layer (3), Dense output layer, with 10 neurons (as we have 10 possible labels/outputs) and using softmax activating function, that converts a vector of real numbers to a vector of probabilites that sums up to 1 (usually used on output layers of classification task)
])

model.compile( #using keras to compile the neural network model
    optimizer='adam', #defining the optimizer algorithm, in this case adam
    loss='sparse_categorical_crossentropy', #defining the loss function to be used, in this case sparse_categorial_crosentropy
    metrics=['accuracy'] #define what metric from the output we are looking for, in this case accuracy (probability of being each possible label) 
)

model.fit(train_images, train_labels, epochs=10)  #.fit() is to train our model by passing the training data, labels and epochs

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) #.evaluate() is to test the method by providing the test data, labels and we can specify the verbose mode to give more details about the result on output
print('Test accuracy:', test_acc) #printing the accuracy on the testing

predictions = model.predict(test_images) #now let's use the model to predict the classification of multiple images, just as example we are using the same images used for testing
#predictions = model.predict(test_image[0]) #this would predict just for a single image, in this case, the first one
print(predictions[0]) #prints an array with predicitons (probability distribution of being each possible label) for the first image (index 0)
prediction_index = np.argmax(predictions[0]) #use numpy function called argmax, to get the index of the higher value from an array, in this case get the label with higher probability
print("The prediction for the first image is : "+class_names[prediction_index]) #printing the predicted label
print("The expected output is ",class_names[test_labels[0]]) #shows the expected output labels
print("prediction is correct" if test_labels[0] == prediction_index else "prediction is incorrect") #compares the prediction with the expected label for this image and say if it is correct or not
