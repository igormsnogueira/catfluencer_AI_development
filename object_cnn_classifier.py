import tensorflow as tf
from keras import datasets, layers, models
import matplotlib.pyplot as plt

#  LOAD AND SPLIT DATASET INTO TRAINING DATA AND TESTING DATA. This dataset is a collection of 60.000 images of size 32x32 including animals and veicles, we will classify them
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Preprocess image by Normalizing pixel values to be between 0 and 1 as it is easier for the neural network to work with and makes it faster and improves the general performance of the model by reducing the effect of outliers
train_images, test_images = train_images / 255.0, test_images / 255.0

#defining the possible labels/outputs , each output here will be represented by its correspondent index in the data
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

IMG_INDEX = 7  # defining a random image index to display it, just so we can see how it looks like

plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary) #defining the image to be displayed, which is the one at index 7
plt.xlabel(class_names[train_labels[IMG_INDEX][0]]) #adding a subtitle to the image, which will be the label(output) of it, the first column (index 0) is the name (categorical value) of it. ex: horse
plt.show() #opens the image and show on the computer

#creating a sequential model(linear stack of layers where each layer is connected to the previous one)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))) #creating an convolutional input layer with relu activation function and 32 filters having sample size of 3x3,  having the input as an image with size 32x32 pixels and 3 color channels (rgb)
model.add(layers.MaxPooling2D((2, 2))) #second layer is a pooling layer, we do a pooling on the feature maps by using sample size of 2x2 to get a pixel from each to represent this piece in the output feature map
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #third layer is a convolutional layer using the relu activation function and 64 having sample size of of 3x3
model.add(layers.MaxPooling2D((2, 2))) #fourth layer is a pooling layer with 2x2 sample size
model.add(layers.Conv2D(64, (3, 3), activation='relu')) #fifth layer is a convolutional layer using the relu activation function and 64 having sample size of of 3x3
#Until this part we just extracted features from the image, now we need add dense layers to classify the input based on those features
model.add(layers.Flatten()) #sixth layer puts the data from the previous layer in one dimension by flattening it
model.add(layers.Dense(64, activation='relu')) #7th is a dense layer, so, all neurons from previous layer connected to all neurons in this layer. We use a relu activation function here and 64 neurons
model.add(layers.Dense(10)) #output layer is also a dense layer with 10 neurons, one for each class

print(model.summary()) #this is used to display some information about the created model and its layers

#compiling the model
model.compile(optimizer='adam', #using adam optimizer function to tweak the weight and biases in order to reduce the loss
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), #using the sparse categorical crossentropy loss function to calculare the loss/erro
              metrics=['accuracy']) #the metric to be used is the accuracy, which is the probability of the input being each of the possible classes

#training the model by providing the train images, labels , number of epochs and in this example we are also providing the testing/validation data 
#This is usually done to monitor the model's performance on the testing data during training. By doing so, one can track how well the model is generalizing to new data and identify whether the model is overfitting or underfitting.
#but the testing data IS NOT used to train and tweak the weights/biases here
history = model.fit(train_images, train_labels, epochs=10, #history will contain some information about the train
                    validation_data=(test_images, test_labels))

#testing the model by providing the test images and labels, setting verbose mode to 2 to display more info and then printing the test_acc
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2) 
print(test_acc)

