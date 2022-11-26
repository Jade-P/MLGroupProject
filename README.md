# Data Exploration + Preprocessing
The purpose of Data Exploration is to better understand the raw data, and the purpose of Preprocessing the data is to prepare it to be more easily and more efficiently processed by the machine learning model.

## Exploration
When exploring the data, we found that the data has the following attributes:
- each image is already 48x48 pixels
- some images have differing grayscale
- some images contain non-faces
- some images were duplicates of other images in the dataset
- each emotion class has a different number of images

The following captures the number of images in each class and in the training and testing environments.

<img width="407" alt="Screen Shot 2022-11-20 at 5 40 15 PM" src="https://user-images.githubusercontent.com/60015396/202942961-0e7d0049-cbab-4799-b8f8-f4f1c2c1b909.png">


You can find plots, graphs, and other elements of our data exploration in our Jupyter notebook here: https://colab.research.google.com/drive/1yU7sL6httnSwWoeMtvxaqGt_K2QgvANT?usp=sharing or in the Preprocessing.ipynb file here in GitHub.

### Bias
We know it is important, especially when dealing with face image data, to try to make datasets unbiased. As we looked through our image data manually, we noticed that there was diversity of age and race among the faces. We did not quantify what percentage of faces fell into different age and race categories, since we thought that somehow identifying ages and races with code would involve a whole separate machine learning model. From our manual look through, however, there did not seem to be any glaring bias in the data.

## Preprocessing
Preprocessing steps we already completed:
- dropping any image that did not contain exactly one face (including non-faces or multiple faces)
- dropping images that were duplicates of other images in the dataset
- transforming images to the same grayscale
- normalizing all images to scale pixel values between 0-1

Preprocessing steps we plan to complete when we create our model:
- add a resizing layer using keras.layers.Resizing
- standardizing/scaling images using keras.preprocessing.image.ImageDataGenerator

# First Model

## CNN
We first built a CNN with hidden layers having relu activation functions, batch normalization in between convolutional layers, and an output layer having only 1 node and a sigmoid activation function. Our choice of output layer was a mistake that led to extremely low accuracy, because our model is meant for multi-class classification, whereas a single output node is meant for binary classification. We learned that we needed to one-hot-encode our y_train and y_test arrays, and that we needed our output layer to have 6 nodes and a softmax activation function, which would give the probability that the input belonged to each of the 6 emotion classes. Using categorical_crossentropy as our loss function allowed the model to match the highest probability from the 6 nodes populated by the softmax activtion function to the appropriate emotion class. After we understood this, we trained our model (which was very similar to a model we found in this article: https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f) over 100 epochs and achieved 91.56% training accuracy and 66.03% testing accuracy. To customize the model and make sure we weren't just using the model from the article, we added some more convolutional layers with kernels of 7x7 and 5x5, which we thought would allow the model to take into account more details and complexity. After these changes and training over 100 epochs, our model achieved 96.65% training accuracy and 63.80% testing accuracy.
