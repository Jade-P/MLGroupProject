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
