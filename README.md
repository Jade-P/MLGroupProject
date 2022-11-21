# Data Exploration + Preprocessing
The purpose of Data Exploration is to better understand the raw data, and the purpose of Preprocessing the data is to prepare it to be more easily and more efficiently processed by the machine learning model.

## Exploration
When exploring the data, we found that the data has the following attributes:
- each image is already 48x48 pixels
- some images have differing grayscale
- some images contain non-faces
- some images were duplicates of other images in the dataset
- each emotion class has a different number of images

You can find plots, graphs, and other elements of our data exploration in our Jupyter notebook here: TO DO

### Bias
We know it is important, especially when dealing with face image data, to try to make datasets unbiased. As we looked through our image data manually, we noticed that there was diversity of age and race among the faces. We did not quantify what percentage of faces fell into different age and race categories, since we thought that somehow identifying ages and races with code would involve a whole separate machine learning model. From our manual look through, however, there did not seem to be any glaring bias in the data.

