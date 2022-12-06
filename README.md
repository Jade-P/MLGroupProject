# Emotion Detection with Webcam

## Abstract
This project involves training a neural network to distinguish between the emotions (anger, sadness, neutral, happiness, fear, surprise) communicated by facial expressions. We will then run the neural network on a live webcam, at which point it will be able to take in real-world data (people’s faces) and decide which emotion the face is most likely expressing out of the 6 options. 
 
Dataset we will be using:
https://www.kaggle.com/datasets/apollo2506/facial-recognition-dataset?fbclid=IwAR3Q5t6xAClg5DRKHm5kGDV81vOwu0FOJeGGoHnjjmhAL2hbiZ6qBqferRM

## Introduction 
	
In the beginning of this project, our group wanted to work not only with an interesting dataset, but also with hardware to apply our model. As such, we looked online for datasets that we believed would be able to incorporate these two goals. In this search, we found the dataset by Gotam Dahiya that showed different faces making different expressions that were sorted by emotions. As a group, we decided to create models that could go through this data and use a camera so that we could try to detect facial expressions. Facial expressions are important, as they are a major type of social cues. In fact, it is extremely important to read others' facial expressions to recognize their emotions in order to communicate efficiently. Automatic human emotion recognition can help young children to learn various emotions, and moreover, can help hospitals to better communicate with patients that are restricted from showing emotions due to deficiencies. 

## Method
### Data exploration

You can run the data exploration portion of our project here:
[https://colab.research.google.com/drive/1Aq2uXzQSteBLBaTvhKM411fgV3PmvDPd?usp=share_link ](https://github.com/Jade-P/MLGroupProject/blob/main/Data_exploration_final.ipynb)

When exploring the data, we found that the data has the following attributes:
- each image is already 48x48 pixels
- some images have differing grayscale
- some images contain non-faces
- some images were duplicates of other images in the dataset
- each emotion class has a different number of images

The following captures the number of images in each class and in the training and testing environments.

<img width="407" alt="Screen Shot 2022-11-20 at 5 40 15 PM" src="https://user-images.githubusercontent.com/60015396/202942961-0e7d0049-cbab-4799-b8f8-f4f1c2c1b909.png">

The following show examples of each class:

<img width="633" alt="Screen Shot 2022-12-05 at 5 36 22 PM" src="https://user-images.githubusercontent.com/60015396/205786995-62028dd6-b22c-459e-8558-b226c4edd523.png">

<img width="639" alt="Screen Shot 2022-12-05 at 5 36 48 PM" src="https://user-images.githubusercontent.com/60015396/205787056-652889c7-5077-4888-861e-f51747c52160.png">

<img width="632" alt="Screen Shot 2022-12-05 at 5 37 13 PM" src="https://user-images.githubusercontent.com/60015396/205787110-a7191797-51dc-45ad-a1dc-166705e5d5fc.png">

<img width="631" alt="Screen Shot 2022-12-05 at 5 37 37 PM" src="https://user-images.githubusercontent.com/60015396/205787197-fcc635e6-bd5c-4ef3-ab43-b459278e5baa.png">

<img width="635" alt="Screen Shot 2022-12-05 at 5 38 02 PM" src="https://user-images.githubusercontent.com/60015396/205787262-823dc828-43ae-4bf6-9641-8ce184fcd6a7.png">

<img width="626" alt="Screen Shot 2022-12-05 at 5 38 27 PM" src="https://user-images.githubusercontent.com/60015396/205787311-58f38c8f-c1e9-4a4b-b687-7678a20ac195.png">

### Bias
We know it is important, especially when dealing with face image data, to try to make datasets unbiased. As we looked through our image data manually, we noticed that there was diversity of age and race among the faces. We did not quantify what percentage of faces fell into different age and race categories, since we thought that somehow identifying ages and races with code would involve a whole separate machine learning model. From our manual look through, however, there did not seem to be any glaring bias in the data.

### Image preprocessing

#### Remove Duplicates

For each emotion category in the dataset, we extracted the memory size of each image and compared images which have the same memory size with structural similarity using skimage library. Then using the dictionary data structure, we grouped the same images, saving the first image in the group as the key of the dictionary and the same images as values of the key. Finally, by iterating through each key, we removed the duplicates using the values of the key which makes only the key image remain. 

#### Remove outliers

Training images that did not contain faces, or included more than one face were removed. This process was done manually. We then added these outliers to a list. By iterating over the elements of the outliers list, we removed them from the training dataset.

#### Image augmentation

For emotions that had fewer training images, we oversampled some data by flipping it horizontally, randomly changing the brightness and contrast, randomly rotating up to 15 degrees, cropping and resizing the image. The amount of oversampling was about 80% of the difference between the amount of data and the number of the largest dataset.

### 1st Model: Convolutional Neural Networks 
We built a CNN with hidden layers having relu activation functions, batch normalization in between convolutional layers, and an output layer having  6 nodes and a softmax activation function, which would give the probability that the input belonged to each of the 6 emotion classes. Using categorical_crossentropy as our loss function allowed the model to match the highest probability from the 6 nodes populated by the softmax activation function to the appropriate emotion class.
You can find and run our CNN here: [CNN_final.ipynb](https://github.com/Jade-P/MLGroupProject/blob/main/CNN_final.ipynb)

### Second Model: VGG-16 

You can find and run our VGG-16 model here: https://github.com/Jade-P/MLGroupProject/blob/main/VGG16.ipynb

VGG-16 is a model using CNN layers which used to win Imagenet competition in the past. Using the module implemented in keras library, we imported the model and used it as base model and added several layers so that we can use it for our classification which has 6 classes in particular. The model consists of 16 layers with weights including convolution layer, max pooling layer and fully connected layer with softmax in the last. We connected this with a flattening layer and softmax with 6 units which are the number of the classes, referring to implementation of kaggle example code.

### Face Detection in Live Feed using Haar Cascade 
The Haar Cascade classifier is an algorithm that was introduced to identify faces in images/videos. The classification will be trained with a training set provided by the opencv repository (vpisarev). This repository contains multiple datasets that train on creating the rectangle over particular parts of the face, for this project we used the entire frontal face. This is a reliable source because it is published by OpenCV. 

Moreover, to apply this algorithm we have to normalize and put the entire live feed in a greyscale because the Haar Cascade requires normalized gray-scaled data. Then we can find the face coordinates using the Haar Cascade classifier. Then we can apply a rectangle over each face that is found by the classifier. Then we have to change the faces found in the image to be applicable to the model, so the image found in the rectangle is transformed into a 48x48 pixel image because that was the parameters in which the model was trained on. Then the image found in the rectangle is normalized and grayscale again to ensure the highest level of accuracy. After this the best performing model is applied to the feed and outputs an image with a label denoting the emotion. To end the live feed, you have to press ‘q’.  

The following is where the notebook for the Live Feed Detection can be found: https://github.com/Jade-P/MLGroupProject/blob/main/Live_Feed_Face_Detection_(ONLY_JUPYTER_NOTEBOOK)_.ipynb 


## Result
### Custom CNN
We trained our model with a batch size of 64 and a total of 100 epochs. This setting seems suitable to us as we balance both model performance and training time. Finally, we obtain a training accuracy of 0.85 and a training loss of 0.45. An accuracy close to 1 and a loss close to 0 indicates that the model fits data very well. Training accuracy is not used to assess the model performance, since the model could be well fitted to the training data but poorly fitted to new data. Instead, having a training set enables us to make a better decision on whether the model becomes more accurate for unseen data.

<img width="694" alt="Screen Shot 2022-12-05 at 5 49 32 PM" src="https://user-images.githubusercontent.com/60015396/205788761-21b4ea14-52cc-4a7a-b1f8-da3982b72942.png">

Under the same model, we obtain a testing accuracy of 0.66 and a testing loss of 1.21. A testing accuracy of 66% is deemed acceptable, but further refinements to our dataset and/or model building parameters could be made, as suggested later in the report.

A plot of the training vs testing accuracy ("val_accuracy" for validation accuracy) is shown below:
<img width="429" alt="new_acc" src="https://user-images.githubusercontent.com/60015396/205798291-c2bda2f0-3888-48ff-bd6a-e731d495074a.png">

This plot indicates strong overfitting. The following graphic from https://deepdatascience.wordpress.com/2016/11/17/how-to-detect-model-overfiting-by-training-accuracy/ makes that clear:

![training-accuray-explaining-if-a-model-is-overfitting](https://user-images.githubusercontent.com/60015396/204122835-59ddb2d8-5ac1-49ae-a4a8-778eaea1faaa.jpeg)

Next, we investigate how training and testing loss change as the number of epochs increases. From the graph, as expected, training loss steadily decreases as the number of epochs increases. However, testing loss quickly decreases at first, levels off and gradually increases afterwards. This suggests that our final model may be overfitted. 

This image is an example of overfitting:
<img width="374" alt="Screen Shot 2022-12-05 at 5 50 37 PM" src="https://user-images.githubusercontent.com/60015396/205788918-edcd236d-fd70-42ae-bb43-4a320ca7c2dd.png">

Below was our actual result, which follows the pattern of overfitting as shown above:
<img width="303" alt="Screen Shot 2022-12-05 at 5 50 00 PM" src="https://user-images.githubusercontent.com/60015396/205788835-cec52023-6426-4120-905c-3f10c3450bbb.png">

The weights and biases of our CNN model are continually adjusted over epochs. When we train the model over too many epochs, the weights and biases of the model may be over-tuned such that the model fits the training data very well. This is why we have a smaller training error as the number of epoch increases. However, this model may not fit the actual real-world data accurately, which is seen when the testing error increases after a suitable number of epochs. The model now seems to have ‘memorized the training set’ after learning from the training set.

When we tested this CNN using Haar Cascade, we obtained these example results:

<img width="503" alt="Screen Shot 2022-12-05 at 6 41 58 PM" src="https://user-images.githubusercontent.com/60015396/205796178-2c1dffb2-dacf-4f68-ba12-24dc99af464c.png">

<img width="501" alt="Screen Shot 2022-12-05 at 6 43 05 PM" src="https://user-images.githubusercontent.com/60015396/205796240-5ac33057-eed4-4b25-ada3-0556ea081a84.png">

#### Confusion matrix:

<img width="429" alt="Screen Shot 2022-12-05 at 5 51 49 PM" src="https://user-images.githubusercontent.com/60015396/205789066-99e3f0fe-d34c-415e-bc12-a4548b46821b.png">
Angry = 0
Fear = 1
Happy = 2
Neutral = 3
Sad = 4
Surprise = 5

Angry (0) was most frequently mistaken, when mistaken at all, for Sad (4).
Fear (1) was also most frequently mistaken for Sad (4).
Happy (2) was most frequently mistaken for Neutral (3).
Neutral (3) was most frequently mistaken for Sad (4).
Sad (4) was most frequently mistaken for Fear (1).
Surprise (5) was most frequently mistaken for Fear (1).

### VGG-16
<img width="692" alt="Screen Shot 2022-12-05 at 5 55 20 PM" src="https://user-images.githubusercontent.com/60015396/205789566-2dd5f5ae-276a-40c9-8e7c-ceb1ac9c0fe3.png">

We trained our model with a batch size of 64 and a total of 20 epochs. We obtained a testing accuracy of 0.598 and testing loss of 1.95. The number of epochs was set to a reasonable checkpoint where the test accuracy stopped making improvements after trial and error. Though the train accuracy seemed to be making progress, we stopped training because the test loss was fluctuating and increasing, indicating overfitting. 

<img width="693" alt="Screen Shot 2022-12-05 at 5 55 53 PM" src="https://user-images.githubusercontent.com/60015396/205789687-eaafca25-0dc9-40d2-ac4c-1f35e03ab29e.png">

As discussed before, as the number of epochs increases the model showed strong overfitting with only the training accuracy and validation loss increasing. Moreover, the VGG-16 tended to learn faster in the same period of time, reaching overfitting in smaller epochs.

## Discussion
### Overfitting
We tried various methods to reduce overfitting: simplifying the CNN, regularization, k-fold cross-validation, and image augmentation. 
Simplifying the CNN actually made the testing accuracy worse.
After evaluating the issues with the original CNN model, it seemed as though a L2 regularization may help with overfitting, however once applied the model could not perform above 55% accuracy, making this method ineffective. After applying L2 regularization, L1 was applied to ensure that every method was used to increase the accuracy, and L1, as expected, performed even worse, with not being able to break 25% accuracy. Thus regularization was not an acceptable method of overcoming overfitting. 
Furthermore, k-fold cross-validation seemed like an applicable method to reduce overfitting. However, after applying the method, it seemed that the model had a 56.66% accuracy, making the method have a lower accuracy than that of our normal model. Thus, k-fold cross-validation was not an effective method for overcoming overfitting.

### Image Preprocessing
Through preprocessing the training dataset, we were aiming to reduce overfitting by removing duplicates so that the model is not trained with the same data. We also removed outliers which were considered as faulty inputs to clean our training process and prevent performance degradation. In oversampling for image augmentation, we provided variation in brightness and contrast to a reasonable extent considering our original dataset. We set random rotation up to only 15 degrees because some original data is lost after rotation and the test will be conducted with faces without large variation in rotation. Oversampling was done for every emotion class except Happy considering the data distribution we have looked through in data exploration.

### Parameter tuning
After getting results from the VGG-16 model, we decided to focus on fine tuning our CNN model which had better results. To make further improvements, we have tuned parameters of CNN model such as Dropout rate, the depth of model, number of layers, number of features, batch size and kernel size. Though some parameter changes could slow the overfitting down, overall performance at last did not include significant improvements. Below is the test accuracy of each CNN model with parameter tuning. Dropout rates were adjusted from 0.5 to 0.6 or 0.7, batch sizes were adjusted from 64 to 128. Increasing dropout rate or batch size of 64 slowed down training and overfitting in some cases but again did not actually improve overall performance.
  
<img width="698" alt="Screen Shot 2022-12-05 at 6 00 13 PM" src="https://user-images.githubusercontent.com/60015396/205790192-eb69f972-8574-4913-997b-72aa287bf669.png">

### Convolutional Neural Networks 
There are many image classification models, but we specifically chose to use a Convolutional Neural Network (CNN). This is because our input data consists of images that are two dimensional, and a convolutional filter would work well to identify the patterns that most likely indicate each emotion that we are trying to identify.	

We first built a CNN with hidden layers having relu activation functions, batch normalization in between convolutional layers, and an output layer having only 1 node and a sigmoid activation function. Our choice of output layer was a mistake that led to extremely low accuracy, because our model is meant for multi-class classification, whereas a single output node is meant for binary classification. We learned that we needed to one-hot-encode our y_train and y_test arrays, and that we needed our output layer to have 6 nodes and a softmax activation function, which would give the probability that the input belonged to each of the 6 emotion classes. Using categorical_crossentropy as our loss function allowed the model to match the highest probability from the 6 nodes populated by the softmax activation function to the appropriate emotion class.

The main limitation of CNN is that the training time is very large. If you are using CPU instead of GPU to train the model, it could take anywhere from a day to a week. Even with GPU, it takes minutes to train the model. Also, a lot of training data is needed for a CNN to be effective. 
### Haar Cascade Detection
The Haar Cascade classifier detects objects in videos/images quickly and effectively. Underneath the classification is a degenerate decision tree , also known as a “cascade,” that decides whether the object is a face. The term “Haar” comes from the Haar Basis function and contributes to the fast computational time. 

Images are subsetted into smaller “sub-windows” and are applied to the decision tree, to determine if it has the appropriate features to contain a face, and if not they can be filtered out. This method is so effective that it has been shown that after the first filter (or the first node of the decision tree), the classifier can get rid of half of the image, noting that it does not contain a face. Then the sub-windows are put through more classifiers to final determine the location of a face. Interestingly, the Haar Cascade method does not use pixels directly but instead features that can be encoded, and the motivation for this is because systems that use features are considerably faster then pixel based systems. Overall the Haar Cascade classifier uses a combination of integrals and subsetting to create an effective method of determining the coordinates of a face. 


### Limitations

#### Haar Cascade 
The Haar Cascade algorithm has a difficult time in detecting faces that are not in a prime spot, so the algorithm cannot detect faces that are too close to the camera, too far from the camera, or if the face is not looking towards the camera. Moreover, the lighting has a significant role in the difficulty of detecting a face, so the lighting also has to be optimal. Lastly, it does occasionally have a false positive and tell you there is a face where there is not one. 

All of these aspects do contribute to the accuracy of the overall emotion detection because if the face detection is not completely effective then the classification model cannot appropriately applied and may yield inaccurate results.

#### Fundamental issue within the topic
As could be seen in the plotted dataset, there was some data that could be even confusing for humans, especially considering the Neutral, Sad and Surprise. In other words, determining human emotion from a face is actually somewhat subjective. For example, a smiling face does not always indicate Happy emotion and crying face does not always indicates Sad emotion. Therefore, there is a fundamental limitation in the topic of classifying human emotions by just calculating visual data.

#### Faulty test dataset
During data exploration, we also found out that there are some faulty inputs which were not a face or have multiple faces in the test dataset. Since we have cleaned the training dataset and excluded faulty inputs, outliers in the test dataset may have degraded the overall performance. 

#### Face variation
Within the dataset, some face images varied in face angles and could have impacted the model’s performance because the algorithm could have created a template of face to determine the emotion. As an imbalanced dataset resulted in a high score in a particular class, less images of faces with different angles might have been trained less, resulting in poor results on faces in different angles.

After evaluating each model, the best method to use in the live feed was the CNN model.

## Conclusion
Overall our training accuracy for both models, CNN and VGG-16, was very high, about 90%. The testing accuracy wasn’t as good, since both models got around 60%. This was surprising, because VGG-16 is known to have a higher accuracy than a normal CNN model. Training accuracy being much more accurate than testing accuracy tells us that we have an overfitting model. We tried to reduce overfitting by the methods described in the discussion section, but they all seemed to fail. We still had a similar testing accuracy around 60%, and we just got a lower training accuracy.

If we did emotion detection again in the future, we would find more data since CNN works best with a large dataset. Also, we used our webcam for live face detection because we didn’t have the correct camera for our Jetson Nano. If we had more time to work on this project, we would find the correct camera for the Jetson Nano, and build an autonomous solution with the Jetson Nano. This would extend our project to cover real-life usecases (e.g. clinical psychology settings), where obtaining a patient's emotions during clinical visits could enhance existing medical diagnosis.

## Collaboration

Arandi Mallawa
Title: Haar Cascade Detection Algorithm Expert / Model Application Engineer / Moderator
Contribution: Contributed to the pre-processing of data, by finding all non-faces in the ‘Angry’ dataset in the training and testing environments. Researched and found the most efficient way of detecting faces, then wrote the code necessary to find a box around a face in a live feed. Moreover, helping teammates with overfitting in CNN model by applying regularization to the model. Then after making the final decision about the model applying the best model found by model engineers to the face detection algorithm to produce the end result. Then writing found research and limitations in the final research paper. Apart from the direct work done on the project, moderating discussions were other key aspects of the contribution done by Arandi, such as setting up zoom meetings and leading many of the conversations during those meetings. 

Aurora Travers
Title: CNN Engineer / Editor
Contribution: Contributed to the pre-processing of data, by finding all non-faces in the “Surprised” dataset in the training and testing environments. Researched about how to build an effective CNN model and provided a skeleton to our CNN model for Jade to work on. Provided an overview in the introduction and methods (CNN part) section, and finished up our write up by writing the conclusion section.

Choon Yong Chan 
Title: Jetson Nano Engineer / Data Analyst / Editor
Contribution: As Data Analyst: Preprocess the data for the 'Sad' emotion by filtering out invalid images from the data set. As Editor: Took lead in sprucing up the Python Notebook with brief descriptions, to improve readability for our audience. Led the 'Results' section of the write up. As Jetson Nano Engineer: Took lead in procuring, setting up, troubleshooting and running the model on the Jetson Nano. Experimented with the Camera module of the Jetson Nano, which was subsequently deprioritized.

Jade Phoreman
Title: Project Manager / CNN Engineer / Writing Editor
Contribution: As “Project Manager”: Helping to initiate and schedule team meetings, delegate tasks, and setup and manage the GitHub repository. As “CNN Engineer”: coded our final CNN model, based on Aurora’s previous work and research. As “Writing Editor”: wrote/edited the README content for the first 2 milestones. Other responsibilities: helping to preprocess data for the “Neutral” emotion; trying to simplify the CNN model to reduce overfitting but that approach did not help9

Rodney Johnston
Title: Data Analyst/ Editor/ K-Fold Cross Validation Tester
Contribution: Contributed to the pre-processing of data, by finding all the non-faces in the ‘Fear’ testing and training datasets. Helped the CNN team by working with K-Fold Cross Validation to see if it was an effective method to reduce overfitting. Furthermore, wrote the introduction and contributed to the overfitting section

Sohyun Yoo
Title: Data analyst / Data preprocessor / Editor / VGG-16 Engineer
Contribution: Contributed to the pre-processing of data, by finding all the non-faces in the ‘Happy’ testing and training datasets. Created the base python notebook for data exploration and image preprocessing. Coded data analysis visualization, removing duplicates, and image augmentation (oversampling) in image preprocessing using sources. Contributed to fine tuning changing parameters such as the depth of layers, dropout rates and batch size. Contributed to separating and organizing python notebooks according to the sections. Researched about how to save models and preprocessed data and provided code for it. Wrote data exploration/preprocessing, VGG-16, parameter tuning and limitation sections for the writeup.


Citations
apollo2506. (2020, June 30). Facial recognition. Kaggle. Retrieved December 5, 2022, from https://www.kaggle.com/code/apollo2506/facial-recognition
Avinash. (2018, December 16). How to save our model to google drive and reuse it. Medium. Retrieved December 5, 2022, from https://medium.com/@ml_kid/how-to-save-our-model-to-google-drive-and-reuse-it-2c1028058cb2


basel99. (2021, August 6). Facial recognition. Kaggle. Retrieved December 5, 2022, from https://www.kaggle.com/code/basel99/facial-recognition
Convolutional Neural Network (CNN)  :   Tensorflow Core. TensorFlow. (n.d.). Retrieved December 5, 2022, from https://www.tensorflow.org/tutorials/images/cnn


Convolutional Neural Network. Engati. (n.d.). Retrieved December 5, 2022, from https://www.engati.com/glossary/convolutional-neural-network#toc-what-are-the-disadvantages-of-convolutional-neural-networks-


Sharma, N. (2018, December 25). How to do facial emotion recognition using a CNN? Medium. Retrieved December 5, 2022, from https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f


TISTORY. (2020, May 7). Python을 이용하여 중복 사진 정리하기 :: Opencv, compare_ssim. Mizys. Retrieved December 5, 2022, from https://mizykk.tistory.com/55
vpisarev, haarcascade_frontalface_default.xml, (2013), GitHub repository, https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml 


Viola, P., &amp; Jones, M. (n.d.). Rapid object detection using a boosted cascade of Simple features. Proceedings of the 2001 IEEE Computer Society Conference on Computer Vision and Pattern Recognition. CVPR 2001. https://doi.org/10.1109/cvpr.2001.990517


Versloot, C. (2022, February 15). how-to-use-k-fold-cross-validation-with-keras. GitHub. Retrieved December 5, 2022, from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md 


  Yurtsever, O., Ozan YurtseverOzan Yurtsever 1, VanBantamVanBantam 71955 silver badges2323 bronze badges, Krunal VKrunal V 1, samvoit4samvoit4 6566 bronze badges, PeterPeter 1, aps014aps014 1, Illuminati0x5BIlluminati0x5B 60477 silver badges2323 bronze badges, user5648046user5648046, & Luca VavassoriLuca Vavassori 3111 silver badge66 bronze badges. (1966, June 1). Keras: CNN model is not learning. Stack Overflow. Retrieved December 5, 2022, from https://stackoverflow.com/questions/55776436/keras-cnn-model-is-not-learning 
