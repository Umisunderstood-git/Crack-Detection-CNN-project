This project is aimed towards creating a CNN model trained on positive and negative classes of images. 
Positive-> images with having cracks
Negative -> images do not having Crack

This code follows a flow of industrial MLops steps. 

1. Data validationn-
We have used os module to check whether data set folder exists. We then also have extracted the paths of postiive and negative
folders in their seperate respective variables

After checking for two seperate folders in the data set, that is positive or negative we check whether they have images in them
The logic is used is by checking for several image extensions and extracting each in a seperate variable in form of lists
so we now have two variables 
postive_images and negative_ images (these are just lists of names of image files)

Now since we are going to use keras preprocessing's image datagen 
keras datgagen is known for editing images on the fly and it also does the classification by looking at the folder structure,
That means for folder one images it will automatically label it 0

Industrial MLops included 3 highlevel steps that we are trying to replicate

1. Train
2. Validation
3. Test

2.Data preparation
So the aim is to have 3 seperate directories for each and have two independent directories for crack and no crack in each

So we will have
Train/Crack and Train/nocrack
Validate/Crack and Validate/nocrack

Then we have used shutil copy in our self defined function split_n_copy. Here we have used list slicing to distribute the data

70 percent to train
15 percenet to validate and 
15 percent to test

The logic is getting all this seperated in name list of each, and then adding them to the above declared directories

2.1 Image preprocessing and augmentation

We declare 3 objects of keras.preprocessing's Image datagen
This augments images on the fly. This means that augmented images will not be stored in your hard disk, but in memory while sending 
images for training

We have chosen image size of 128x128 dimensions

the same object of flow_from_directory works to pick images from directory. and the class_mode="binary" tells it that the directory has two folders each of one class
so that it can label appropriately

3.Model building
We are making custom CNN model, so we use keras.models sequential model. 
This allows us to define our own cnn layers

We have chosen 3 layers of convolutional-maxpooling (max pooling was chosen since the image data set is pretty clear. WE dont have much fear of loosing spatial featurs)
As a CNN we have then added two layers of a feed forward neural network with activation ReLu and a dropout layer.

Reason of dropout -> To prevent overfitting, we randomly drop few layers so that model doesnt answer by memorizing exactly from them

we then compile our sequential model


4. Early stopping
   We have used early stoping and model checkpoint to stop training the model when we reach to a good level of accuracy

 5.Training the model. 
 In this step we gave our train_data object declared on the flow_from_directory. This will preproces/augment image to our size and options on the fly while going for the training
 Epochs chosen here are 15

Validation_data parameter in model.fit allows us to point towards the place we have our validation dataset

What happens during training?
Model only trains on the train dataset. This means cnn will do a forward pass and back propogation/weights update on the train dataset. After each of this steps it will go to validation data set
to check the accuracy of whatever it has learnt till now. 
NOTE: It will not change its weights and biases (parameters) on the basis of validation dataset

So what is the need of validation? 
Data scientist can see how model performed at the end of each epoch as it gets loaded in 
model.history["val_accuracy"]. This tells the models accuracy on the validation dataset. So now this will enable us to identify overfitting i.e if in a particular step models training accuracy is high but,
validation accuracy is low, this means model has overfitted .


6. Model evaluation

We have plot graph of various accuracy metrics along with confusion matrix and classification report


7. Save model

   Ultimately we save our model using model.save




NOTE: This is designed to work on linux and macos systems. For windows just change "/" to "\" wherever needed and you shall be good.


