#Crack detection cnn
import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img,img_to_array
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Sequential, load_model
from keras.losses import binary_crossentropy
from keras.metrics import binary_accuracy
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import confusion_matrix,classification_report


############################################################################
#Defining parameters for the model 
############################################################################

IMAGE_SIZE=128
BATCH_SIZE=32
EPOCHS=15






# Step 1: Data validation 

####Folder validation
def checkDir(foldername):
    if os.path.exists(foldername)==True:
        return True
    else:
        return False


Present_path=os.getcwd()
main_folder=os.path.join(Present_path,"Crack-Detection","archive-2")

project_folder=os.path.join(main_folder,"Project_folder")



Positive_folder=os.path.join(main_folder,"Positive")
Negative_folder=os.path.join(main_folder,"Negative")

if checkDir(main_folder)==False:
    print("Folder not found",main_folder)
    exit()
else:
    print("Main folder exists at path : ",main_folder)


if checkDir(main_folder)==True:
    if checkDir(Positive_folder)== True and checkDir(Negative_folder)==True:
        print("Both positive and negative class of images folder exist at path ", Positive_folder," and ",Negative_folder)
    elif checkDir(Positive_folder)==True and checkDir(Negative_folder)==False:
        print("Positive class folder found. Unable to locate negative class folder")
    elif checkDir(Negative_folder)==True and checkDir(Positive_folder)==False:
        print("Negative class folder found. Unable to locate Positive class folder")
    else:
        print("Unable to locate positive as well as negative named folders")   
        exit()     



#Step2- Validating whether image files are present in the folder
#NOTE: There can be other files than images in a folder. We will need to make sure that we just take in image files


def getImages(file_name):
    valid_extensions=((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    return [
        file for file in os.listdir(file_name)
        if file.lower().endswith(valid_extensions)
    ]

positive_images=getImages(Positive_folder)
negative_images=getImages(Negative_folder)


print("Number of original positive images: ",len(positive_images)) #20,000
print("Number of original negative images: ",len(negative_images)) #20,000


#Adding one more validation checkpoint

if len(positive_images)==0 or len(negative_images)==0:
    print("Error: Positive or negative folder contains no images")


##Splitting data into train, validate and test

folders=["train/crack","validate/crack","test/crack","train/Nocrack","validate/Nocrack","Test/Nocrack"]

for folder in folders:
    path=os.path.join(project_folder,folder)
    os.makedirs(name=path,exist_ok=True)


def split_n_copy(source_folder,img_files,class_nameFlag=True):

    if class_nameFlag==True:
        class_name="crack"
    else:
        class_name="Nocrack"

    totalImgCount=len(img_files)
    train_count=int(totalImgCount*0.7) 
    test_count=int(totalImgCount*0.15 ) 
    validate_count=int((totalImgCount*0.15))
    train_imgs=img_files[:train_count]
    validate_imgs=img_files[train_count:train_count+validate_count]
    test_imgs=img_files[train_count+validate_count:]
    
    print("Length of total images is", totalImgCount)
    print("We are allocating these number of images for training", train_count)
    print("We are allocating these number if images for testing : ", test_count)
    print("We are allocating these number of images for validating: ", validate_count)

    print("number of images selected for training : ", len(train_imgs))
    print("number of images selected for testing : ", len(test_imgs))
    print("number of images selected for validating : ", len(validate_imgs))

    #Now i need to move the images to respective folders
     
   #loop to insert training images in the train directory
    Train_directory= os.path.join(project_folder,"train/"+class_name)   
    for i in train_imgs:
        try:
            shutil.copy(source_folder+"/"+i,Train_directory)
        except Exception as e:
            print(e)

   #loop to insert testing images in the test directory
    test_directory=os.path.join(project_folder,"test/"+class_name)
    for i in test_imgs:
        try:
            shutil.copy(source_folder+"/"+i,test_directory)
        except Exception as e:
            print(e)
   #loop to insert validating images 
    validate_directory=os.path.join(project_folder,"validate/"+class_name)
    for i in validate_imgs:
        try:
            shutil.copy(source_folder+"/"+i,validate_directory)
        except Exception as e:
            print(e)
    print("Images copied in ",Train_directory,"\n",test_directory,"\n",validate_directory)


split_n_copy(Positive_folder,positive_images,class_nameFlag=True)
split_n_copy(Negative_folder,negative_images,class_nameFlag=False)



#Creating train, validation and test paths
train_dir=os.path.join(project_folder, "train")
test_dir=os.path.join(project_folder, "test")
validate_dir=os.path.join(project_folder, "validate")



##Image preprocessing and augmentation 

train_datagen=ImageDataGenerator(
    rotation_range=15,
    rescale=1.0/255,
    width_shift_range=0.1,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1.0/255)

validate_datagen=ImageDataGenerator(rescale=1.0/255)

train_data=train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)


validate_data=validate_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)
test_data=test_datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

#Here we have just created a object for each train,test and validate directory
#The image augmentation, scaling and resize will happen on the fly while training by keras Image generator
print("Class indices : ",train_data.class_indices)



##Model building-> Building industrial custom CNN model

model=Sequential() #Sequential model object

#Defining a convo layer and adding it in the sequential model at the same time
model.add(Conv2D(filters=32,kernel_size=(3,3),activation="relu",padding="same",input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters=64,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=128,kernel_size=(3,3),activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=256,kernel_size=(3,3),activation="relu",padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) #=> End of the convolutional layer


model.add(Dense(256,activation="relu"))
model.add(Dropout(0.5))



model.add(Dense(128,activation="relu"))
model.add(Dropout(0.5))


model.add(Dense(1,activation="sigmoid")) #Final layer, we apply sigmoid here since its a binary classification case study



###COMPILE THE MODEL:

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)


model.summary()



##Industrial training call backsz

early_stop=EarlyStopping(
    monitor="val_loss",
    patience=4, 
    restore_best_weights=True
)

checkPoint= ModelCheckpoint(
    "Best_Crack_detection_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,

)

reduce_lr=ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=0.00001,
    verbose=1

)


####Train the built CNN model

history=model.fit(
train_data,
epochs=EPOCHS,
validation_data=validate_data,
callbacks=[early_stop,checkPoint,reduce_lr]


)



###Plot accuracy graph############

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Mandar's CNN Training vs Validation Accuracy")
plt.legend()
plt.show()



###Plot loss graph

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Mandar's CNN Training vs Validation Loss")
plt.legend()
plt.show()




##Evaluate model on Test dataset

test_loss, test_accuracy = model.evaluate(test_data)

print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy * 100)


## Confusion matrix and classification report

predictions=model.predict(test_data)
predicted_classes=(predictions>0.5).astype(int).reshape(-1)

actual_classes=test_data.classes

print("Confusion matrix ")
print(confusion_matrix(actual_classes,predicted_classes))

print("Classification report")
print(classification_report(
    actual_classes,
    predicted_classes,
    target_names=list(test_data.class_indices.keys())

))



####Save final model 

model.save("Final_Mandar's_CNN_Crack_Detection_Model.keras")
print("Final model saved successfully")
