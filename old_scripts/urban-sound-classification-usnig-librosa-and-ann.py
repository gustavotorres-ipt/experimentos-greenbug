#| # <center> **Audio Classification**

#| <div align='center'><img src='https://production-media.paperswithcode.com/datasets/UrbanSound8K-0000003722-02faef06.jpg'></div>

#| ***

#| ### **AIM : Aiming to develop a Neural Network Model that can predict or classify a sound or audio that belongs to which class.**

#| >

#| #### Importing all necessary libraries

#https://www.kaggle.com/sachinsarkar/urban-sound-classification-usnig-librosa-and-ann/notebook
import os
import numpy as np
import pandas as pd
import librosa as lb
import IPython.display as ipd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import Sequential, layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

#-------------------------------


#| #### Importing the metadata file contains the information about the audio dataset.

metadata = pd.read_csv('./input/UrbanSound8K.csv')
print(metadata.shape)
metadata.head()

#-------------------------------


#| #### Audio Classes

classes = metadata.groupby('classID')['class'].unique()
classes

#-------------------------------

sns.countplot(x='class',data=metadata)
plt.xticks(rotation=90)
plt.show()

#-------------------------------

#Carregar um exemplo de som
def plot_sound(filename):
    librosa_audio_data, librosa_sample_rate = lb.load(filename)
    time_x = np.arange(len(librosa_audio_data)) / librosa_sample_rate
    plt.plot(time_x, librosa_audio_data)
    plt.xlabel("Tempo (s)")

filename = './input/fold5/190893-2-0-11.wav'
plot_sound(filename)

ipd.Audio(filename)

#| #### A function that extract and returns numeric features from audio file

def feature_extractor(path):
    data, simple_rate = lb.load(path)

    data = lb.feature.mfcc(y=data, n_mfcc=128)
    data = np.mean(data,axis=1)
    return data

#-------------------------------

audio_dataset_path='input/'

#| #### Extracting Features from Audio files and preparing the dataset

x, y = [], []
for i,rows in tqdm(metadata.iterrows()):
    path = os.path.join(audio_dataset_path, 'fold' + str(rows['fold']), str(rows['slice_file_name']))

    x.append(feature_extractor(path))
    y.append(rows['classID'])
x = np.array(x)

y_labels = np.array(y)
x.shape, y_labels.shape

#| #### One Hot Transformation

y = to_categorical(y_labels)
y.shape

#| #### Train, Test and validation Split

xtrainval, xtest, ytrainval, ytest = train_test_split(x,y,test_size=0.1,stratify=None,random_state=387)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrainval,ytrainval,test_size=0.2, random_state=387)

print('\nNumber of samples for Train set :',xtrain.shape[0])
print('Number of samples for Validation set :',xvalid.shape[0])
print('Number of samples for Test set :',xtest.shape[0])

#| #### Artificial Neural Network Model Building

model = Sequential(
                        [
                            layers.Dense(1000,activation='relu',input_shape=(128,)),
                            layers.Dense(750,activation='relu'),
                            layers.Dense(500,activation='relu'),
                            layers.Dense(250,activation='relu'),
                            layers.Dense(100,activation='relu'),
                            layers.Dense(50,activation='relu'),
                            layers.Dense(10,activation='softmax')
                        ]
                   )
model.summary()

#| #### Training and Compilation of the model

print(xtrain.shape)
print(ytrain.shape)

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
training = model.fit(xtrain,ytrain,validation_data=(xvalid,yvalid),epochs=20)

#| #### Training History

train_hist = pd.DataFrame(training.history)
train_hist

#| #### Visualizing Training History

plt.figure(figsize=(20,8))
plt.plot(train_hist[['loss','val_loss']])
plt.legend(['loss','val_loss'])
plt.title('Loss Per Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.figure(figsize=(20,8))
plt.plot(train_hist[['accuracy','val_accuracy']])
plt.legend(['accuracy','val_accuracy'])
plt.title('Accuracy Per Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

#| #### Model Performance Analysis on Test Data

ytrue = np.argmax(ytest,axis=1)
ypred = np.argmax(model.predict(xtest),axis=1)
print('\n\nClassification Report : \n\n',classification_report(ytrue,ypred))

#-------------------------------

plt.figure(figsize=(10,4))
plt.title("Confusion matrix for testing data", fontsize = 15)
plt.xlabel("Predicted class")
plt.ylabel("True class")
sns.heatmap(confusion_matrix(ytrue,ypred),annot=True,
           xticklabels = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'enginge_idling', 'gun_shot', 'jackhammer', 'siren','street_music'],
           yticklabels=['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'enginge_idling', 'gun_shot', 'jackhammer', 'siren','street_music'])

plt.show()

#| #### The final Prediction function that takes the audio path and returns the predicted class along with audio

def predict(path):
    audio = np.array([feature_extractor(path)])
    classid = np.argmax(model.predict(audio)[0])
    print('Class predicted :',classes[classid][0],'\n\n')
    return ipd.Audio(path)

#| #### Testing the Prediction Function on a Audio file

predict('./input/fold6/104327-2-0-26.wav')

#| # <center> **Thank You**

