import tensorflow as tf
print(tf.__version__)
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
from wandb.keras import WandbCallback
from tensorflow.keras.utils import to_categorical
import time
import wandb
from PIL import Image
import skimage.transform
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
import pandas as pd
import random
import joblib
import matplotlib.pyplot as plt
from albumentations import (HorizontalFlip, Blur, VerticalFlip, Transpose, RandomCrop, 
                            RandomGamma, ShiftScaleRotate,
                            HueSaturationValue, RGBShift, RandomBrightness, RandomContrast, CLAHE) 

def resize_images(x, shape=(160,160,3)):
    X_data_resized = [skimage.transform.resize(image, shape) for image in x]
    
    return np.array(X_data_resized)
    
def create_image_data(dataset, split, img_spec='std'):
    #Raman Data/tox21/tox21-featurized/smiles2img/engd/index/train_dir/
    #TODO: Add option for img_spec here. Path to dir changes accordingly
    splits = ['train', 'test', 'valid']
    if(split not in splits):
        msg = 'split should be in: '+splits
        return msg
    
    #BASE_DIR='Raman Data/'
    path=dataset+'/'+dataset+'-featurized/smiles2img/'+img_spec+'/index/'+split+'_dir/'
    print(path)
    if(not os.path.exists(path)):
        msg = "Check path"
        return msg
    
    x0=joblib.load(path+'shard-0-X.joblib')
    x0_resized = resize_images(x0)
    x1=joblib.load(path+'shard-1-X.joblib')
    x1_resized = resize_images(x1)
    x2=joblib.load(path+'shard-2-X.joblib')
    x2_resized = resize_images(x2)
    x3=joblib.load(path+'shard-3-X.joblib')
    x3_resized = resize_images(x3)
    x4=joblib.load(path+'shard-4-X.joblib')
    x4_resized = resize_images(x4)
    images=np.concatenate((x0_resized, x1_resized, x2_resized, x3_resized, x4_resized))
    
    
    print("Images length: ", len(images))
    
    return images

def get_labels(dataset, split, img_spec='std'):
    #BASE_DIR='Raman Data/'
    path=dataset+'/'+dataset+'-featurized/smiles2img/'+img_spec+'/index/'+split+'_dir/'
    print(path)
    if(not os.path.exists(path)):
        msg = "Check path"
        return msg
    
    y0=joblib.load(path+'shard-0-y.joblib')
    y1=joblib.load(path+'shard-1-y.joblib')
    y2=joblib.load(path+'shard-2-y.joblib')
    y3=joblib.load(path+'shard-3-y.joblib')
    y4=joblib.load(path+'shard-4-y.joblib')
    labels=np.concatenate((y0, y1, y2, y3, y4))
    print("Labels length: ", len(labels))
    
    return labels


def augment(aug, image):
    aug_image = aug(image=image)['image']

    return aug_image

def get_balanced_data(x,y,shuffle=True):
    minority_class=1
    x = list(x)
    y = list(y)
    
    augmented_x = []
    augmented_y = []
    
    length = len(x)
    for i in range(length):
        if(int(y[i])==minority_class):
            #Augment x
            img = x[i]
            aug1 = augment(RandomBrightness(p=1), img)
            aug2 = augment(RandomContrast(p=1), img)
            aug3 = augment(ShiftScaleRotate(p=1), img)
            aug3 = augment(CLAHE(p=1), img)
            aug4 = augment(HueSaturationValue(p=1), img)
            aug5 = augment(Transpose(p=1), img)
            
            augmented_x.append(np.array(aug1))
            augmented_x.append(np.array(aug2))
            augmented_x.append(np.array(aug3))
            augmented_x.append(np.array(aug4))
            augmented_x.append(np.array(aug5))
            
            augmented_y.append(np.array(minority_class))
            augmented_y.append(np.array(minority_class))
            augmented_y.append(np.array(minority_class))
            augmented_y.append(np.array(minority_class))
            augmented_y.append(np.array(minority_class))
            
    x.append(augmented_x)
    y.append(augmented_y)
    
    if(shuffle):
        temp = list(zip(x, y)) 
        random.shuffle(temp) 
        x, y = zip(*temp) 
            
    return x,y

def plot_acc(history, key1, key2, val_present):
    plt.plot(history.history[key1])
    if(val_present):
        plt.plot(history.history[key2])
        plt.title('Train vs Validation Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper left')
    else:
        plt.title('Train Accuracy')
        plt.legend(['train'], loc='upper left')
        
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    
    plt.show()
    
def plot_loss(history, key1, key2, val_present):
    plt.plot(history.history['loss'])
    if(val_present):
        plt.plot(history.history['val_loss'])
        plt.title('Train vs Validation Loss')
        plt.legend(['Train', 'Validation'], loc='upper left')
    else:
        plt.title('Train Loss')
        plt.legend(['Train'], loc='upper left')
        
    plt.ylabel('loss')
    plt.xlabel('epoch')
    
    plt.show()
