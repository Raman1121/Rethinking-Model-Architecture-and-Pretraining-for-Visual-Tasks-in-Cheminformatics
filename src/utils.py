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

def get_module(length=50, width=1):
    lengths = [50, 101, 152]
    widths = [1,3,4]
    
    if(length not in lengths or width not in widths):
        msg = "Length should be in "+lengths+"and width should be in "+widths
        return msg
    
    base_url = 'https://tfhub.dev/google/bit/'
    model_url = base_url+'m-r'+str(length)+'x'+str(width)+'/1'
    print(model_url)

    module = hub.KerasLayer(model_url)
    return module

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

def plot_acc(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Train vs Validation Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Train vs Validation Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def init_params(data_len, img_size):
    momentum=0.9
    BATCH_SIZE=512
    lr = 0.003 * BATCH_SIZE / 512
    STEPS_PER_EPOCH = 10
        
    if(data_len<20000):
        SCHEDULE_LENGTH=500
        SCHEDULE_BOUNDARIES = [200, 300, 400]
    elif(data_len>=20000 and data_len<50000):
        SCHEDULE_LENGTH=10000
        SCHEDULE_BOUNDARIES = [3000, 6000, 9000]
    else:
        SCHEDULE_LENGTH = 20000
        SCHEDULE_BOUNDARIES = [6000, 12000, 18000]
    
    SCHEDULE_LENGTH = SCHEDULE_LENGTH * 512 / BATCH_SIZE
    SCHEDULE_BOUNDARIES = [int(0.3*SCHEDULE_LENGTH), int(0.6*SCHEDULE_LENGTH), int(0.9*SCHEDULE_LENGTH)]
    
    return momentum, BATCH_SIZE, lr, STEPS_PER_EPOCH, SCHEDULE_LENGTH, SCHEDULE_BOUNDARIES

