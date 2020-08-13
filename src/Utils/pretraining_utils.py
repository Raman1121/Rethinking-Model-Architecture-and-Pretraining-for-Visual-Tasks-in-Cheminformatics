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