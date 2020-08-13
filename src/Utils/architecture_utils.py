import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
import efficientnet.tfkeras as efn


def get_model(weights, length, shape):
    if(weights==None):
        print('Using Random weights')
    else:
        print('Using '+str(weights)+' weights')
    assert length in [50, 101, 152]
    
    if(length == 50):
        from tensorflow.keras.applications import ResNet50
        model = ResNet50(include_top=False, weights=weights, input_shape=shape)
    elif(length == 101):
        from tensorflow.keras.applications import ResNet101
        model = ResNet101(include_top=False, weights=weights, input_shape=shape)
    elif(length == 152):
        from tensorflow.keras.applications import ResNet152
        model = ResNet152(include_top=False, weights=weights, input_shape=shape)

    return model

def get_cbr_tiny(num_classes, input_shape=(160, 160, 3)):
    '''
    kernel_size=(3,3)
    (Conv64-bn-relu)-MaxPool
    (Conv128-bn-relu)-MaxPool
    (Conv256-bn-relu)-MaxPool
    (Conv512-bn-relu)-MaxPool
    GlobalAvgPool-Classification
    '''
    kernel_size=(3,3)
    stride=(2,2)
    pool_size=(2,2)
    filters1=64
    filters2=128
    filters3=256
    filters4=512
    
    model = Sequential()

    #Block 1
    model.add(layers.Conv2D(filters=filters1, kernel_size=kernel_size, activation='relu', input_shape=input_shape)) #Specifying input shape since first layer
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 2
    model.add(layers.Conv2D(filters=filters2, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 3
    model.add(layers.Conv2D(filters=filters3, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 4
    model.add(layers.Conv2D(filters=filters4, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))

    #Classification Block
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model

def get_cbr_small(num_classes, input_shape=(160, 160, 3)):
    '''
    kernel_size=(5,5)
    (Conv64-bn-relu)-MaxPool
    (Conv128-bn-relu)-MaxPool
    (Conv256-bn-relu)-MaxPool
    (Conv512-bn-relu)-MaxPool
    GlobalAvgPool-Classification
    '''
    kernel_size=(5,5)
    stride=(2,2)
    pool_size=(2,2)
    filters1=32
    filters2=64
    filters3=128
    filters4=256
    
    model = Sequential()

    #Block 1
    model.add(layers.Conv2D(filters=filters1, kernel_size=kernel_size, activation='relu', input_shape=input_shape)) #Specifying input shape since first layer
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 2
    model.add(layers.Conv2D(filters=filters2, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 3
    model.add(layers.Conv2D(filters=filters3, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 4
    model.add(layers.Conv2D(filters=filters4, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))

    #Classification Block
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model

def get_cbr_large(num_classes, input_shape=(160, 160, 3)):
    '''
    kernel_size=(7,7)
    (Conv32-bn-relu)-MaxPool
    (Conv64-bn-relu)-MaxPool
    (Conv128-bn-relu)-MaxPool
    (Conv256-bn-relu)-MaxPool
    (Conv512-bn-relu)-MaxPool
    GlobalAvgPool-Classification
    '''
    kernel_size=(7,7)  #For Kernel size
    stride=(2,2)
    pool_size=(3,3) #For MaxPool
    filters1=32
    filters2=64
    filters3=128
    filters4=256
    filters5=512
    
    model = Sequential()

    #Block1
    model.add(layers.Conv2D(filters=filters1, kernel_size=kernel_size, activation='relu', input_shape=input_shape)) #Specifying input shape since first layer
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 2
    model.add(layers.Conv2D(filters=filters2, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 3
    model.add(layers.Conv2D(filters=filters3, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 4
    model.add(layers.Conv2D(filters=filters4, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 5
    model.add(layers.Conv2D(filters=filters5, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))

    #Classification Block
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model

def get_cbr_large_wide(num_classes, input_shape=(160, 160, 3)):
    '''
    kernel_size=(7,7)
    (Conv64-bn-relu)-MaxPool
    (Conv128-bn-relu)-MaxPool
    (Conv256-bn-relu)-MaxPool
    (Conv512-bn-relu)-MaxPool
    GlobalAvgPool-Classification
    '''
    
    kernel_size=(7,7)  #For Kernel size
    stride=(2,2)
    pool_size=(3,3) #For MaxPool
    filters1=64
    filters2=128
    filters3=256
    filters4=512

    model = Sequential()

    #Block1
    model.add(layers.Conv2D(filters=filters1, kernel_size=kernel_size, activation='relu', input_shape=input_shape)) #Specifying input shape since first layer
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 2
    model.add(layers.Conv2D(filters=filters2, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 3
    model.add(layers.Conv2D(filters=filters3, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    #Block 4
    model.add(layers.Conv2D(filters=filters4, kernel_size=kernel_size, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=pool_size, strides=stride))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dropout(0.4))

    #Classification Block
    model.add(layers.Dense(num_classes, activation='sigmoid'))

    return model


def get_effnet(shape, weights, num_classes):
    input = layers.Input(shape=shape)

    base_model = efn.EfficientNetB3(weights=weights, include_top=False, input_shape=shape)
    base_model.trainable = True
    
    output = base_model(input)
    output = layers.GlobalMaxPooling2D()(output)
    output = layers.Dense(256)(output)
    output = layers.LeakyReLU(alpha = 0.25)(output)
    output = layers.Dropout(0.25)(output)

    output = layers.Dense(16,activation="relu")(output)
    output = layers.Dropout(0.15)(output)

    output = layers.Dense(num_classes, activation="sigmoid")(output)
    
    model = Model(input,output)

    return model

    