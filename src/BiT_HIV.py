#Using BiT on HIV Dataset

from utils import *
import tensorflow as tf
print(tf.__version__)

import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras import optimizers
from wandb.keras import WandbCallback
from tensorflow.keras.utils import to_categorical

class MyBiTModel(tf.keras.Model):
  """BiT with a new head."""

  def __init__(self, num_classes, module):
    super().__init__()

    self.num_classes = num_classes
    self.head1 = tf.keras.layers.Dense(512, kernel_initializer='zeros') #Necessary to initialize with zeros
    self.head2 = tf.keras.layers.Dense(num_classes, kernel_initializer='zeros')
    self.bit_model = module
  
  def call(self, images):
    # No need to cut head off since we are using feature extractor model
    bit_embedding = self.bit_model(images)
    x1 = self.head1(bit_embedding)
    x2 = self.head2(x1)
    
    return x2

x_train = create_image_data(dataset='hiv', split='train', img_spec='std')
print(x_train.shape, y_train.shape)

x_val = create_image_data(dataset='hiv', split='valid', img_spec='std')
print(x_val.shape, y_val.shape)

x_train = create_image_data(dataset='hiv', split='test', img_spec='std')
print(x_train.shape, y_train.shape)

y_train = get_labels('hiv', 'train', 'std')
y_val = get_labels('hiv', 'valid', 'std')
y_train = get_labels('hiv', 'test', 'std')

#x, y = get_balanced_data(x_train, y_train, True)
momentum, BATCH_SIZE, lr, STEPS_PER_EPOCH, SCHEDULE_LENGTH,SCHEDULE_BOUNDARIES = init_params(x_train.shape[0], x_train.shape[1])
print(momentum, BATCH_SIZE, lr, STEPS_PER_EPOCH, SCHEDULE_LENGTH, SCHEDULE_BOUNDARIES)

lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=SCHEDULE_BOUNDARIES, 
                                                                   values=[lr, lr*0.1, lr*0.001, lr*0.0001])
optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
opt2 = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

aug = ImageDataGenerator(rotation_range=180,
                        horizontal_flip=True,
                        vertical_flip=True,
                        )
module = get_module(length=50, width=1)
model = MyBiTModel(num_classes=2, module=module)

model.compile(optimizer=optimizer,
              loss=loss_fn,
              metrics=['accuracy'])

model_dir = '/Models/HIV/'
model_name = 'ResNet50x1-BiT-Imbalanced.h5'
mc = ModelCheckpoint(filepath=model_dir+model_name, monitor='val_loss', 
                     patience=5, verbose=1, save_weights_only=False, save_best_only=True)
es = EarlyStopping(patience=5, verbose=1, monitor='val_loss')

start = time.time()

history = model.fit(aug.flow(x_train, y_train, batch_size=BATCH_SIZE),
          epochs=50,
          verbose=1,
          steps_per_epoch = 10,
          validation_data=(x_val, y_val),
          callbacks=[mc, es, WandbCallback(data_type="image", validation_data=(x_val, y_val))])

wandb.log({"training_time":time.time()-start})





