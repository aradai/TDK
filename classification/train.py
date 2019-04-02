#importing needed libraries
import tensorflow as tf
import keras.backend as K
import keras
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
#from keras.layers import Input, Dense, Activation, Flatten, Conv2D, AveragePooling2D, BatchNormalization, Dropout
import keras.layers as l
from keras.optimizers import Adam
import numpy as np
import sys
import os
#tf.enable_eager_execution()
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

def parser(record):
    keys_to_features = {
        "image_raw": tf.FixedLenFeature([], tf.string),
        "label":     tf.FixedLenFeature([], tf.int64)
    }
    parsed = tf.parse_single_example(record, keys_to_features)
    image = tf.decode_raw(parsed["image_raw"], tf.uint8)
    image = tf.cast(image, tf.float32)
    #image = tf.reshape(image, shape=[224, 224, 3])
    label = tf.cast(parsed["label"], tf.int32)

    return {'image': image}, label


def input_fn(filenames):
  dataset = tf.data.TFRecordDataset(filenames=filenames, num_parallel_reads=40)
  dataset = dataset.apply(
      tf.contrib.data.shuffle_and_repeat(1024, 1)
  )
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(parser, 32)
  )
  #dataset = dataset.map(parser, num_parallel_calls=12)
  #dataset = dataset.batch(batch_size=1000)
  dataset = dataset.prefetch(buffer_size=2)
  return dataset


def train_input_fn():
    return input_fn(filenames=["train.tfrecords"])

def test_input_fn():
    return input_fn(filenames=["test.tfrecords"])

def val_input_fn():
    return input_fn(filenames=["val.tfrecords"])

# Load mydataset
(X_train, y_train) = train_input_fn()
(X_test, y_test) = test_input_fn()

# convert brightness values from bytes to floats between 0 and 1:
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# save validation set (for model tuning)
(X_val, y_val) = val_input_fn()
X_val = X_val.astype('float32')
X_val /= 255


print("Train data shape ", X_train.shape)
print("Train label shape ", y_train.shape)
print("Validation data shape ", X_val.shape)
print("Validation label shape ", y_val.shape)
print("Test data shape ", X_test.shape)
print("Test label shape ", y_test.shape)

INPUT_SHAPE=X_train.shape[1:]
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NB_EPOCHS = 30
#DECAY = 0.1
PATH="model/cp.ckpt"
DIR = os.path.dirname(PATH)
os.makedirs(os.getcwd()+DIR,exist_ok=True)

# Define callbacks
checkpoint = callbacks.ModelCheckpoint(PATH,
                                              save_weights_only=True,
                                              save_best_only=True,
                                              verbose=1,
                                              mode='max',
                                              monitor='val_acc')

reduceLR = callbacks.ReduceLROnPlateau(monitor='val_acc',
                                       factor=0.1,
                                       patience=3,
                                       mode='max',
                                       verbose=1
                                      )
earlyStop = callbacks.EarlyStopping(monitor='val_acc',
                                    patience=10,
                                    mode='max',
                                    restore_best_weights=True,
                                    verbose=1,
                                    #TODO baseline
                                    baseline=0.7
)



# Define the network
x = l.Input(shape=INPUT_SHAPE)

# Build the network with the following layers: Conv2D, MaxPooling2D, Flatten, Dense

out = l.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation='relu')(x)
out = l.BatchNormalization()(out)
out = l.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same", activation='relu')(out)
out = l.BatchNormalization()(out)

for i in [2, 4, 8]: #deep (till 32) 0.794
  out = l.MaxPooling2D(pool_size=2)(out)
  out = l.Conv2D(64*i, kernel_size=(3, 3), strides=1, padding="same")(out)
  out = l.Dropout(0.25)(out) #w/0.8128 #w/o 0.7975
  out = l.BatchNormalization()(out)
  out = l.LeakyReLU()(out) #w ReLU/ 0.8165 ELU 0.8256 LeakyReLU 0.8458
  plus1 = plus = out
  out = l.Conv2D(64*i, kernel_size=(3, 3), strides=1, padding="same")(out)
  out = l.Dropout(0.25)(out)
  out = l.BatchNormalization()(out)
  out = l.LeakyReLU()(out)
  out = l.add([plus1,out])
  out = l.Conv2D(64*i, kernel_size=(3, 3), strides=1, padding="same")(out)
  out = l.Dropout(0.25)(out)
  out = l.BatchNormalization()(out)
  out = l.LeakyReLU()(out)
  out = l.add([plus,out])


out = l.AveragePooling2D(pool_size=(K.int_shape(out)[1:3]))(out)
out = l.Flatten()(out)
#out = l.Dense(512, activation="relu")(out) #ezzel 79.36
out = l.Dense(10, activation='softmax')(out)
# ... AveragePooling2D, Flatten, Dense w/ softmax

model = Model(inputs=x, outputs=out)
#TODO not open close, do with code rn it has to be commented
#model.load_weights(PATH)
model.summary()

model.compile(optimizer=Adam(lr=LEARNING_RATE), #, decay=DECAY
               loss='sparse_categorical_crossentropy',
               metrics=['accuracy'])


history = model.fit(X_train, y_train,
                     batch_size=BATCH_SIZE,
                     epochs=NB_EPOCHS,
                     verbose=1,
                     validation_data=(X_val, y_val),
                     callbacks=[checkpoint,reduceLR,earlyStop])

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])