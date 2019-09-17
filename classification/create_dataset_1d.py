from random import shuffle
import glob
import sys
import cv2
import numpy as np
import librosa 
import os
from sklearn.preprocessing import normalize

import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

#def normalize(ls):
#    l = []
#    mi = min(ls[0])
#    ma = max(ls[0])
#    for i in ls[0]:
#        l.append = (i - mi) / (ma - mi)
#    return l



path = os.getenv("PATH")
if "./NODE/conda_env/bin" not in path:
    path += os.pathsep + "./NODE/conda_env/bin"
    os.environ["PATH"] = path

def load_data(addr):
    #print(addr)
    x, sr = librosa.load(addr, sr = 22050, mono = True) #, duration = 1.0
    if x is None:
        return None

    zero_crossings = librosa.zero_crossings(x)
    avg_zero_crossings = sum(zero_crossings)/x.shape[0]
    y = np.append(x,avg_zero_crossings)

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)
    y = np.append(y,normalize(spectral_centroids))

    spectral_rolloff = librosa.feature.spectral_rolloff(x, sr=sr)
    y = np.append(y,normalize(spectral_rolloff))

    mfccs = librosa.feature.mfcc(x, sr=sr)
    y = np.append(y, normalize(mfccs))

    return y.tolist()
 
def createDataRecord(out_filename, addrs, labels):
    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(out_filename)
    for i in range(len(addrs)):
        # print how many images are saved every 1000 images
        if not i % 100:
            print('Train data: {}/{}'.format(i, len(addrs)))
            sys.stdout.flush()
        # Load the image
        data = load_data(addrs[i])

        label = labels[i]

        if data is None:
            continue

        # Create a feature
        feature = {
            'image_raw': _float_feature(data),
            'label': _int64_feature(label)
        }
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        
        # Serialize to string and write on the file
        writer.write(example.SerializeToString())
        
    writer.close()
    sys.stdout.flush()

train_path = 'genres/*/*.wav'
# read addresses and labels from the 'train' folder
addrs = glob.glob(train_path)
#labels = [0 if 'blues' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
labels = []
for addr in addrs:
    if 'blues' in addr:
        labels.append(0)
    elif 'classical' in addr:
        labels.append(1)
    elif 'country' in addr:
        labels.append(2)
    elif 'disco' in addr:
        labels.append(3)
    elif 'hiphop' in addr:
        labels.append(4)
    elif 'jazz' in addr:
        labels.append(5)
    elif 'metal' in addr:
        labels.append(6)
    elif 'pop' in addr:
        labels.append(7)
    elif 'reggae' in addr:
        labels.append(8)
    elif 'rock' in addr:
        labels.append(9)

# to shuffle data
c = list(zip(addrs, labels))
shuffle(c)
addrs, labels = zip(*c)
    
# Divide the data into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.6*len(addrs))]
train_labels = labels[0:int(0.6*len(labels))]
val_addrs = addrs[int(0.6*len(addrs)):int(0.8*len(addrs))]
val_labels = labels[int(0.6*len(addrs)):int(0.8*len(addrs))]
test_addrs = addrs[int(0.8*len(addrs)):]
test_labels = labels[int(0.8*len(labels)):]

createDataRecord('train.tfrecords', train_addrs, train_labels)
createDataRecord('val.tfrecords', val_addrs, val_labels)
createDataRecord('test.tfrecords', test_addrs, test_labels)