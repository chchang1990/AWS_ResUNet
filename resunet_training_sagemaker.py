# -*- coding: utf-8 -*-
""" Acknowlegement 
Credits to (1) https://github.com/feevos/resuneta, (2) https://github.com/Akhilesh64/ResUnet-a 
for helping me have a clearer understanding on the ResUNet-a D6 structure.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from sagemaker_tensorflow import PipeModeDataset

import argparse
import os
import re
import time

HEIGHT = 256
WIDTH = 256
DEPTH = 4
NUM_CLASSES = 2


# # ----- Data pipline -----
#def read_tf_dataset(tfrecords_pattern_path, batch_size, shuffle):
def read_tf_dataset(channel_name, batch_size):
    
    # ----- Function to import the TFRecords -----
    def read_tfrecords(files):
         return tf.data.TFRecordDataset(files, compression_type="GZIP")

    # ----- TFRecord-decoding function -----
    def _parse_image_function(example_proto, batch_size=batch_size, shuffle=shuffle):

        image_feature_description = {
              'image': tf.io.FixedLenFeature([], tf.string),
              'extent': tf.io.FixedLenFeature([], tf.string),
              'boundary': tf.io.FixedLenFeature([], tf.string),
              'distance': tf.io.FixedLenFeature([], tf.string),
              'color': tf.io.FixedLenFeature([], tf.string)
        }

        # Parse the input tf.Example proto using the dictionary above.
        single_example = tf.io.parse_example(example_proto, image_feature_description)

        tile_size = 256

        image =  tf.io.decode_raw(single_example['image'],out_type='float32')
        img_array = tf.reshape( image, (batch_size, tile_size, tile_size, 4))
        img_array = tf.cast(img_array, tf.float32)


        extent =  tf.io.decode_raw(single_example['extent'],out_type='float32')
        extent = tf.reshape(extent, (batch_size, tile_size,tile_size))
        extent = tf.cast(extent,tf.uint8)

        boundary =  tf.io.decode_raw(single_example['boundary'],out_type='float32')
        boundary = tf.reshape(boundary, (batch_size, tile_size,tile_size))
        boundary = tf.cast(boundary,tf.uint8)

        distance =  tf.io.decode_raw(single_example['distance'],out_type='float32')
        distance = tf.reshape(distance, (batch_size, tile_size,tile_size, 2))
        distance = tf.cast(distance,tf.float32)

        color =  tf.io.decode_raw(single_example['color'],out_type='float32')
        color = tf.reshape(color, (batch_size, tile_size,tile_size, 3))
        color = tf.cast(color,tf.float32)

        mask={}
        mask['extent'] = tf.one_hot(extent, 2) # need to one-hot encode
        mask['boundary'] = tf.one_hot(boundary,2) # need to one-hot encode
        mask['distance'] = distance
        mask['color'] = color

        return img_array, mask


    # # ----- Main function of TFRecords data pipeline -----
    #files = tf.data.Dataset.list_files(tfrecords_pattern_path)
    #dataset = files.interleave(read_tfrecords, cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)

    # # Print to make sure the number of data is correct
    # # Each shard has 78 data
    #num_elements = 0
    #for _ in dataset:
    #    num_elements = num_elements + 1
    #print(num_elements)

    dataset = PipeModeDataset(channel=channel_name, record_format='TFRecord')

    # Validation and test sets do not need to be shuffled.
    if not channel_name=='eval' :
        # Ideally, buffer_size should be the same as data size.
        # But it depends on the size of data and the available RAM.
        dataset = dataset.shuffle(buffer_size=100)

    # Mini-batching for batch training
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)

    # Note: Applying cache() seems to explode the system RAM, and thus is not used here.
    # dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_image_function, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    return dataset



# # ----- Define ResUnet-a D6 structure -----
class ResUnet(object):

    def __init__(self, num_classes=2, input_shape=(256,256,4)):
        self.num_classes = num_classes
        self.height, self.width, self.channels = input_shape


    def residual_block(self, input, filters, kernel_size, dilation_rates, strides=1):
        out = [input]
        for rate in dilation_rates:
            x = BatchNormalization()(input)
            x = Activation('relu')(x)
            x = Conv2D(filters, kernel_size, dilation_rate = rate, padding='same', strides=strides)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(filters, kernel_size, dilation_rate = rate, padding='same', strides=strides)(x)
            out.append(x)
        out = Add()(out)
        return out

    def PSPPooling(self, input, filters):
        out = [input]
        for i in [1,2,4,8]:
            x = MaxPooling2D(pool_size = i, strides = i)(input)
            x = UpSampling2D(size = i)(x)
            x = Conv2D((filters//4), 1, padding='same')(x)
            x = BatchNormalization()(x)
            out.append(x)

        out = Concatenate(axis=-1)([out[0],out[1],out[2],out[3]])
        out = Conv2D(filters, 1, padding = 'same')(out)
        out = BatchNormalization()(out)
        return out

    def combine(self, x, y, filters):
        x = UpSampling2D(size=2)(x)
        x = Activation('relu')(x)
        x = Concatenate(axis=-1)([x,y])
        x = Conv2D(filters, 1, padding = 'same')(x)
        x = BatchNormalization()(x)
        return x

    def build_model(self):

        # ----- Encoder -----
        input = Input(shape=(self.height,self.width, self.channels))
        x = Conv2D(32, 1, padding= 'same')(input)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        a = x                                                      # conv1

        x = self.residual_block(x, 32, 3, [1,3,15,31])
        b = x                                                      # Dn1

        x = Conv2D(64, 1, strides = 2, padding='same')(x)
        x = self.residual_block(x, 64, 3, [1,3,15,31])
        c = x                                                      # Dn2

        x = Conv2D(128, 1, strides = 2, padding='same')(x)
        x = self.residual_block(x, 128, 3, [1,3,15])
        d = x                                                      # Dn3

        x = Conv2D(256, 1, strides = 2, padding='same')(x)
        x = self.residual_block(x, 256, 3, [1,3,15])
        e = x                                                      # Dn4

        x = Conv2D(512, 1, strides = 2, padding='same')(x)
        x = self.residual_block(x, 512, 3, [1])
        f = x                                                      # Dn5

        x = Conv2D(1024, 1, strides = 2, padding='same')(x)
        x = self.residual_block(x, 1024, 3, [1])
        g = x                                                      # Dn6

        # ----- Bridge -----
        x = self.PSPPooling(x, 1024)
        x = Activation('relu')(x)                                  # middle

        # ----- Decoder -----
        x = self.combine(x, f, 512)                                # middle+Dn5
        x = self.residual_block(x, 512, 3, [1,3,15])

        x = self.combine(x, e, 256)                                # UpConv1+Dn4
        x = self.residual_block(x, 256, 3, [1,3,15])

        x = self.combine(x, d, 128)                                # UpConv2+Dn3
        x = self.residual_block(x, 128, 3, [1,3,15,31])

        x = self.combine(x, c, 64)                                 # UpConv3+Dn2
        x = self.residual_block(x, 64, 3, [1,3,15,31])

        x = self.combine(x, b, 32)                                 # UpConv4+Dn1
        x = self.residual_block(x, 32, 3, [1,3,15,31])

        #x1 = self.combine(x, a, 32)
        x1 = Concatenate(axis=-1)([x,a])                      # UpConv5+conv1
        x = self.PSPPooling(x1, 32)
        x = Activation('relu')(x)

        # ----- Multi-tasking -----
        # -- Color (HSV) --
        color = Conv2D(3, 1)(x1)
        color = Activation('sigmoid', name='color')(color)

        # -- Distance --
        dist = ZeroPadding2D(padding=1)(x1)
        dist = Conv2D(32, 3)(dist)
        dist = BatchNormalization()(dist)
        dist = Activation('relu')(dist)
        dist = ZeroPadding2D(padding=1)(dist)
        dist = Conv2D(32, 3)(dist)
        dist = BatchNormalization()(dist)
        dist = Activation('relu')(dist)
        if self.num_classes==1:
            act_func = 'sigmoid'
        else:
            act_func = 'softmax'
        dist = Conv2D(self.num_classes, 1, activation=act_func, name = 'distance')(dist)

        # -- Boundaries --
        bound = Concatenate(axis=-1)([x, dist])
        bound = ZeroPadding2D(padding=1)(bound)
        bound = Conv2D(32, 3)(bound)
        bound = BatchNormalization()(bound)
        bound = Activation('relu')(bound)
        bound = Conv2D(self.num_classes, 1, activation='sigmoid', name = 'boundary')(bound)

        # -- Extents --
        seg = Concatenate(axis=-1)([x,bound,dist])
        seg = ZeroPadding2D(padding=1)(seg)
        seg = Conv2D(32, 3)(seg)
        seg = BatchNormalization()(seg)
        seg = Activation('relu')(seg)
        seg = ZeroPadding2D(padding=1)(seg)
        seg = Conv2D(32, 3)(seg)
        seg = BatchNormalization()(seg)
        seg = Activation('relu')(seg)
        seg = Conv2D(self.num_classes, 1, activation=act_func, name = 'extent')(seg)

        model = Model(inputs = input, outputs={'extent': seg, 'boundary': bound, 'distance': dist, 'color': color})

        return model




# # ----- Tanimoto loss function -----
def Tanimoto_loss(label, pred):
    """
    Implementation of Tanimoto loss in tensorflow 2.x
    -------------------------------------------------------------------------
    Tanimoto coefficient with dual from: Diakogiannis et al 2019 (https://arxiv.org/abs/1904.00592)
    """
    smooth = 1e-5

    Vli = tf.reduce_mean(tf.reduce_sum(label, axis=[1,2]), axis=0)
    # wli =  1.0/Vli**2 # weighting scheme
    wli = tf.math.reciprocal(Vli**2) # weighting scheme

    # ---------------------This line is taken from niftyNet package --------------
    # ref: https://github.com/NifTK/NiftyNet/blob/dev/niftynet/layer/loss_segmentation.py, lines:170 -- 172
    # First turn inf elements to zero, then replace that with the maximum weight value
    new_weights = tf.where(tf.math.is_inf(wli), tf.zeros_like(wli), wli)
    wli = tf.where(tf.math.is_inf(wli), tf.ones_like(wli) * tf.reduce_max(new_weights), wli)
    # --------------------------------------------------------------------

    square_pred = tf.square(pred)
    square_label = tf.square(label)
    add_squared_label_pred = tf.add(square_pred, square_label)
    sum_square = tf.reduce_sum(add_squared_label_pred, axis=[1, 2])

    product = tf.multiply(pred, label)
    sum_product = tf.reduce_sum(product, axis=[1, 2])

    sum_product_labels = tf.reduce_sum(tf.multiply(wli, sum_product), axis=-1)

    denomintor = tf.subtract(sum_square, sum_product)

    denomintor_sum_labels = tf.reduce_sum(tf.multiply(wli, denomintor), axis=-1)

    loss = tf.divide(sum_product_labels + smooth, denomintor_sum_labels + smooth)

    return loss


def Tanimoto_dual_loss():
    '''
        Implementation of Tanimoto dual loss in tensorflow 2.x
        ------------------------------------------------------------------------
            Note: to use it in deep learning training use: return 1. - 0.5*(loss1+loss2)
            OBS: Do use note's advice. Otherwise tanimoto doesn't work
    '''
    def loss(label, pred):
        loss1 = Tanimoto_loss(pred, label)
        pred = tf.subtract(1.0, pred)
        label = tf.subtract(1.0, label)
        loss2 = Tanimoto_loss(label, pred)
        loss = (loss1+loss2)*0.5
        return 1.0 - loss
    return loss





# # ----- Main function to compile and train the model -----
# -- Define Loss function --
loss = Tanimoto_dual_loss()
losses = {'extent': loss, 'boundary': loss, 'distance': loss, 'color': loss}

def main(args):
    # Hyper-parameters
    epochs       = args.epochs
    lr           = args.learning_rate
    batch_size   = args.batch_size

    # SageMaker options
    training_dir   = args.training
    validation_dir = args.validation
    eval_dir       = args.eval

    #train_dataset = read_tf_dataset(training_dir+'/train_*-of-*.tfrecords', batch_size=batch_size, shuffle=True)
    #val_dataset = read_tf_dataset(validation_dir+'/val_*-of-*.tfrecords', batch_size=batch_size, shuffle=False)
    #eval_dataset = read_tf_dataset(eval_dir+'/test_*-of-*.tfrecords', batch_size=batch_size, shuffle=False)

    train_dataset = read_tf_dataset(training_dir, batch_size=batch_size)
    val_dataset = read_tf_dataset(validation_dir, batch_size=batch_size)
    eval_dataset = read_tf_dataset(eval_dir, batch_size=batch_size)

    # Get the model
    resunet_a = ResUnet(NUM_CLASSES, (HEIGHT, WIDTH, DEPTH)) #, args.layer_norm)
    model = resunet_a.build_model()

    # Compile model
    model.compile(optimizer=Adam(lr=lr),
                  loss=losses,
                  metrics={'extent': ['accuracy']})

    # Train model
    history = model.fit(train_dataset, batch_size, validation_data=val_dataset, epochs=epochs)

    # Evaluate model performance
    score = model.evaluate(eval_dataset, batch_size, verbose=1)
    print('Test loss    :', score[0])
    print('Test accuracy:', score[1])

    # Save model to model directory
    model.save(f'{os.environ["SM_MODEL_DIR"]}/{time.strftime("%m%d%H%M%S", time.gmtime())}', save_format='tf')


# # ----- Argument parser -----
#%%
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--batch-size',    type=int,   default=16)

    # SageMaker parameters
    parser.add_argument('--model_dir',        type=str)
    parser.add_argument('--training',         type=str,   default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--validation',       type=str,   default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--eval',             type=str,   default=os.environ['SM_CHANNEL_EVAL'])

    args = parser.parse_args()
    main(args)
