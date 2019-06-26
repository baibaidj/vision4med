# coding=utf-8
# @Time	  : 2019-02-14 12:37
# @Author   : Monolith
# @FileName : unet_convlstm_inception_2d.py

from keras.models import Model
from keras.layers import (
    Input,Activation,Dense,Flatten,Lambda, Add, TimeDistributed,Bidirectional,
    GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, BatchNormalization)

from keras.layers.convolutional import (
    Conv2D, Conv3D, Deconvolution3D,
    MaxPooling2D,  UpSampling2D, Conv2DTranspose,
    AveragePooling2D,
)
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.core import SpatialDropout3D, SpatialDropout2D
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import Adam
import tensorflow as tf


def conv2d_basic(x, filters, kernel_size = (3, 3),strides=(1, 1),
                 dilation_rate=(1, 1), padding='same', activation ='selu',
                 dropout = 0.2, weight_decay = 1e-4, name=None):
    """Utility function to apply conv + BN.

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        kernel_size: kernel size as in `Conv2D`.
        strides: strides in `Conv2D`.
        padding: padding mode in `Conv2D`.
        activation: activation in `Conv2D`.
        use_bias: whether to use a bias in `Conv2D`.
        name: name of the ops; will become `name + '_ac'` for the activation
            and `name + '_bn'` for the batch norm layer.

    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    x = Conv2D(filters, kernel_size, strides = strides,
               dilation_rate=dilation_rate,
               padding=padding,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay),
               name = name)(x)
    if activation is not None:
        ac_name = None if name is None else name + '_ac'
        x = Activation(activation, name=ac_name)(x)
    x = SpatialDropout2D(dropout)(x)
    return x

def conv2d_lstm(x, filters, kernel_size = (3, 3),strides=(1, 1),
                 dilation_rate=(1, 1), padding='same', activation ='tanh',
                 dropout = 0.2, name=None):
    x = Bidirectional(ConvLSTM2D(filters,
                kernel_size,
                strides=strides,
                padding=padding,
                dilation_rate=dilation_rate,
                activation=activation,
                dropout=dropout,
                return_sequences=True,
                name = name))(x)
    x = BatchNormalization(axis=-1)(x)
    return x

def conv2d_distributed(x, filters, kernel_size = (3, 3),strides=(1, 1),
                 dilation_rate=(1, 1), padding='same', activation ='relu',
                 dropout = 0.2, weight_decay = 1e-4, name=None):
    x = BatchNormalization(axis=-1)(x)
    x = Activation(activation)(x)

    x = TimeDistributed(Conv2D(filters, kernel_size, strides=strides,
               dilation_rate=dilation_rate,
               padding=padding,
               kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay),
               name=name))(x)
    # if activation is not None:
    #     ac_name = None if name is None else name + '_ac'
    #     x = Activation(activation, name=ac_name)(x)
    # x = TimeDistributed(SpatialDropout2D(dropout))(x)
    return x



def inception_dilation_lstm(inputs, f, bottle_neck = True):
    conv3 = conv2d_lstm(inputs, f, (3, 3), padding='same',dilation_rate=(1,1))

    conv5 = conv2d_lstm(inputs, f, (3, 3), padding='same',dilation_rate=(2,2))

    # conv7 = conv2d_lstm(inputs, f, (3, 3), padding='same',dilation_rate=(3,3))

    # conv9 = conv2d_distributed(inputs, f, (3, 3), padding='same')

    x = Concatenate()([conv3, conv5]) #, conv7, conv9
    if bottle_neck:
        x = TimeDistributed(Conv2D(f, 1, kernel_initializer='he_normal'))(x)
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)
    return x

def inception_dilation_distributed(inputs, f, bottle_neck = True):
    conv3 = conv2d_distributed(inputs, f, (3, 3), padding='same', dilation_rate=(1, 1))

    conv5 = conv2d_distributed(inputs, f, (3, 3), padding='same', dilation_rate=(2, 2))

    # conv7 = conv2d_distributed(inputs, f, (3, 3), padding='same', dilation_rate=(3, 3))

    # conv9 = conv2d_distributed(inputs, f, (3, 3), padding='same')

    x = Concatenate()([conv3, conv5]) #, conv7, conv9

    if bottle_neck:
        x = TimeDistributed(Conv2D(f , 1, kernel_initializer='he_normal'))(x)
        x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
        x = Activation('relu')(x)

    return x


def downsample_distributed_2d(input, pool_size = (2, 2), pool_type = 'averagepool'):
    if pool_type == 'maxpool':
        return TimeDistributed(MaxPooling2D(pool_size=pool_size))(input)
    elif pool_type == 'averagepool':
        return TimeDistributed(AveragePooling2D(pool_size=pool_size))(input)
    else:
        raise('pool_type can only be maxpool or averagepool')

def upsample_distributed_2d(input, pool_size = (2, 2), pool_type = 'up'):
    if pool_type == 'up':
        x = TimeDistributed(UpSampling2D(size=pool_size))(input)
    else:
        nb_filters = K.get_variable_shape(input)[-1]
        x =  TimeDistributed(Conv2DTranspose(nb_filters, (3, 3), activation='relu',
                                             padding='same', strides=(2, 2),
                                             kernel_initializer='he_normal',
                                             kernel_regularizer=l2(1e-4)))(input)
    return x



def incept_lstm_unet(img_shape = (224,224), num_channel=4):

    inputs = Input((None, img_shape[0], img_shape[1], 1))

    with tf.device('/device:GPU:0'):

        conv1 = inception_dilation_distributed(inputs, 16)
        print("conv1 shape:", conv1.shape)

        down1 = downsample_distributed_2d(conv1)
        print("pool1 shape:", down1.shape)

        conv2 = inception_dilation_distributed(down1, 32)
        print("conv2 shape:", conv2.shape)

        down2 = downsample_distributed_2d(conv2)
        print("pool2 shape:", down2.shape)

        conv3 = inception_dilation_distributed(down2, 64)
        print("conv3 shape:", conv3.shape)

        down3 = downsample_distributed_2d(conv3)
        print("pool3 shape:", down3.shape)

        conv4 = inception_dilation_distributed(down3,128)
        print("conv4 shape:", conv4.shape)

        # down4 = downsample_distributed_2d(conv4)
        # print("pool4 shape:", down3.shape)

        # conv5 = inception_dilation_distributed(down4, 128)
        # print("conv5 shape:", conv5.shape)

        # up4 = upsample_distributed_2d(conv5)
        # print("up4 shape:", up4.shape)
        # up4 = Concatenate()([up4, conv4])
        # print("up4 shape:", up4.shape)

        # conv4up = inception_dilation_lstm(up4, 64)
        # print("conv4up shape:", conv4up.shape)

        up3 = upsample_distributed_2d(conv4)
        print("up3 shape:", up3.shape)
        up3 = Concatenate()([up3, conv3])
        print("up3 shape:", up3.shape)

        conv3up = inception_dilation_lstm(up3, 32)
        print("conv3up shape:", conv3up.shape)

        up2 = upsample_distributed_2d(conv3up)
        print("up2 shape:", up2.shape)
        up2 = Concatenate()([up2, conv2])
        print("up2 shape:", up2.shape)

        conv2up = inception_dilation_lstm(up2, 16)
        print("conv2up shape:", conv2up.shape)

    with tf.device('/device:GPU:1'):

        up1 = upsample_distributed_2d(conv2up)
        print("up1 shape:", up1.shape)
        up1 = Concatenate()([up1, conv1])
        print("up1 shape:", up1.shape)

        conv1up = inception_dilation_lstm(up1, 8)
        print("conv1up shape:", conv1up.shape)

        if num_channel==1:
            output = TimeDistributed(Conv2D(1, (3, 3),
                                            activation ='sigmoid',
                                            padding='same', use_bias=False
                                            ))(conv1up)
            # output =  conv2d_distributed(conv1up, 1,
            #                              kernel_size = (3, 3),
            #                              activation ='sigmoid',
            #                              weight_decay = 1e-4,
            #                              dropout = 0,
            #                              name=None)

        else:
            output = TimeDistributed(Conv2D(num_channel+1, (3, 3),
                                            activation='softmax',
                                            padding='same', use_bias=False
                                            ))(conv1up)
    print("output shape:", output.shape)

    model = Model(inputs=inputs, outputs=output)
    #model.summary()
    # loss = self.dice_coef(label_train,[out1, out2, out3])
    #model.compile(optimizer=Adam(lr=0.0001), loss=dice_loss, metrics=[dice_coef])
    print('model compile')
    return model

if __name__=='__main__':
    from keras.utils.vis_utils import plot_model
    from time import time
    import os
    result_path = r'/home/dejun/Pictures'
    tic = time()

    test_net = incept_lstm_unet(img_shape=(224,224), num_channel=4)
    toc = time()
    print(round(toc-tic, 4))
    test_net.summary()
    # 存储模型图
    plot_model(test_net, to_file=os.path.join(result_path, 'incept_lstm_unet.png'), show_shapes=True)