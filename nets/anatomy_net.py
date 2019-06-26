# coding=utf-8
# @Time	  : 2019-05-13 15:19
# @Author   : Monolith
# @FileName : anatomy_net.py

from nets.net_basis import *

def se3d_residual_block(inputs):
    # Global average pooling
    nb_channels = K.get_variable_shape(inputs)[-1]

    net = GlobalAveragePooling3D()(inputs)
    # net = K.mean(inputs, axis = [1,2], keepdims=False)
    # net_copy = Permute((1, 2, 4, 3), name=f'permute_1_{name}')(x_copy) # (B, T, H, W, C) -> (B, T, H, C, W)
    net = Reshape((1, 1, 1, nb_channels))(net)
    # print(net.shape)
    net = Conv3D(nb_channels, kernel_size=1,
                 kernel_initializer='he_normal',
                 )(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Conv3D(nb_channels, kernel_size=1,
                 kernel_initializer='he_normal',
                 )(net)
    net = BatchNormalization()(net)
    # net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    # net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    # net = slim.batch_norm(net, fused=True)
    net = Activation('sigmoid')(net)  # tf.sigmoid(net)
    # Permute((1, 2, 4, 3), name=f'permute_1_{name}')(x_copy)  # (B, T, H, W, C) -> (B, T, H, C, W)
    net = Multiply()([inputs, net])

    return net

def se3d_fc(inputs):
    # Global average pooling
    nb_channels = K.get_variable_shape(inputs)[-1]

    net = GlobalAveragePooling3D()(inputs)
    net = Dense(nb_channels, kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('relu')(net)
    net = Dense(nb_channels, kernel_initializer='he_normal')(net)
    net = BatchNormalization()(net)
    net = Activation('sigmoid')(net)
    net = Reshape((1, 1, 1, nb_channels))(net)
    net = Multiply()([inputs, net])

    return net




if __name__ == '__main__':
    ip = Input((64,64,64,32))
    b = se3d_fc(ip)
    print(K.get_variable_shape(b))
    m = models.Model(ip, b)
    m.summary()
