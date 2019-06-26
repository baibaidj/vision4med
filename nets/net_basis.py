

from keras import models
from keras.layers import *
import keras.backend as K
from keras.utils import Sequence, to_categorical
import tensorflow as tf
from keras.regularizers import l2
from keras.applications.resnet50 import resnet50

def Upsampling(input_tensor, scale):
    dims = K.int_shape(input_tensor)
    # net = tf.keras.UpSampling2D
    # net = Lambda(lambda x: bilinear_upsameple(tensor=x, size=(scale, scale)))(input_tensor)
    if len(dims) == 4:
        net = UpSampling2D(size=(scale, scale), interpolation='bilinear')(input_tensor) #, interpolation='bilinear'
    else:
        net = TimeDistributed(UpSampling2D(size=(scale, scale), interpolation='bilinear'))(input_tensor)
    return net



def bt_act_conv_3d(ip, nb_filter, kernel_size=3, strides = 1, dilation_rate=1,
                   dropout_rate=None, weight_decay=1e-4):
    ''' Apply BatchNorm, Relu, 3x3 Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(ip)
    x = Activation('relu')(x)
    x = Conv3D(int(nb_filter),
               kernel_size,
               strides=strides,
               dilation_rate=dilation_rate,
               kernel_initializer='he_normal',
               padding='same',
               kernel_regularizer=l2(weight_decay),
               use_bias=False)(x)
    if dropout_rate:
        x = SpatialDropout3D(dropout_rate)(x)
    return x


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


# Define custom loss
def custom_loss(layer):
    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true, y_pred):
        return K.mean(K.square(y_pred - y_true) + K.square(layer), axis=-1)

    # Return a function
    return loss


# loss function for segmentation models

def top_k_loss(y_true, y_pred, t = 0.5):
    """
    search for hard pixels within the current mini-batch to calculate loss
    hard pixels defined as those with the prediction probability of correct class less than threshold
    Simply, drop pixels when they are too easy for the model
    in practice, increase threshold t for mini-batch with good performance
    decrease t for those with bad performance


    originally proposed by Wu et al in 2016 in http://arxiv.org/abs/1605.06885
    later adopted by Deepmind in segmenting Hneck OARs http://arxiv.org/abs/1809.04430
    :param y_true:
    :param y_pred:
    :param t: threshold of prediction probability to drop a pixel when calculating loss
    :return:
    """
    # y_true = np.array(y_true, dtype= np.float32)
    # y_pred = np.array(y_pred, dtype= np.float32)
    smooth = 1.0
    target_class_pred = y_true[...,1:] * y_pred[...,1:]
    hard_pixels_0 = tf.greater(target_class_pred, tf.zeros_like(target_class_pred))
    hard_pixels_t = tf.less_equal(target_class_pred, tf.zeros_like(target_class_pred) + tf.constant(t))
    hard_pixels_bool = tf.logical_and(hard_pixels_0, hard_pixels_t)

    hard_pixels_pred_log = tf.log(tf.boolean_mask(y_pred[...,1:], hard_pixels_bool))

    # print(np.argmax(y_true, axis=-1))
    # print(target_class_pred)
    # print(hard_pixels)
    # print(hard_pixels_pred)
    # print(hard_pixels_pred_log)
    denom_term = tf.reduce_sum(tf.cast(hard_pixels_bool, dtype=y_pred.dtype))
    numerator_term = -tf.reduce_sum(hard_pixels_pred_log)

    loss = (numerator_term + smooth)/ (denom_term + smooth)
    return loss


def tversky_loss(y_true, y_pred, alpha=0.5,
                 beta=0.8, weight=(0.25, 1, 1, 1.5)):  # , 3, 3
    """

    :param y_true:
    :param y_pred:
    :param y_pred:
    :param alpha: # 待修改项，调节假阳，默认为0.5
    :param beta: # 待修改项，调节假阴，默认为0.5
    :param weight: # 待修改项， 人为设定权重
    :return:
    """
    class_n = K.get_variable_shape(y_pred)[-1]  # 待修改项，总分类数
    print('number of class %d' % class_n)
    total_loss = 0.
    for i in range(class_n):
        temp_true = y_true[..., i]  # G
        temp_pred = y_pred[..., i]  # P
        TP = K.sum(temp_true * temp_pred)  # G∩P，真阳
        FN = K.sum(temp_true) - K.sum(temp_true * temp_pred)  # G-(G∩P),假阴
        FP = K.sum(temp_pred) - K.sum(temp_true * temp_pred)  # P-(G∩P),假阳
        temp_loss = 1 - (TP + 1e-10) / (TP + alpha * FN + beta * FP + 1e-10)
        if weight is not None:
            temp_loss *= weight[i]
        total_loss += temp_loss
    tversky_loss = total_loss / sum(weight)
    return tversky_loss


# 基于交叉熵的focal_loss
def ce_focal_loss(y_true, y_pred, gamma=2):
    '''
    结合Cross_Entropy的focal_loss
    计算公式：(1-pt)**γ*(-y_true*log(pt))
    '''
    # 可人为设定，默认为2
    cross_entropy = -y_true * K.log(y_pred + 1e-10)
    weight = K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    ce_loss = K.sum(loss) / K.sum(y_true)
    return ce_loss


def dice_basic(y_true, y_pred):
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true * y_pred)
    denom = K.sum(y_true) + K.sum(y_pred)
    return ((2. * intersect + 1e-10) / (denom + 1e-10))


# 多分类时可计算某一类的dice值
def dice_channel(y_true, y_pred, channel=0):
    y_true = y_true[..., channel]
    y_pred = y_pred[..., channel]
    return dice_basic(y_true, y_pred)


def dice_fg(y_true, y_pred):
    y_true = y_true[..., 1:]
    y_pred = y_pred[..., 1:]
    return dice_basic(y_true, y_pred)


def dice_ch1(y_true, y_pred, channel=1):
    return dice_channel(y_true, y_pred, channel)


def dice_ch2(y_true, y_pred, channel=2):
    return dice_channel(y_true, y_pred, channel)


def dice_ch3(y_true, y_pred, channel=3):
    return dice_channel(y_true, y_pred, channel)


def dice_ch4(y_true, y_pred, channel=4):
    return dice_channel(y_true, y_pred, channel)


def dice_ch5(y_true, y_pred, channel=5):
    return dice_channel(y_true, y_pred, channel)

def dice_ch6(y_true, y_pred, channel=6):

    return dice_channel(y_true, y_pred, channel)


def dice_ch7(y_true, y_pred, channel=7):
    return dice_channel(y_true, y_pred, channel)


def dice_ch8(y_true, y_pred, channel=8):
    return dice_channel(y_true, y_pred, channel)

def dice_ch9(y_true, y_pred, channel=9):
    return dice_channel(y_true, y_pred, channel)

def dice_ch10(y_true, y_pred, channel=10):
    return dice_channel(y_true, y_pred, channel)


def dice_loss(y_true, y_pred):
    return 1 - dice_basic(y_true, y_pred)

if __name__ == '__main__':
    import tensorflow as tf

    sess = tf.InteractiveSession()

    a = np.random.randint(0,high=3, size = (3,3))
    # print(a.numpy())
    a = to_categorical(a, num_classes=3)
    a = tf.convert_to_tensor(a, dtype=tf.float32)
    tf.Print(a, [a], message='This is a:')
    b = tf.add(a, a)
    b.eval()

    b = tf.random_uniform((3,3,3), 0, 1, dtype= tf.float32)
    t = 0.5
    c = top_k_loss(a, b, t)
    print(c)