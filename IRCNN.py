import numpy as np

# Sys
import warnings
# Keras Core
from keras.layers.convolutional import MaxPooling2D, Convolution2D, AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Dropout, Dense, Flatten, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.layers import Add
from keras import regularizers
from keras import initializers
from keras.models import Model

# Backend
from keras import backend as K
# Convolution 2D with batch norm
def conv2d_bn(x, nb_filter, num_row, num_col,
            padding='same', strides=(1, 1), use_bias=False):
  """
  Utility function to apply conv + BN. 
  (Slightly modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)
  """
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1
  x = Convolution2D(nb_filter, (num_row, num_col),
                    strides=strides,
                    padding=padding,
                    use_bias=use_bias,
                    kernel_regularizer=regularizers.l2(0.00004),
                    kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(x)
  x = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(x)
  x = Activation('relu')(x)
  return x

# Recurrent convolutional layer
def RCL(input, kernel_size, filedepth):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1

  conv1 = Convolution2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                 kernel_regularizer=regularizers.l2(0.00004),
                 kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(input)

  stack2 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(conv1)
  stack2 = Activation('relu')(stack2)

  RCL = Convolution2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same', 
                 kernel_regularizer=regularizers.l2(0.00004),
                 kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))

  conv2 = RCL(stack2)
  stack3 = Add()([conv1, conv2])
  stack4 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack3)
  stack4 = Activation('relu')(stack4)


  conv3 = Convolution2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                 weights=RCL.get_weights(),
                 kernel_regularizer=regularizers.l2(0.00004),
                 kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(stack4)
  stack5 = Add()([conv1, conv3])
  stack6 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack5)
  stack6 = Activation('relu')(stack6)


  conv4 = Convolution2D(filters=filedepth, kernel_size=kernel_size, strides=(1, 1), padding='same',
                 weights=RCL.get_weights(),
                 kernel_regularizer=regularizers.l2(0.00004),
                 kernel_initializer=initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(stack6)
  stack7 = Add()([conv1, conv4])
  stack8 = BatchNormalization(axis=channel_axis, momentum=0.9997, scale=False)(stack7)
  stack8 = Activation('relu')(stack8)

  return stack8


def IRCNN_block(input):
  if K.image_data_format() == 'channels_first':
    channel_axis = 1
  else:
    channel_axis = -1

  branch_0 = RCL(input, (1, 1), 64)

  branch_1 = RCL(input, (3, 3), 128)

  branch_2 = AveragePooling2D((3,3), strides=(1,1), padding='same')(input)
  branch_2 = RCL(branch_2, (1, 1), 64)

  x = concatenate([branch_0, branch_1, branch_2], axis=channel_axis)
  return x
  def IRCNN_base(input):

  if K.image_data_format() == 'channels_first':
#     inputShape = (3, 256, 256)
    channel_axis = 1
  else:
#     inputShape = (256, 256, 3)
    channel_axis = -1

  # Input Shape is 3 x 256 x 256
  net = Convolution2D(32, (3, 3), strides=(2,2), padding='valid')(input)
  net = conv2d_bn(net, 32, 3, 3, padding='valid')
  net = conv2d_bn(net, 64, 3, 3)

  net = IRCNN_block(input)
                 
  net = conv2d_bn(net, 32, 3, 3, strides=(2,2), padding='valid')
  net = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
  net = Dropout(0.5)(net)

  net = IRCNN_block(input)
                 
  net = conv2d_bn(net, 32, 3, 3, strides=(2,2), padding='valid')
  net = MaxPooling2D((3,3), strides=(2,2), padding='valid')(net)
  net = Dropout(0.5)(net)
                 
  net = IRCNN_block(input)
                 
  net = conv2d_bn(net, 32, 3, 3, strides=(2,2), padding='valid')
  net = GlobalAveragePooling2D()(net)
  net = Dropout(0.5)(net)
  
  
  return net


def main():
    image_size = 256
    no_classes = 8

    if K.image_data_format() == 'channels_first':
      inputs = Input(shape = (3, image_size, image_size))
    else:
      inputs = Input(shape = (image_size, image_size, 3))

    x = Convolution2D(32, (3, 3), strides=(2,2), padding='valid')(inputs)
    x = IRCNN_base(x)
    x = Dense(units=no_classes, activation='softmax')(x)

    model = Model(inputs, x, name='IRCNN')

    model.summary()

    from keras import optimizers
    from keras import callbacks
    import math

    # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
    sgd = optimizers.SGD(lr=0.01)

    filepath="./weights-{epoch:02d}-{val_accuracy:.2f}.hdf5"

    mcp = callbacks.ModelCheckpoint(filepath=filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [mcp]

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"])
    from keras_preprocessing.image import ImageDataGenerator

    aug = ImageDataGenerator(rotation_range=45,
                             fill_mode='wrap',
                             samplewise_center=True,
                             samplewise_std_normalization=True,
                             horizontal_flip=True,
                             vertical_flip=True,
                             validation_split=0.15)
    bs = 16
    
    train_path = './train'
    test_path = './test/'

    train_generator = aug.flow_from_directory(
        train_path,
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=bs,
        subset='training') # set as training data

    validation_generator = aug.flow_from_directory(
        train_path, # same directory as training data
        target_size=(256, 256),
        class_mode="categorical",
        batch_size=bs,
        subset='validation') # set as validation data

    train_step=train_generator.n//train_generator.batch_size
    valid_step=validation_generator.n//validation_generator.batch_size


    #model.fit_generator(
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_step,
        validation_data=validation_generator,
        validation_steps=valid_step,
        epochs=4,
        callbacks=callbacks_list,
        verbose=1)

    print('\nhistory dict:', history.history)
    test_gen = ImageDataGenerator()
    test_data = test_gen.flow_from_directory(test_path, target_size = (256, 256), class_mode=None, batch_size = bs,subset='test')
    # Evaluate the model on the test data using `evaluate`
    print('\n# Evaluate on test data')
    #results = model.evaluate(x_test, y_test, batch_size=128)
    results = model.evaluate(test_data, batch_size=16)
    print('test loss, test acc:', results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    #print('\n# Generate predictions for 3 samples')
    #predictions = model.predict(x_test[:3])
    #print('predictions shape:', predictions.shape)


    # make lr=0.001 after 20 epochs

if __name__ == '__main__':
    main()



