import keras
from keras.models import Model, Sequential, load_model
import keras.layers as layers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K

from keras.applications.vgg16 import VGG16
import ulti
from ulti import load_data

# parameter
ulti.data_dir = "data"
ulti.num_imgs = 500
ulti.ratio_train = 0.8
ulti.w = 224
ulti.h = 224

batch_size = 16
epochs = 300

x_train, y_train = load_data(dset="train")
x_test, y_test = load_data(dset="test")

# model

# # model like vgg
base_model =  VGG16(weights=None, include_top=False, pooling=None, input_shape=(224,224,3))
inputs = base_model.input
# x = base_model.output
x = base_model.get_layer("block3_pool")
x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(100, activation='relu', name='fc1')(x)
x = layers.Dense(1, activation="sigmoid", name="fc2")(x)


model = Model(inputs=inputs,outputs=x)

# train
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# evaluate
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)