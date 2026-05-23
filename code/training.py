import typing
import tensorflow as tf
from keras import layers, models, applications

basemodel = applications.MobileNetV2(
    input_shape=(96, 96, 3), # he so small only need one popcorn
    include_top=False, 
    weights='imagenet'
) 

basemodel.trainable = False

model = models.Sequential([
    basemodel,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(2, activation="softmax") # so sigma softmaxxing rizz
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=['accuracy']
)

model.fit(
# do later im lazy as FUCKKKK
)

# esp32 compabile save
convertor = tf.lite.TFLiteConverter.from_keras_model(model)
convertor.optimizations = {tf.lite.Optimize.DEFAULT}

# do the fitting thing later
with open("model.tflite", "wb") as f:
    f.write(typing.cast(bytes, convertor.convert())) 
