import typing
import os 
import kagglehub as kh
import tensorflow as tf
from keras import layers, models, applications, optimizers, utils

path = os.path.join(kh.dataset_download("techsash/waste-classification-data"), "DATASET", "TRAIN")

train_ds = utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="int",
    image_size=(96, 96),
    batch_size=32,
    shuffle=True,
    subset="training",
    validation_split=0.2,
    seed=42, # meaning of life
) # type: ignore

val_ds = utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="int",
    image_size=(96, 96),
    batch_size=32,
    subset="validation",
    validation_split=0.2,
    seed=42
) # type: ignore

# Organic = 0
# Recyclable = 1

basemodel = applications.MobileNetV2(
    input_shape=(96, 96, 3), # he so small only need one popcorn
    include_top=False, 
    weights='imagenet'
) 

basemodel.trainable = True

for layer in basemodel.layers[:-30]:
    layer.trainable = False

model = models.Sequential([
    basemodel,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(2, activation="softmax") # so sigma softmaxxing rizz
])

model.compile(
    optimizer=optimizers.Adam(1e-5), # type: ignore
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

def preprocess(images, labels):
    return applications.mobilenet_v2.preprocess_input(images), labels

AUTOTUNE = tf.data.AUTOTUNE

processed_train_ds = (
    train_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE) # type: ignore
    .prefetch(AUTOTUNE)
)

processed_val_ds = (
    val_ds
    .map(preprocess, num_parallel_calls=AUTOTUNE) # type: ignore
    .prefetch(AUTOTUNE)
)

model.fit(
    processed_train_ds,
    validation_data=processed_val_ds, # type: ignore
    epochs=15
)

model.save("test.keras")

# esp32 compabile save
convertor = tf.lite.TFLiteConverter.from_keras_model(model)
convertor.optimizations.add(tf.lite.Optimize.DEFAULT)

# do the fitting thing later
with open("model.tflite", "wb") as f:
    f.write(typing.cast(bytes, convertor.convert())) 
