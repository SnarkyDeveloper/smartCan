import os
import kagglehub as kh
from keras import models, utils, applications

path = os.path.join(kh.dataset_download("techsash/waste-classification-data"), "DATASET", "TEST")

test_ds = utils.image_dataset_from_directory(
    path,
    labels="inferred",
    label_mode="int",
    image_size=(96, 96),
    batch_size=32,
    shuffle=False
) # type: ignore

model = models.load_model("test.keras")
if model != None:
    print("Model loaded successfully")
else:
    print("Model failed to load")
    exit(1)

def preprocess(images, labels):
    return applications.mobilenet_v2.preprocess_input(images), labels

processed_test_ds = test_ds.map(preprocess) # type: ignore

import pytest as _ # start pytest block

def test_model_accuracy():
    print("evaling")
    acc = model.evaluate(processed_test_ds) # type: ignore
    print(f"Model accuracy: {acc[1]}")
    assert acc[1] > 0.8, f"Model accuracy is too low: {acc[1]}"
