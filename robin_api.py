from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

ORGANIK = "organik"
ANORGANIK = "anorganik"
B3 = "B3"

class_labels_with_types = {
    "buah_sayuran": ORGANIK,
    "daun": ORGANIK,
    "elektronik": B3,
    "kaca": ANORGANIK,
    "kertas": ANORGANIK,
    "logam": ANORGANIK,
    "makanan": ORGANIK,
    "medis": B3,
    "plastik": ANORGANIK,
    "tekstil": ANORGANIK,
}

class_labels = list(class_labels_with_types.keys())

app = FastAPI()
model = load_model('robin_resnet50')  #use tensorflow format (model can be found here https://www.kaggle.com/models/bahiskaraananda/robin-resnet50/tensorFlow2)

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of a waste item from an uploaded image.

    Args:
    - file: An uploaded image file.

    Returns:
    - A dictionary containing:
        - predicted_class: The class label with the highest probability.
        - waste_type: The type of waste (organik, anorganik, B3).
        - probabilities: A list of probabilities for all classes.
    """

    image = Image.open(file.file).resize((224, 224))  #adjust input size
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  #normalize

    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_labels[predicted_index]
    waste_type = class_labels_with_types[predicted_class]

    return {
        "predicted_class": predicted_class,
        "waste_type": waste_type,
        "probabilities": {
            class_labels[i]: float(prob) for i, prob in enumerate(prediction[0])
        }
    }

"""
The prediction will now return a JSON response like this:

{
    "predicted_class": "plastik",
    "waste_type": "anorganik",
    "probabilities": {
        "buah_sayuran": 0.02,
        "daun": 0.03,
        "elektronik": 0.01,
        "kaca": 0.02,
        "kertas": 0.01,
        "logam": 0.02,
        "makanan": 0.04,
        "medis": 0.01,
        "plastik": 0.84,
        "tekstil": 0.00
    }
}
"""
