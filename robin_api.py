from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()
model = load_model('robin_resnet50') #use the robin tensorflow format (model can be found here https://www.kaggle.com/models/bahiskaraananda/robin-resnet50/tensorFlow2)

class_labels = [
    "buah_sayuran",
    "daun",
    "elektronik",
    "kaca",
    "kertas",
    "logam",
    "makanan",
    "medis",
    "plastik",
    "tekstil"
]

@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    """
    Predict the class of a waste item from an uploaded image.

    Args:
    - file: An uploaded image file.

    Returns:
    - A dictionary containing:
        - predicted_class: The class label with the highest probability.
        - probabilities: A list of probabilities for all classes.
    """

    image = Image.open(file.file).resize((224, 224))  #adjust input size
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  #normalize

    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_labels[predicted_index]

    return {
        "predicted_class": predicted_class,
        "probabilities": {class_labels[i]: float(prob) for i, prob in enumerate(prediction[0])}
    }

"""
The prediction will return a JSON response like this:

{
    "predicted_class": "plastik",
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
