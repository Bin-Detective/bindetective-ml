
# `Robin` BinDetective Waste Classification Model


Robin is a model developed for waste classification, categorizing items into 10 distinct classes. The model has been fine-tuned using different architectures to find the best and optimal performance. Below, you’ll find links to specific branches for each model variant.

---

## **Available Models**
Each model branch contains `model information`, `documentation`, `training and validation details`.

| **Variant**         | **Branch**                                       | **Model**                                                                                 |
|-------------------|-------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| **EfficientNetV2S**| [main-efficientnetv2s](https://github.com/Bin-Detective/bindetective-ml/tree/main-efficientnetv2s) | [Kaggle](https://www.kaggle.com/models/bahiskaraananda/robin-efficientnetv2s)   |
| **EfficientNetB0** | [efficientnetb0](https://github.com/Bin-Detective/bindetective-ml/tree/efficientnetb0)             | |
| **ResNet50**       | [mango-resnet50](https://github.com/Bin-Detective/bindetective-ml/tree/mango-resnet50)             | [Kaggle](https://www.kaggle.com/models/bahiskaraananda/robin-resnet50)                 |


## Classes
1. buah_sayuran (Fruits/Vegetables): `Fruit/vegetable waste, peels, seeds`
2. daun (Leaves): `Leaves, flowers, grass, garden waste`
3. elektronik (Electronics): `Phones, pcb, cables, batteries, etc`
4. kaca (Glass): `Glass bottles, broken glass`
5. kertas (Paper): `Newspapers, cardboard, paper waste`
6. logam (Metal): `Cans, forks, spoons`
7. makanan (Food): `Food remnants, expired foods`
8. medis (Medical): `Medical waste, masks, gloves, etc`
9. plastik (Plastic): `Bottles, straw, plastic bags, etc`
10. tekstil (Textile): `Clothes, fabric`

## Datasets
```bash
robin-base/
├── train/
│   ├── buah_sayuran/
│   ├── daun/
│   ├── elektronik/
│   └── ...
├── val/
│   ├── buah_sayuran/
│   ├── daun/
│   ├── elektronik/
│   └── ...
└── test/
    ├── buah_sayuran/
    ├── daun/
    ├── elektronik/
    └── ...
```
The dataset is balanced with a 1:1 ratio for each class and split into `70/20/10` for training, validation, and testing, consisting of 30,000 images in total.
- Training: `20990 images`
- Validation: `6000 images`
- Test: `3010 images`
- <a href='https://www.kaggle.com/datasets/bahiskaraananda/robin-base' target='_blank'>robin-datasets</a>
- <a href='https://github.com/Bin-Detective/bindetective-ml/blob/main/robin-lite-dataset-preparation.ipynb/' target='_blank'>preprocessing-documentation</a>

## Model Details
- Base model: `EfficientNetV2S (pretrained on ImageNet)`
- Frozen Layers: `Initial layers up to block 143 (block4f_bn)`
- Custom layers: `Global Average Pooling, Dropout (0.5), Dense (10 classes, softmax)`
- Input size: `224,224,3`
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
```bash
Total params: 20344170 (77.61 MB)
Trainable params: 18468850 (70.45 MB)
Non-trainable params: 1875320 (7.15 MB)
```
- <a href='https://www.kaggle.com/models/bahiskaraananda/robin-efficientnetv2s' target='_blank'>robin-efficientnetv2s-model</a>
- <a href='https://github.com/Bin-Detective/bindetective-ml/blob/main-efficientnetv2s/robin-efficientnetv2s-finetuned.ipynb' target='_blank'>robin-efficientnetv2s-training-documentation</a>

## Performance
The current `robin_efficientnetv2s` test accuracy and performance metrics for each class are as follows:
```bash
              precision    recall  f1-score   support
buah_sayuran       0.95      0.91      0.93       301
        daun       0.98      0.98      0.98       301
  elektronik       0.99      0.98      0.99       301
        kaca       0.94      0.97      0.95       301
      kertas       0.96      0.96      0.96       301
       logam       0.94      0.97      0.95       301
     makanan       0.91      0.95      0.93       301
       medis       0.97      0.97      0.97       301
     plastik       0.95      0.89      0.92       301
     tekstil       0.99      0.99      0.99       301

    accuracy                           0.96      3010
   macro avg       0.96      0.96      0.96      3010
weighted avg       0.96      0.96      0.96      3010
```
Confusion matrix:

![image](https://github.com/user-attachments/assets/fabef82f-e42a-43af-9a0e-7c9dd1a41bb8)

## Requirements
- TensorFlow `2.15`
- Python `3.x`
- Keras
- NumPy

## Workflow
Here’s a simple explanation of the workflow for the Robin model:
```plaintext
+-------------------+        +-----------------------------+        +-----------------------------+
| Collect Dataset   | -----> | Preprocess Dataset          | -----> | Model Development           |
| (30000+ images)   |        | - Resize images to 224x224  |        | - Select base architecture  |
+-------------------+        | - Normalize pixel values    |        | - Load pretrained weights   |
                             | - Balance each dataset      |        | - Freeze initial layers     |
                             |   class (1:1)               |        | - Add custom layers         |
                             | - Split into Train/Val/Test |        | - Compile with optimizer    |
                             |   (70/20/10 ratio)          |        |   and loss function         |
                             +-----------------------------+        +-----------------------------+
                                                                                    |
                                                                                    v
                             +-----------------------------+        +------------------------------+
                             | Model Evaluation            | <----- | Model Training               |
                             | - Test on 10% dataset       |        | - Train on 70% dataset       |
                             | - Generate performance      |        | - Validate on 20% dataset    |
                             |   metrics                   |        | - Fine-tune                  |
                             +-----------------------------+        | - Save best-performing model |
                                             |                      +------------------------------+
                                             v
                             +--------------------------------+
                             | Troubleshoot                   |
                             | - Adjust hyperparameters       |
                             | - Fine-tune and analyze        |
                             |   performance for improvements |
                             +--------------------------------+
                                             |
                                             v
                             +----------------------------------+
                             | Deploy the Model                 |
                             | - Save the model in TensorFlow   |
                             |   SavedModel format              |
                             | - Deploy the model using FastAPI |
                             |   for waste classification       |
                             +----------------------------------+
```
