# `Robin` BinDetective Waste Classification Model

## Overview
Robin is a fine-tuned ResNet50 model developed for waste classification, categorizing items into 10 distinct classes.

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
Split ratio `70/20/10`
- Training: `20990 images`
- Validation: `6000 images`
- Test: `3010 images`
- <a href='https://www.kaggle.com/datasets/bahiskaraananda/robin-base' target='_blank'>robin-datasets</a>
- <a href='https://github.com/Bin-Detective/bindetective-ml/blob/main/robin-lite-dataset-preparation.ipynb/' target='_blank'>preprocessing-documentation</a>

## Model
- Base model: `ResNet50 (pretrained on ImageNet)`
- Frozen Layers: `Initial layers up to block 143 (conv5_block1_1_conv)`
- Custom layers: `Global Average Pooling, Dropout (0.5), Dense (10 classes, softmax)`
- Input size: `224,224,3`
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
```bash
Total Parameters: 23,608,202 (90.06 MB)
Trainable Parameters: 14,996,490 (57.21 MB)
Non-Trainable Parameters: 8,611,712 (32.85 MB)
```
- <a href='https://www.kaggle.com/models/bahiskaraananda/robin-resnet50' target='_blank'>robin-model</a>
- <a href='https://github.com/Bin-Detective/bindetective-ml/blob/main/robin-resnet50-finetuned.ipynb' target='_blank'>training-documentation</a>

## Current Performance
The current model test accuracy and performance metrics for each class are as follows:
```bash
              precision    recall  f1-score   support
buah_sayuran       0.89      0.89      0.89       301
        daun       0.96      0.96      0.96       301
  elektronik       0.99      0.95      0.97       301
        kaca       0.90      0.94      0.92       301
      kertas       0.94      0.93      0.93       301
       logam       0.91      0.93      0.92       301
     makanan       0.90      0.93      0.92       301
       medis       0.93      0.93      0.93       301
     plastik       0.88      0.87      0.87       301
     tekstil       0.99      0.97      0.98       301

    accuracy                           0.93      3010
   macro avg       0.93      0.93      0.93      3010
weighted avg       0.93      0.93      0.93      3010
```
Confusion matrix:

![image](https://github.com/user-attachments/assets/1f82f2ef-1c2d-46be-b718-af111d6b345e)

## Requirements
- TensorFlow `2.15`
- Python `3.11+`
- Keras
- NumPy
