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
- Custom layers: `Global Average Pooling, Dropout (0.5), Dense (10 classes, softmax)`
- Input size: `224,224,3`
- Optimizer: `Adam`
- Loss: `Categorical Crossentropy`
- <a href='https://www.kaggle.com/datasets/bahiskaraananda/robin-resnet50' target='_blank'>robin-model</a>
- <a href='https://github.com/Bin-Detective/bindetective-ml/blob/main/robin-resnet50-finetuned.ipynb' target='_blank'>training-documentation</a>


## Requirements
- TensorFlow `2.15`
- Python 3
- Keras
- NumPy
