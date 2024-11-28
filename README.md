# Crop Quality Detection and Classification

This project aims to build a deep learning model that classifies images of crops (fruits and vegetables) and predicts their quality based on pre-defined categories. The model is trained to classify images into different crop categories and to predict the quality of each crop. The project leverages a Convolutional Neural Network (CNN) for classification and prediction tasks.

## Project Overview

The model is capable of:
- Classifying crop images into categories such as Mirchi, Mango, Lemon, Papaya, Potato, Tomato, and Banana.
- Predicting the quality of each crop based on three quality labels: 90, 70, and 60.
- Handling images of unknown objects and non-crop images by returning an "Unknown" category.

## Dataset

The dataset consists of images of different fruits and vegetables categorized by type and quality. Each category contains images labeled with the following quality levels:
- **90**: High quality
- **70**: Medium quality
- **60**: Low quality

### Directory Structure

dataset/ ├── Mirchi/ ├── Mango/ ├── Lemon/ ├── Papaya/ ├── Potato/ ├── Tomato/ ├── Banana/ ├── 90/ ├── 70/ ├── 60/


