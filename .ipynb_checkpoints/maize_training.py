# ======================
# 1. COMPLETE TRAINING PIPELINE
# ======================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import joblib
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                           confusion_matrix, ConfusionMatrixDisplay,
                           precision_score, recall_score, f1_score)
import optuna

# Download dataset (Colab/Kaggle)
# !mkdir -p data
# if not os.path.exists('data/maize'):
#     !wget -q https://data.mendeley.com/public-files/datasets/4drtyfjtfy/files/9b72ae5a-1a8e-46f8-9a3a-735417be2b37/file_downloaded -O maize.zip
#     !unzip -q maize.zip -d data/
#     !rm maize.zip

# Load and preprocess data
data_dir = 'data/maize'
class_names = ['Common_Rust', 'Gray_Leaf_Spot', 'Northern_Leaf_Blight', 'Healthy']

X_paths = []
y = []
for class_idx, class_name in enumerate(class_names):
    class_dir = os.path.join(data_dir, class_name)
    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)
        X_paths.append(img_path)
        y.append(class_idx)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_paths, y, test_size=0.2, random_state=42, stratify=y)

# Feature extraction with ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_paths):
    features = []
    for img_path in tqdm(img_paths, desc="Extracting features"):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feature = base_model.predict(x, verbose=0)
        features.append(feature.flatten())
    return np.array(features)

print("Extracting training features...")
X_train_features = extract_features(X_train)
print("Extracting test features...")
X_test_features = extract_features(X_test)

# Optuna optimization
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'class_weight': 'balanced',
        'random_state': 42
    }
    model = RandomForestClassifier(**params)
    return f1_score(y_train, model.fit(X_train_features, y_train).predict(X_train_features), average='macro')

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Train final model
best_params = study.best_params
best_params['class_weight'] = 'balanced'
final_model = RandomForestClassifier(**best_params)
final_model.fit(X_train_features, y_train)

# Evaluation
y_pred = final_model.predict(X_test_features)
print(classification_report(y_test, y_pred, target_names=class_names))

# ======================
# 2. SAVE MODELS & FEATURES
# ======================
import pickle

# Save models
joblib.dump(final_model, 'maize_rf_model.pkl')

# Save feature extractor (simplified)
with open('feature_extractor.pkl', 'wb') as f:
    pickle.dump(base_model, f)

# Save class names
with open('class_names.pkl', 'wb') as f:
    pickle.dump(class_names, f)