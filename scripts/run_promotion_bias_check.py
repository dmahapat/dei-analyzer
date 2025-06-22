import os
from models.promotion_predictor import train_model

# Train model
model, features = train_model()

print("✅ Model trained successfully on features:", features)