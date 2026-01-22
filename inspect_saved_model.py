import tensorflow as tf
import os

model_path = r"c:\aashish programming files\freelance\nithya-analysis ml model\models\saved\kaggle_model_v3_savedmodel"

print(f"Loading model from {model_path}...")
try:
    model = tf.saved_model.load(model_path)
    print("Model loaded successfully.")
    
    print("Signatures:", list(model.signatures.keys()))
    
    infer = model.signatures["serving_default"]
    print("Input keys:", list(infer.structured_input_signature[1].keys()) if infer.structured_input_signature[1] else "No structured input")
    print("Output keys:", list(infer.structured_outputs.keys()))
    
except Exception as e:
    print(f"Error loading model: {e}")
