"""
Mudra Predictor for Real-time Inference
Integrates with existing video inference pipeline.
"""
import logging
import numpy as np
import cv2
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from src import config
from src.dataset_loader import load_class_mappings
from src.extraction import FeatureExtractor

logger = logging.getLogger(__name__)



class SafeInputLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        print(f"DEBUG: SafeInputLayer init. Name in kwargs: {kwargs.get('name')}")
        # Clean clean clean to avoid Keras 3 incompatibility
        if 'batch_shape' in kwargs:
             kwargs.pop('batch_shape')
        if 'batch_input_shape' in kwargs:
             kwargs.pop('batch_input_shape')
        
        # Force hardcoded input shape for our specific model
        kwargs['input_shape'] = (224, 224, 3)
        super().__init__(**kwargs)
        print(f"DEBUG: SafeInputLayer created with name: {self.name}")

class SafeRescaling(keras.layers.Rescaling):
    def __init__(self, scale, offset=0.0, **kwargs):
        print(f"DEBUG: SafeRescaling init. Name in kwargs: {kwargs.get('name')}")
        # Always remove dtype
        if 'dtype' in kwargs:
            kwargs.pop('dtype') 
        super().__init__(scale, offset=offset, **kwargs)
        print(f"DEBUG: SafeRescaling created with name: {self.name}")

class SafeNormalization(keras.layers.Normalization):
    def __init__(self, axis=-1, mean=None, variance=None, invert=False, **kwargs):
        print(f"DEBUG: SafeNormalization init. Name in kwargs: {kwargs.get('name')}")
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
        
        # Sanitize axis if it's a list [3] -> 3
        if isinstance(axis, list) and len(axis) == 1:
            axis = axis[0]
            
        super().__init__(axis=axis, mean=mean, variance=variance, invert=invert, **kwargs)
        print(f"DEBUG: SafeNormalization created with name: {self.name}")

# Mock DTypePolicy class if Keras tries to instantiate it
class DTypePolicy:
    def __init__(self, *args, **kwargs):
        self.name = 'float32'
        self.compute_dtype = 'float32'
        self.variable_dtype = 'float32'

class MudraPredictor:
    """Real-time mudra prediction from hand landmarks."""
    
    def __init__(self, model_path=None, model_type='hybrid', extractor=None):
        try:
            k_ver = getattr(keras, '__version__', 'Unknown')
        except: k_ver = 'Unknown'
        logger.info(f"TF Version: {tf.__version__}, Keras Version: {k_ver}")
        self.model_type = model_type
        self.is_rf_model = False
        self.input_shape = (224, 224) # Default Hybrid/Image
        self.idx_to_class = {}
        self.class_to_idx = {}
        
        # Load model - Priority: Kaggle v3 (Keras) -> Kaggle v3 (H5) -> Kaggle v2 -> ...
        if model_path is None:
            cnn_path = Path(config.MODEL_DIR) / 'mudra_cnn_model.h5'
            kaggle_v3_keras = Path(config.MODEL_DIR) / 'kaggle_model_v3.keras'
            kaggle_v3_h5 = Path(config.MODEL_DIR) / 'kaggle_model_v3.h5'
            kaggle_v2 = Path(config.MODEL_DIR) / 'kaggle_model_v2.h5'
            kaggle_latest = Path(config.MODEL_DIR) / 'mudra_cnn_model_kaggle_latest.h5'
            kaggle_path = Path(config.MODEL_DIR) / 'mudra_cnn_model_kaggle.h5'
            rf_path = Path(config.MODEL_DIR) / 'mudra_rf_model.pkl'
            h5_path = Path(config.MODEL_DIR) / f'mudra_classifier_{model_type}_final.h5'
            
            # Priority Logic
            if (Path(config.MODEL_DIR) / 'kaggle_model_v3_savedmodel').exists():
                model_path = Path(config.MODEL_DIR) / 'kaggle_model_v3_savedmodel'
                logger.info(f"Found Kaggle v3 SavedModel at {model_path}")
            elif kaggle_v3_keras.exists():
                model_path = kaggle_v3_keras
                logger.info(f"Found Kaggle v3 Keras model at {model_path}")
            elif kaggle_v3_h5.exists():
                model_path = kaggle_v3_h5
                logger.info(f"Found Kaggle v3 H5 model at {model_path}")
            elif kaggle_v2.exists():
                model_path = kaggle_v2
                logger.info(f"Found Kaggle v2 CNN model at {model_path}")
            elif kaggle_latest.exists():
                model_path = kaggle_latest
                logger.info(f"Found Kaggle Latest model at {model_path}")
            elif kaggle_path.exists():
                model_path = kaggle_path
                logger.info(f"Found Kaggle CNN model at {model_path}")
            elif cnn_path.exists():
                model_path = cnn_path
                logger.info(f"Found Local CNN model at {model_path}")
            elif rf_path.exists():
                model_path = rf_path
                self.is_rf_model = True
                logger.info(f"Found Random Forest model at {model_path}")
            elif h5_path.exists():
                model_path = h5_path
                logger.info(f"Found Legacy H5 model at {model_path}")
            else:
                raise FileNotFoundError("No mudra model found in models/saved/")
        
        self.load_model(model_path)

        # Auto-detect input shape from loaded model
        if self.model and hasattr(self.model, 'input_shape'):
            shape = self.model.input_shape
            if shape and len(shape) == 4: # (None, H, W, C)
                self.input_shape = (shape[1], shape[2])
                logger.info(f"Model input shape detected: {self.input_shape}")
        elif self.model and hasattr(self.model, 'signatures') and "serving_default" in self.model.signatures:
            # For SavedModel, try to get input shape from signature
            try:
                input_tensor_spec = self.model.signatures["serving_default"].structured_input_signature[1]['keras_tensor']
                if input_tensor_spec.shape and len(input_tensor_spec.shape) == 4:
                    self.input_shape = (input_tensor_spec.shape[1], input_tensor_spec.shape[2])
                    logger.info(f"SavedModel input shape detected: {self.input_shape}")
            except Exception as e:
                logger.warning(f"Could not determine SavedModel input shape: {e}")
        
        # Load class mappings if not established by reconstruction
        if not self.idx_to_class: # Check if it's still empty
            try:
                mapping = load_class_mappings()
                self.num_classes = mapping['num_classes']
                self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
                self.class_to_idx = mapping['class_to_idx']
                logger.info(f"Loaded {self.num_classes} classes from class mappings.")
            except FileNotFoundError:
                logger.warning("Class mapping not found. Using default mapping.")
                self.num_classes = 50
                self.idx_to_class = {i: f"Mudra_{i}" for i in range(self.num_classes)}
                self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        
        # Feature extractor
        if extractor is None:
            self.extractor = FeatureExtractor()
        else:
            self.extractor = extractor
            
    def load_model(self, model_path):
        if self.is_rf_model:
            import joblib
            self.model = joblib.load(model_path)
            logger.info("Random Forest model loaded successfully.")
            return

        model_path = str(model_path)
        logger.info(f"Loading mudra model from {model_path}")
        
        # Check if it is a SavedModel directory (Kaggle Export)
        if os.path.isdir(model_path):
            try:
                logger.info("Detected SavedModel directory. utilizing tf.saved_model.load (Robust Mode)...")
                self.model = tf.saved_model.load(model_path)
                self.inference_func = self.model.signatures["serving_default"]
                self.is_saved_model = True
                logger.info("SavedModel loaded successfully via low-level API.")
                return
            except Exception as e:
                logger.error(f"Failed to load SavedModel: {e}")
                
        # Standard .keras or .h5 loading
        self.is_saved_model = False
        try:
            logger.info("Attempting standard keras.models.load_model...")
            self.model = keras.models.load_model(model_path, compile=False)
            logger.info("Standard load successful.")
        except Exception as e:
            logger.warning(f"Standard load failed: {e}. Attempting robust model reconstruction...")
            self.model, loaded_classes = self.reconstruct_and_load_model(model_path)
            if not self.idx_to_class: 
                 pass

    def reconstruct_and_load_model(self, model_path):
        """
        Manually reconstruct EfficientNetB0 architecture and load weights 
        to bypass Keras config deserialization issues.
        """
        logger.info("Attempting robust model reconstruction...")
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model
        
        # Consistent 47 classes (All 50 from YAML minus Tripathaka which had 0 samples)
        # Assuming sorted order as per flow_from_dataframe
        # Use the loaded class names if available, otherwise fallback
        if hasattr(self, 'idx_to_class') and self.idx_to_class:
            effective_classes = [self.idx_to_class[i] for i in sorted(self.idx_to_class.keys())]
            logger.info(f"Using {len(effective_classes)} classes from loaded JSON for reconstruction.")
        else:
            # Fallback to hardcoded list (legacy)
            yaml_names = ['Alapadmam', 'Anjali', 'Aralam', 'Ardhachandran', 'Ardhapathaka', 'Berunda', 'Bramaram', 
                          'Chakra', 'Chandrakala', 'Chaturam', 'Garuda', 'Hamsapaksha', 'Hamsasyam', 'Kangulam', 
                          'Kapith', 'Kapotham', 'Karkatta', 'Kartariswastika', 'Katakamukha', 'Katakavardhana', 
                          'Katrimukha', 'Khatva', 'Kilaka', 'Kurma', 'Matsya', 'Mayura', 'Mrigasirsha', 'Mukulam', 
                          'Mushti', 'Nagabandha', 'Padmakosha', 'Pasha', 'Pathaka', 'Pushpaputa', 'Sakata', 
                          'Samputa', 'Sarpasirsha', 'Shanka', 'Shivalinga', 'Shukatundam', 'Sikharam', 
                          'Simhamukham', 'Suchi', 'Swastikam', 'Tamarachudam', 'Tripathaka', 'Trishulam', 'Varaha']
            effective_classes = sorted([n for n in yaml_names if n != 'Tripathaka'])
        
        num_classes = len(effective_classes)
        
        # Note: Training used ImageDataGenerator(rescale=1./255) AND EfficientNet built-in rescaling
        # So the model expects tiny inputs [0, 1/255] if the built-in rescaling is active?
        # Or if built-in rescaling expects [0, 255]?
        # We will verify this in inference by trying both scales.
        
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        logger.info(f"Reconstructed architecture with {num_classes} output classes.")
        
        try:
             model.load_weights(str(model_path))
             logger.info("Weights loaded successfully.")
        except ValueError as e:
             logger.warning(f"Standard loading failed: {e}. Trying by_name...")
             model.load_weights(str(model_path), by_name=True, skip_mismatch=True)
        
        return model, effective_classes
    
    def preprocess_landmarks(self, landmarks):
        """Normalize landmarks (same as training)."""
        if landmarks is None:
            return None
        if np.all(landmarks == 0):
            return landmarks
        
        normalized = landmarks.copy()
        for hand_idx in range(2):
            offset = hand_idx * 63
            hand_landmarks = landmarks[offset:offset + 63]
            if np.all(hand_landmarks == 0): continue
            
            wrist_x, wrist_y, wrist_z = hand_landmarks[0], hand_landmarks[1], hand_landmarks[2]
            for i in range(21):
                normalized[offset + i*3] -= wrist_x
                normalized[offset + i*3 + 1] -= wrist_y
                normalized[offset + i*3 + 2] -= wrist_z
            
            hand_span = np.max(np.abs(normalized[offset:offset + 63]))
            if hand_span > 0:
                normalized[offset:offset + 63] /= hand_span
        return normalized
    
    def predict_mudra(self, hand_landmarks, image=None, top_k=3):
        with open("debug_trace.log", "a") as f: f.write(f"ENTRY: hand_landmarks type: {type(hand_landmarks)}\n")
        try:
             with open("debug_trace.log", "a") as f: f.write(f"ENTRY: shape: {hand_landmarks.shape}\n")
        except: pass

        # CNN Inference (Auto-detected 32x32 or 224x224 input)
        # Use self.input_shape instead of model.input_shape (SavedModel doesn't have it)
        if hasattr(self, 'input_shape'):
            
            # Check if input is already a crop (3D array)
            is_pre_cropped = False
            crop_success = False
            try:
                 with open("debug_trace.log", "a") as f: f.write(f"CHECK: ndim={hand_landmarks.ndim}\n")
            except: pass
            
            if isinstance(hand_landmarks, np.ndarray) and hand_landmarks.ndim == 3:
                with open("debug_trace.log", "a") as f: f.write("BRANCH: IF (Pre-cropped)\n")
                is_pre_cropped = True
                crop_success = True
                hand_crop = hand_landmarks
                with open("debug_trace.log", "a") as f: f.write(f"PREDICTOR: Pre-cropped input detected {hand_crop.shape}\n")
            else:
                with open("debug_trace.log", "a") as f: f.write("BRANCH: ELSE (Not Pre-cropped)\n")
                if image is None: 
                    logger.warning("Predict Mudra: Image is None!")
                    return [("Unknown", 0.0)]
                hand_crop = image
                
            target_h, target_w = self.input_shape
            
            # --- Hand Cropping Logic (IMPROVED: Center-Crop) ---
            # Only run if not pre-cropped
            if not is_pre_cropped:
                try:
                    h, w = image.shape[:2]
                    landmarks_all = hand_landmarks.reshape(-1, 3) # (42, 3)
                    valid_points = []
                    for pt in landmarks_all:
                         # Check if point is valid (not all zeros)
                         if np.all(pt == 0): continue
                         # Denormalize
                         px = int(pt[0] * w)
                         py = int(pt[1] * h)
                         valid_points.append([px, py])
                    
                    if valid_points:
                        valid_points = np.array(valid_points)
                        x_min = np.min(valid_points[:, 0])
                        y_min = np.min(valid_points[:, 1])
                        x_max = np.max(valid_points[:, 0])
                        y_max = np.max(valid_points[:, 1])
                        
                        # Calculate center and dimensions
                        center_x = (x_min + x_max) // 2
                        center_y = (y_min + y_max) // 2
                        
                        # Define square crop size (largest dimension + 2x margin for context)
                        hand_width = x_max - x_min
                        hand_height = y_max - y_min
                        margin = max(hand_width, hand_height) // 2  # 50% margin on each side
                        crop_size = max(hand_width, hand_height) + (2 * margin)
                        
                        # Extract square region centered on hand
                        half_size = crop_size // 2
                        x_start = max(0, center_x - half_size)
                        y_start = max(0, center_y - half_size)
                        x_end = min(w, center_x + half_size)
                        y_end = min(h, center_y + half_size)
                        
                        # Only crop if we have a valid region
                        if x_end > x_start and y_end > y_start:
                            hand_crop = image[y_start:y_end, x_start:x_end]
                            crop_success = True
                            
                            # If crop is not perfectly square due to edge constraints, pad minimally
                            crop_h, crop_w = hand_crop.shape[:2]
                            if crop_h != crop_w:
                                max_dim = max(crop_h, crop_w)
                                delta_w = max_dim - crop_w
                                delta_h = max_dim - crop_h
                                top = delta_h // 2
                                bottom = delta_h - top
                                left = delta_w // 2
                                right = delta_w - left
                                hand_crop = cv2.copyMakeBorder(
                                    hand_crop, top, bottom, left, right,
                                    cv2.BORDER_REPLICATE  # Replicate edge pixels instead of black
                                )
                except Exception as e:
                    logger.warning(f"Cropping failed: {e}")
                
                # Fallback: If cropping failed and landmarks are invalid, return Unknown
                if not crop_success:
                    if np.all(hand_landmarks == 0):
                        return [("Unknown", 0.0)]
                    # If we have landmarks but crop failed, use full frame as last resort
                    hand_crop = image
            if hand_crop is None or hand_crop.size == 0:
                return []

            try:
                target_h, target_w = self.input_shape
                
                # Ensure RGB (cv2 loads BGR)
                if len(hand_crop.shape) == 3 and hand_crop.shape[2] == 3:
                    img_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb = hand_crop # If hand_crop is not 3-channel, assume it's already RGB or grayscale
                
                img_resized = cv2.resize(img_rgb, (target_w, target_h))
            
                # NOTE: The SavedModel likely includes a Rescaling(1./255) layer.
                # Passing raw [0, 255] values to avoid double rescaling.
                img_norm = img_resized.astype('float32')
                
                # Expand dims for batch: (1, 224, 224, 3)
                input_data = np.expand_dims(img_norm, axis=0) 
                
                if getattr(self, 'is_saved_model', False):
                    # Robust Inference for SavedModel directory
                    import tensorflow as tf
                    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)
                    # Signature key found via inspection: 'keras_tensor'
                    preds_dict = self.inference_func(keras_tensor=input_tensor)
                    # Output key found via inspection: 'output_0'
                    predictions = preds_dict['output_0'].numpy()[0]
                elif self.is_rf_model:
                    # RF expects flattened input (if trained on features) or specific formatting
                    # Assuming fallback logic or RF not primary focus now
                    predictions = self.model.predict_proba(input_data.reshape(1, -1))[0]
                else:
                    # Standard Keras Model
                    predictions = self.model.predict(input_data, verbose=0)[0]
                
                # --- DEBUG: Print Top Preds ---
                top_3_debug = np.argsort(predictions)[-3:][::-1]
                logger.info(f"DEBUG Preds: {[f'{self.idx_to_class.get(i)}: {predictions[i]:.4f}' for i in top_3_debug]}")
                
                top_indices = np.argsort(predictions)[-top_k:][::-1]
                return [(self.idx_to_class.get(i, f"Unknown_{i}"), float(predictions[i])) for i in top_indices]
            except Exception as e:
                with open("debug_trace.log", "a") as f: f.write(f"PREDICTOR: CRITICAL ERROR: {e}\n")
                logger.error(f"Prediction error: {e}")
                print(f"CRITICAL PREDICTION ERROR: {e}") 
                return []
        else:
             # Debug why hasattr failed
             try:
                 logger.warning(f"CNN skipped. model type: {type(self.model)}")
                 logger.warning(f"Has input_shape? {hasattr(self.model, 'input_shape')}")
             except: pass

        # Legacy/Hybrid Inference (Fall-through)
        landmarks_norm = self.preprocess_landmarks(hand_landmarks)
        if landmarks_norm is None:
             return [("Unknown", 0.0)]
             
        if self.model_type == 'hybrid':
             image_resized = cv2.resize(image, config.MUDRA_IMAGE_SIZE) / 255.0
             input_data = {'landmarks': np.expand_dims(landmarks_norm, axis=0), 'images': np.expand_dims(image_resized, axis=0)}
        elif self.model_type == 'landmark':
             input_data = np.expand_dims(landmarks_norm, axis=0)
        
        predictions = self.model.predict(input_data, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.idx_to_class.get(idx, f"Unknown_{idx}")
            confidence = float(predictions[idx])
            results.append((class_name, confidence))
        
        return results
    
    def predict_from_frame(self, frame, top_k=3):
        """
        Extract landmarks from frame and predict mudra.
        
        Args:
            frame: OpenCV BGR image
            top_k: Return top-k predictions
        
        Returns:
            List of (class_name, confidence) tuples
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks
        hand_landmarks, _, _ = self.extractor.extract_landmarks(image_rgb)
        
        # Predict
        return self.predict_mudra(hand_landmarks, frame, top_k=top_k)
    
    def close(self):
        """Clean up resources."""
        self.extractor.close()


def test_predictor():
    """Test mudra predictor."""
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    
    predictor = MudraPredictor(model_type='landmark')
    
    # Test with dummy data
    dummy_landmarks = np.random.randn(126)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    results = predictor.predict_mudra(dummy_landmarks, dummy_image)
    
    logger.info("Test Predictions:")
    for class_name, confidence in results:
        logger.info(f"  {class_name}: {confidence:.4f}")
    
    predictor.close()


if __name__ == "__main__":
    test_predictor()
