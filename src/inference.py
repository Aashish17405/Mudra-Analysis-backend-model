import cv2
import numpy as np
import json
import os
import logging
from tqdm import tqdm
from src import config
from src.extraction import FeatureExtractor
from src.narrative import STEP_MEANINGS

logger = logging.getLogger(__name__)

class StepPredictor:
    def __init__(self):
        # In a real scenario, we would load a trained model here.
        # self.model = keras.models.load_model('...')
        self.step_classes = ["Alarippu", "Jathiswaram", "Shabdam", "Varnam", "Padam", "Tillana"]
        pass

    def predict_sequence(self, feature_sequence, frame_idx=0):
        """
        Mock prediction.
        """
        # If features are all zero (dummy), use time-based cycling to show capabilities
        # If landmarks are all zero (dummy), use time-based cycling
        # Ignore last channel (emotion) which might be non-zero (neutral=6)
        landmarks_only = feature_sequence[:, :-1]
        if np.all(landmarks_only == 0):
            # Change step every 45 frames (~1.5 sec) to show granular timeline
            seg_idx = (frame_idx // 45) % len(self.step_classes)
            return self.step_classes[seg_idx]
            
        val = np.mean(feature_sequence)
        idx = int((val * 1000) % len(self.step_classes))
        return self.step_classes[idx]

def run_inference(video_path, output_json_path):
    """
    Process video, extract features, and infer steps.
    """
    logger.info(f"Running inference on {video_path}")
    
    extractor = FeatureExtractor(use_static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will slide a window over the video
    window_size = config.SEQUENCE_LENGTH
    stride = config.STRIDE
    
    feature_buffer = []
    
    current_step = None
    start_frame = 0
    predictions = []
    
    # Process frame by frame
    for frame_idx in tqdm(range(total_frames), desc="Inference"):
        ret, frame = cap.read()
        if not ret:
            break
            
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Extract
        hand, pose, _ = extractor.extract_landmarks(image_rgb)
        emotion = extractor.extract_emotion(frame)
        
        feature_vec = np.concatenate([hand, pose, [emotion]])
        feature_buffer.append(feature_vec)
        
        # 2. Predict if window full
        if len(feature_buffer) >= window_size:
            # Take last window
            window = np.array(feature_buffer[-window_size:])
            
            # Predict
            # For this MVP, we need a simpler logic: 
            # If we predict X, does it trigger a new segment?
            predictor = StepPredictor()
            pred_label = predictor.predict_sequence(window, frame_idx)
            
            # 3. Aggregate into timeline
            # If label changes, close previous segment
            if pred_label != current_step:
                if current_step is not None:
                    predictions.append({
                        "step": current_step,
                        "start_frame": start_frame,
                        "end_frame": frame_idx,
                        "meaning": STEP_MEANINGS.get(current_step, "A traditional dance step.")
                    })
                
                current_step = pred_label
                start_frame = frame_idx
            
            # Slide: Removing from buffer is one way, or just appending.
            # To emulate sliding window properly, we keep appending and just slice.
            # But memory buffer will grow. Better to pop if using stride?
            # Actually, `feature_buffer` should be a rolling buffer.
            # But here we are iterating frame by frame.
            
    # Close final segment
    if current_step is not None:
         predictions.append({
            "step": current_step,
            "start_frame": start_frame,
            "end_frame": total_frames,
            "meaning": STEP_MEANINGS.get(current_step, "A traditional dance step.")
        })
        
    cap.release()
    extractor.close()
    
    # 4. Save
    with open(output_json_path, 'w') as f:
        json.dump(predictions, f, indent=4)
        
    logger.info(f"Inference complete. Saved to {output_json_path}")
    return predictions
