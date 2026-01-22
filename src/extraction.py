import cv2
import numpy as np
import logging
import src.config as config
import os

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MP_AVAILABLE = True
except (ImportError, AttributeError):
    MP_AVAILABLE = False

class FeatureExtractor:
    def __init__(self, use_static_image_mode=False):
        """
        Initialize MediaPipe Hands and Pose models using the new Task API.
        """
        self.logger = logging.getLogger(__name__)
        self.mp_available = MP_AVAILABLE
        
        self.hand_landmarker = None
        self.pose_landmarker = None
        
        if self.mp_available:
            try:
                # Load Hand Landmarker
                hand_model_path = os.path.abspath("models/mediapipe/hand_landmarker.task")
                if not os.path.exists(hand_model_path):
                    self.logger.error(f"Hand model not found at {hand_model_path}")
                    self.mp_available = False
                else:
                    base_options = python.BaseOptions(model_asset_path=hand_model_path)
                    options = vision.HandLandmarkerOptions(
                        base_options=base_options,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        running_mode=vision.RunningMode.IMAGE # or VIDEO if needed, but IMAGE is safer for loop
                    )
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

                # Load Pose Landmarker
                pose_model_path = os.path.abspath("models/mediapipe/pose_landmarker_full.task")
                if not os.path.exists(pose_model_path):
                     self.logger.error(f"Pose model not found at {pose_model_path}")
                     # Ensure we don't break if only pose is missing?
                else:
                    base_options_pose = python.BaseOptions(model_asset_path=pose_model_path)
                    options_pose = vision.PoseLandmarkerOptions(
                        base_options=base_options_pose,
                        output_segmentation_masks=False,
                        min_pose_detection_confidence=0.5,
                        min_pose_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        running_mode=vision.RunningMode.IMAGE
                    )
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(options_pose)
            
            except Exception as e:
                self.logger.error(f"Failed to initialize MediaPipe tasks: {e}")
                self.mp_available = False

        if not self.mp_available:
            self.logger.warning("MediaPipe not available. Using Dummy Feature Extraction.")

    def extract_landmarks(self, image_rgb):
        """
        Extract hand and pose landmarks. Returns zeros if MP missing.
        """
        results_dict = {}
        
        if not self.mp_available or self.hand_landmarker is None:
            return np.zeros(126), np.zeros(99), {}, [] # Dummy (4 items)
            
        # Convert to MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # --- Process Hands ---
        hand_result = self.hand_landmarker.detect(mp_image)
        # results_dict['hands'] = hand_result # Store object if needed
        
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))
        
        # New API: hand_result.hand_landmarks is list of list of NormalizedLandmark
        # hand_result.handedness is list of list of Category
        
        if hand_result.hand_landmarks:
             for idx, landmarks in enumerate(hand_result.hand_landmarks):
                 # Get handedness
                 # handedness is list of list, but usually 1 Category per hand
                 if idx < len(hand_result.handedness):
                     classification = hand_result.handedness[idx][0]
                     label = classification.category_name # "Left" or "Right"
                     
                     points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                     
                     if label == 'Left':
                         left_hand = points
                     else:
                         right_hand = points

        hand_vec = np.concatenate([left_hand.flatten(), right_hand.flatten()])
        
        # --- Extract Hand Crops ---
        hand_crops = []
        if hand_result.hand_landmarks:
            with open("debug_trace.log", "a") as f: f.write(f"EXTRACT: Found {len(hand_result.hand_landmarks)} hands\n")
            h, w, c = image_rgb.shape
            for landmarks in hand_result.hand_landmarks:
                 x_vals = [lm.x for lm in landmarks]
                 y_vals = [lm.y for lm in landmarks]
                 min_x, max_x = min(x_vals), max(x_vals)
                 min_y, max_y = min(y_vals), max(y_vals)
                 
                 padding_x = 0.05
                 padding_y = 0.05
                 min_x = max(0.0, min_x - padding_x)
                 max_x = min(1.0, max_x + padding_x)
                 min_y = max(0.0, min_y - padding_y)
                 max_y = min(1.0, max_y + padding_y)
                 
                 x1 = int(min_x * w)
                 y1 = int(min_y * h)
                 x2 = int(max_x * w)
                 y2 = int(max_y * h)
                 
                 with open("debug_trace.log", "a") as f: f.write(f"EXTRACT: BBox {x1}:{x2}, {y1}:{y2}\n")
                 
                 if x2 > x1 and y2 > y1:
                     crop = image_rgb[y1:y2, x1:x2]
                     crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                     hand_crops.append(crop_bgr)
                 else:
                     with open("debug_trace.log", "a") as f: f.write(f"EXTRACT: Invalid BBox\n")
        
        with open("debug_trace.log", "a") as f: f.write(f"EXTRACT: Generated {len(hand_crops)} hand crops\n")

        # --- Process Pose ---
        pose_vec = np.zeros((33, 3))
        if self.pose_landmarker:
            pose_result = self.pose_landmarker.detect(mp_image)
            # results_dict['pose'] = pose_result
            
            if pose_result.pose_landmarks:
                # List of list, usually 1 pose
                landmarks = pose_result.pose_landmarks[0]
                pose_vec = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        
        pose_vec = pose_vec.flatten()
        
        return hand_vec, pose_vec, results_dict, hand_crops

    def extract_emotion(self, image_bgr):
        """
        Extract dominant emotion using DeepFace.
        """
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_map = {emo: i for i, emo in enumerate(emotions)}
        
        if not DEEPFACE_AVAILABLE:
            return 6 # Neutral

        try:
            objs = DeepFace.analyze(
                img_path=image_bgr, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if len(objs) > 0:
                dominant_emotion = objs[0]['dominant_emotion']
                return emotion_map.get(dominant_emotion, 6)
            else:
                return 6
        except Exception as e:
            return 6

    def close(self):
        if self.hand_landmarker:
            self.hand_landmarker.close()
        if self.pose_landmarker:
            self.pose_landmarker.close()
