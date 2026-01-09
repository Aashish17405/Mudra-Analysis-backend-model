import cv2
import numpy as np
import logging
import src.config as config

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
except (ImportError, AttributeError):
    MP_AVAILABLE = False
    mp_hands = None
    mp_pose = None
    mp_drawing = None

class FeatureExtractor:
    def __init__(self, use_static_image_mode=False):
        """
        Initialize MediaPipe Hands and Pose models.
        """
        self.logger = logging.getLogger(__name__)
        
        if MP_AVAILABLE:
            self.hands = mp_hands.Hands(
                static_image_mode=use_static_image_mode,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.pose = mp_pose.Pose(
                static_image_mode=use_static_image_mode,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.logger.warning("MediaPipe not available. Using Dummy Feature Extraction.")

    def extract_landmarks(self, image_rgb):
        """
        Extract hand and pose landmarks. Returns zeros if MP missing.
        """
        results_dict = {}
        
        if not MP_AVAILABLE:
            # Return dummy vectors
            return np.zeros(126), np.zeros(99), {}
            
        # Process Hands
        hand_results = self.hands.process(image_rgb)
        results_dict['hands'] = hand_results
        
        # Process Pose
        pose_results = self.pose.process(image_rgb)
        results_dict['pose'] = pose_results
        
        # --- Format Hand Data ---
        left_hand = np.zeros((21, 3))
        right_hand = np.zeros((21, 3))
        
        if hand_results.multi_hand_landmarks:
            for idx, hand_handedness in enumerate(hand_results.multi_handedness):
                label = hand_handedness.classification[0].label
                landmarks = hand_results.multi_hand_landmarks[idx]
                points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                if label == 'Left':
                    left_hand = points
                else:
                    right_hand = points
        
        hand_vec = np.concatenate([left_hand.flatten(), right_hand.flatten()])
        
        # --- Format Pose Data ---
        pose_vec = np.zeros((33, 3))
        if pose_results.pose_landmarks:
            pose_vec = np.array([[lm.x, lm.y, lm.z] for lm in pose_results.pose_landmarks.landmark])
            
        pose_vec = pose_vec.flatten()
        
        return hand_vec, pose_vec, results_dict

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
        if MP_AVAILABLE:
            self.hands.close()
            self.pose.close()
