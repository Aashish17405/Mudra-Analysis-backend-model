import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Raw Data Paths
RAW_MUDRAS_DIR = os.path.join(DATA_DIR, 'raw', 'mudras')
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, 'raw', 'videos')

# Processed Data Paths
PROCESSED_MUDRAS_DIR = os.path.join(DATA_DIR, 'processed', 'mudra_frame_dataset')
PROCESSED_SEQUENCES_DIR = os.path.join(DATA_DIR, 'processed', 'step_sequence_dataset')

# Mudra Dataset Paths
RAW_MUDRAS_KAGGLE_DIR = os.path.join(DATA_DIR, 'raw', 'mudras', 'kaggle_50_mudras')
RAW_MUDRAS_ASAMYUKTHA_DIR = os.path.join(DATA_DIR, 'raw', 'mudras', 'asamyuktha_27')
RAW_MUDRAS_BARATH_DIR = os.path.join(DATA_DIR, 'raw', 'mudras', 'barath_mudras')

# Processed Mudra Features
PROCESSED_MUDRA_FEATURES = os.path.join(DATA_DIR, 'processed', 'mudra_features')

# Model Paths
MODEL_DIR = os.path.join(BASE_DIR, 'models', 'saved')
MUDRA_MODEL_PATH = os.path.join(MODEL_DIR, 'mudra_classifier.h5')
STEP_MODEL_PATH = os.path.join(MODEL_DIR, 'step_sequence_model.h5')

# Create directories if they don't exist
os.makedirs(PROCESSED_MUDRAS_DIR, exist_ok=True)
os.makedirs(PROCESSED_SEQUENCES_DIR, exist_ok=True)
os.makedirs(PROCESSED_MUDRA_FEATURES, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# Feature Extraction Constants
N_HAND_LANDMARKS = 21
N_POSE_LANDMARKS = 33
LANDMARK_DIM = 2 # x, y (we might want z, but usually x,y is enough for 2D video unless specified otherwise. User said "126" for hand which is 21*3*2 implies 21 points, 3 coords? Wait. 
# User said: "hand_landmarks (21 × 3 × 2 = 126)"
# 21 points * 2 hands * 3 coords (x,y,z)? OR 21 points * 3 coords * 2 (hands)?
# MediaPipe Hands returns 21 landmarks. 
# If we have 2 hands, that equals 42 landmarks. 
# 42 * 3 (x,y,z) = 126. Correct.
# So we need both Left and Right hands.

FEATURES_HAND_DIM = 126 # 21 * 3 (x,y,z) * 2 (hands)
FEATURES_POSE_DIM = 99  # 33 * 3 (x,y,z)
FEATURES_EMOTION_DIM = 1 # Just the ID or probability vector? User said "emotion_id (int)" 
# BUT "All learning must be done on numerical landmark vectors". 
# An ID is numerical but categorical. Usually one-hot or embedding is better for DL. 
# For now, we will store what the user asked: "emotion_id (int)".

# Sequence Generation Constants
SEQUENCE_LENGTH = 30     # frames (approx 1 sec at 30fps)
STRIDE = 10              # frames sliding window
FPS_TARGET = 30          # Normalize videos to this FPS
INFERENCE_INTERVAL_SECONDS = 3.0 # Process every 3 seconds for uniformity

# Mudra Model Training Constants
MUDRA_IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
EARLY_STOPPING_PATIENCE = 15
REDUCE_LR_PATIENCE = 7

# Logging
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

