import os
import cv2
import json
import numpy as np
import glob
from tqdm import tqdm
import logging
from src.extraction import FeatureExtractor
from src import config

logger = logging.getLogger(__name__)

def process_mudra_images(extractor: FeatureExtractor):
    """
    Iterates through mudra image folders, extracts landmarks, and saves dataset.
    Assumes directory structure: data/raw/mudras/<class_name>/<image_files>
    """
    logger.info("Starting Mudra Image Processing...")
    
    X_list = []
    y_list = []
    
    # Get all class folders
    class_folders = glob.glob(os.path.join(config.RAW_MUDRAS_DIR, '*'))
    classes = [os.path.basename(f) for f in class_folders if os.path.isdir(f)]
    
    if not classes:
        logger.warning(f"No class folders found in {config.RAW_MUDRAS_DIR}. Skipping.")
        return

    # Create a mapping for labels
    class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(classes))}
    
    # Save label mapping
    mapping_path = os.path.join(config.PROCESSED_MUDRAS_DIR, 'label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_to_idx, f, indent=4)
        
    for cls_name in classes:
        cls_idx = class_to_idx[cls_name]
        folder_path = os.path.join(config.RAW_MUDRAS_DIR, cls_name)
        images = glob.glob(os.path.join(folder_path, '*')) # all files
        
        logger.info(f"Processing class '{cls_name}' ({len(images)} images)")
        
        for img_path in tqdm(images, desc=cls_name):
            image = cv2.imread(img_path)
            if image is None:
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract features (Only Hands needed for Mudra Dataset as per request)
            # "Mudra Image Datasets ... used exclusively for: Hand-landmark -> mudra classification"
            hand_vec, _, _ = extractor.extract_landmarks(image_rgb)
            
            # If hands are not detected, hand_vec might be all zeros. 
            # We can decide to skip or keep. 
            # Usually better to skip empty samples for training.
            if np.all(hand_vec == 0):
                continue
                
            X_list.append(hand_vec)
            y_list.append(cls_idx)
            
    if X_list:
        X = np.array(X_list)
        y = np.array(y_list)
        
        np.save(os.path.join(config.PROCESSED_MUDRAS_DIR, 'X_landmarks.npy'), X)
        np.save(os.path.join(config.PROCESSED_MUDRAS_DIR, 'y_mudra_labels.npy'), y)
        logger.info(f"Saved Mudra Dataset: X={X.shape}, y={y.shape}")
    else:
        logger.warning("No valid mudra data processed.")

def process_video_sequences(extractor: FeatureExtractor):
    """
    Iterates through videos, reads annotations, extracts sequences, and saves dataset.
    Assumes structure: data/raw/videos/video1.mp4 and video1.json
    """
    logger.info("Starting Video Sequence Processing...")
    
    X_seq_list = []
    y_seq_list = []
    
    video_files = glob.glob(os.path.join(config.RAW_VIDEOS_DIR, '*.mp4')) + \
                  glob.glob(os.path.join(config.RAW_VIDEOS_DIR, '*.avi')) + \
                  glob.glob(os.path.join(config.RAW_VIDEOS_DIR, '*.mov'))
                  
    if not video_files:
        logger.warning(f"No video files found in {config.RAW_VIDEOS_DIR}. Skipping.")
        return

    # Use a dynamic label map for steps found in JSONs
    step_labels = set()
    
    # First pass: collect all labels (optional, but good for consistent mapping)
    # Or just build it on the fly. Let's build a set first to alphabetize.
    temp_annotations = []
    for vid_path in video_files:
        base_name = os.path.splitext(os.path.basename(vid_path))[0]
        json_path = os.path.join(config.RAW_VIDEOS_DIR, f"{base_name}.json")
        
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                # Ensure data is a list
                if isinstance(data, dict): data = [data]
                
                temp_annotations.append((vid_path, data))
                for item in data:
                    step_labels.add(item['step'])
        else:
            logger.warning(f"No annotation file for {base_name}. Skipping video.")

    if not step_labels:
        logger.warning("No annotations found. Exiting video processing.")
        return
        
    step_to_idx = {step: i for i, step in enumerate(sorted(list(step_labels)))}
    
    # Save mapping
    mapping_path = os.path.join(config.PROCESSED_SEQUENCES_DIR, 'step_label_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump(step_to_idx, f, indent=4)
        
    # Main Processing
    for vid_path, annotations in temp_annotations:
        logger.info(f"Processing video: {os.path.basename(vid_path)}")
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            logger.error(f"Could not open {vid_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # We need random access or linear scan. 
        # Since annotations define ranges, best to iterate frame by frame and match ranges.
        # OR: specific frame read if ranges are sparse.
        # Let's read the whole video into memory? No, might be too big.
        # Let's iterate linearly and buffer needed frames.
        
        # Optimization: Sort annotations by start_frame
        annotations.sort(key=lambda x: x['start_frame'])
        
        current_ann_idx = 0
        current_frame_idx = 0
        
        # Buffer to hold features for the current active segment
        # Actually, simple approach: linear scan. 
        # Check if current frame sits in any annotation interval.
        
        pbar = tqdm(total=total_frames, desc="Frames", leave=False)
        
        active_segments = [] # List of {'step':.., 'buffer': [features...], 'end':...}
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check for new segments starting here
            # (Loop through all annotations? If sorted, check current head)
            # Since multiple segments might overlap (unlikely in dance but possible), check all?
            # Assuming non-overlapping for simplicity or check all.
            for ann in annotations:
                if ann['start_frame'] == current_frame_idx:
                    active_segments.append({
                        'step': ann['step'],
                        'end_frame': ann['end_frame'],
                        'features': []
                    })
                    
            if not active_segments:
                current_frame_idx += 1
                pbar.update(1)
                continue
                
            # If we are in at least one segment, process frame
            # Process Frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Feature Extraction
            hand_vec, pose_vec, _ = extractor.extract_landmarks(image_rgb)
            emotion_id = extractor.extract_emotion(frame) # DeepFace needs BGR usually or path.
            # DeepFace.analyze can take numpy array (BGR).
            
            # Combine features
            # Frame_t = [hand (126), pose (99), emotion (1)]
            feature_vec = np.concatenate([hand_vec, pose_vec, [emotion_id]])
            
            # Add to active segments
            remaining_segments = []
            for seg in active_segments:
                seg['features'].append(feature_vec)
                
                if current_frame_idx >= seg['end_frame']:
                    # Segment done. Create sliding windows.
                    full_seq = np.array(seg['features'])
                    label_idx = step_to_idx[seg['step']]
                    
                    # Sliding Window
                    # shape: (T, D)
                    T = full_seq.shape[0]
                    W = config.SEQUENCE_LENGTH
                    S = config.STRIDE
                    
                    if T >= W:
                        for start_t in range(0, T - W + 1, S):
                            window = full_seq[start_t : start_t + W]
                            X_seq_list.append(window)
                            y_seq_list.append(label_idx)
                    else:
                        logger.warning(f"Segment for {seg['step']} too short ({T} < {W}). Ignoring.")
                        
                else:
                    remaining_segments.append(seg)
            
            active_segments = remaining_segments
            current_frame_idx += 1
            pbar.update(1)
            
        cap.release()
        pbar.close()
        
    # Save Final Datasets
    if X_seq_list:
        X = np.array(X_seq_list)
        y = np.array(y_seq_list)
        
        np.save(os.path.join(config.PROCESSED_SEQUENCES_DIR, 'X_sequences.npy'), X)
        np.save(os.path.join(config.PROCESSED_SEQUENCES_DIR, 'y_step_labels.npy'), y)
        logger.info(f"Saved Sequence Dataset: X={X.shape}, y={y.shape}")
    else:
        logger.warning("No sequence data generated.")

