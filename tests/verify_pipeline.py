import os
import cv2
import numpy as np
import json
import shutil
import unittest
from unittest.mock import MagicMock, patch
import src.config as config
from src.processing import process_mudra_images, process_video_sequences

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Setup dummy directories
        self.test_dir = os.path.join(config.DATA_DIR, 'test_env')
        self.raw_mudras = os.path.join(self.test_dir, 'raw', 'mudras')
        self.raw_videos = os.path.join(self.test_dir, 'raw', 'videos')
        self.processed_mudras = os.path.join(self.test_dir, 'processed', 'mudra_frame_dataset')
        self.processed_sequences = os.path.join(self.test_dir, 'processed', 'step_sequence_dataset')
        
        os.makedirs(os.path.join(self.raw_mudras, 'TestMudra'), exist_ok=True)
        os.makedirs(self.raw_videos, exist_ok=True)
        os.makedirs(self.processed_mudras, exist_ok=True)
        os.makedirs(self.processed_sequences, exist_ok=True)
        
        # Patch config paths to point to test_env
        self.orig_mudra_dir = config.RAW_MUDRAS_DIR
        self.orig_video_dir = config.RAW_VIDEOS_DIR
        self.orig_proc_mudra = config.PROCESSED_MUDRAS_DIR
        self.orig_proc_seq = config.PROCESSED_SEQUENCES_DIR
        
        config.RAW_MUDRAS_DIR = self.raw_mudras
        config.RAW_VIDEOS_DIR = self.raw_videos
        config.PROCESSED_MUDRAS_DIR = self.processed_mudras
        config.PROCESSED_SEQUENCES_DIR = self.processed_sequences
        
        # Create dummy image
        self.dummy_img_path = os.path.join(self.raw_mudras, 'TestMudra', 'img1.jpg')
        cv2.imwrite(self.dummy_img_path, np.zeros((100, 100, 3), dtype=np.uint8))
        
        # Create dummy video
        self.dummy_vid_path = os.path.join(self.raw_videos, 'test_dance.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.dummy_vid_path, fourcc, 30.0, (100, 100))
        for _ in range(200): # 200 frames
            out.write(np.zeros((100, 100, 3), dtype=np.uint8))
        out.release()
        
        # Create dummy json
        self.dummy_json_path = os.path.join(self.raw_videos, 'test_dance.json')
        annotations = [
            {"step": "TestStep", "start_frame": 10, "end_frame": 60}, # 50 frames, enough for one window
        ]
        with open(self.dummy_json_path, 'w') as f:
            json.dump(annotations, f)

    def tearDown(self):
        # Restore paths
        config.RAW_MUDRAS_DIR = self.orig_mudra_dir
        config.RAW_VIDEOS_DIR = self.orig_video_dir
        config.PROCESSED_MUDRAS_DIR = self.orig_proc_mudra
        config.PROCESSED_SEQUENCES_DIR = self.orig_proc_seq
        
        # Clean up
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_mudra_processing(self):
        print("\nTesting Mudra Processing...")
        mock_extractor = MagicMock()
        # Mock extract_landmarks to return random valid shape
        # Hand vec: 126
        mock_extractor.extract_landmarks.return_value = (np.random.rand(126), np.random.rand(99), {})
        
        process_mudra_images(mock_extractor)
        
        # Check outputs
        x_path = os.path.join(self.processed_mudras, 'X_landmarks.npy')
        y_path = os.path.join(self.processed_mudras, 'y_mudra_labels.npy')
        
        self.assertTrue(os.path.exists(x_path))
        self.assertTrue(os.path.exists(y_path))
        
        X = np.load(x_path)
        y = np.load(y_path)
        
        print(f"Mudra X shape: {X.shape}, y shape: {y.shape}")
        self.assertEqual(X.shape[1], 126)
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(X), 1) # 1 image

    def test_sequence_processing(self):
        print("\nTesting Sequence Processing...")
        mock_extractor = MagicMock()
        mock_extractor.extract_landmarks.return_value = (np.random.rand(126), np.random.rand(99), {})
        mock_extractor.extract_emotion.return_value = 1 # happy
        
        process_video_sequences(mock_extractor)
        
        # Check outputs
        x_path = os.path.join(self.processed_sequences, 'X_sequences.npy')
        y_path = os.path.join(self.processed_sequences, 'y_step_labels.npy')
        
        self.assertTrue(os.path.exists(x_path))
        self.assertTrue(os.path.exists(y_path))
        
        X = np.load(x_path)
        y = np.load(y_path)
        
        print(f"Sequence X shape: {X.shape}, y shape: {y.shape}")
        # Sequence Length 30, Features 126+99+1=226
        self.assertEqual(X.shape[1], 30)
        self.assertEqual(X.shape[2], 226)
        self.assertTrue(len(X) > 0)

if __name__ == '__main__':
    unittest.main()
