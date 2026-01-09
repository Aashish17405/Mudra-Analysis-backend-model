# Bharatanatyam Data Engineering Pipeline

A robust pipeline to ingest Bharatanatyam dance videos and mudra images, extract MediaPipe landmarks and facial emotions, and produce normalized `.npy` datasets.

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Directory Structure**
   Ensure your data is placed as follows:

   ```
   data/
   ├── raw/
   │   ├── mudras/
   │   │   ├── Alapadma/
   │   │   ├── Pataka/
   │   │   └── ...
   │   └── videos/
   │       ├── dance_video_1.mp4
   │       ├── dance_video_1.json
   │       └── ...
   ```

   **Video Annotation Format (`.json`)**:

   ```json
   [
     {
       "step": "Alarippu",
       "start_frame": 120,
       "end_frame": 165
     }
   ]
   ```

## Usage

Run the pipeline via command line:

```bash
# Process both Mudra Images and Video Steps
python main_pipeline.py --mode all

# Process only Mudra Images
python main_pipeline.py --mode mudra

# Process only Video Sequences
python main_pipeline.py --mode steps

# Run Inference (Generate Timeline) on a Video
python main_pipeline.py --mode inference --video_path "data/raw/videos/test_video.mp4"
```

## Inference Mode

The system can infer dance steps from a raw video file.

- **Input**: Path to `.mp4` video.
- **Output**: A JSON file (e.g., `test_video_inferred.json`) containing the timeline of detected steps.
- **Note**: Currently uses a heuristic fallback if ML models are not fully trained or if running in "Safe Mode".

## Output

Processed data will be saved in `data/processed/`:

- `mudra_frame_dataset/`

  - `X_landmarks.npy`: Shape `(N_samples, 126)`
  - `y_mudra_labels.npy`: Shape `(N_samples,)`
  - `label_mapping.json`: Class name to index mapping.

- `step_sequence_dataset/`
  - `X_sequences.npy`: Shape `(N_sequences, 30, 226)`
    - (Feature dim = 126 + 99 + 1 = 226)
  - `y_step_labels.npy`: Shape `(N_sequences,)`
  - `step_label_mapping.json`: Step name to index mapping.
