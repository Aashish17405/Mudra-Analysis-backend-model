# Nithya Analysis ML Model

This repository contains the machine learning pipeline for analyzing Bharatanatyam dance performances. It accurately detects hand gestures (Mudras) and identifies complex dance sequences using computer vision and deep learning.

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or 3.10 (Recommended for TensorFlow stability)
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Aashish17405/Mudra-Analysis-backend-model.git
   cd Mudra-Analysis-backend-model
   ```

2. **Set up a virtual environment:**

   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

## 🛠️ Usage

### Running Inference

To analyze a Bharatanatyam performance video and generate a mudra narrative:

```bash
python main_pipeline.py --mode inference --video_path "sample videos/sample1.mp4"
```

**Arguments:**

- `--mode inference`: Runs the full detection and narrative generation pipeline.
- `--video_path`: Path to the input video file.
- `--no_mudra`: (Optional) Flag to disable mudra detection.

The results will be saved as:

- `[video_name]_inferred.json`: Detailed detection temporal data.
- `[video_name]_mudra_story.txt`: A plain-text narrative explaining the detected gestures.

## 🏗️ System Architecture

The system operates on a dual-path architecture to analyze both static gestures and temporal dance steps simultaneously.

```mermaid
graph TD
    A[Input Video] --> B[Frame Extraction];
    B --> C{MediaPipe Extraction};

    C -->|Hand Landmarks| D[Mudra Detection Pipeline];
    C -->|Body Pose + Hands| E[Dance Step Pipeline];

    subgraph "Mudra Detection (Static)"
    D --> D1[Crop Hand Region];
    D1 --> D2[Pre-processing & Rescaling];
    D2 --> D3[EfficientNetB0 CNN];
    D3 --> D4[Mudra Label (e.g., 'Pataka')];
    end

    subgraph "Dance Step Detection (Temporal)"
    E --> E1[Feature Concatenation];
    E1 --> E2[Sliding Window Buffer];
    E2 --> E3[Sequence Model (LSTM/GRU)];
    E3 --> E4[Step Label (e.g., 'Natayarambham')];
    end

    D4 --> F[Narrative Generator];
    E4 --> F;
    F --> G[Final JSON & Text Report];
```

## 📊 Dataset & Augmentation

The model is trained on a curated dataset of Bharatanatyam imagery, processed to ensure robustness against lighting and orientation changes.

### Dataset Structure

- **Raw Data**: Organized by class folders (`data/raw/mudras/<class_name>/`).
- **Processing**: Images are resized and normalized before feature extraction.
- **Landmarks**: 21 3D points per hand (126 features total) extracted using MediaPipe.
- **Videos**: Annotated video segments for dynamic step training.

### Data Augmentation

To improve generalization and prevent overfitting, the training pipeline applies the following augmentations to the training set (3x multiplier):

1.  **Horizontal Flip**: Simulates mirrored performances.
2.  **Rotation**: Random rotations of ±15 degrees to handle wrist variations.
3.  **Brightness Adjustment**:
    - **Brighter**: +20% HSV Value channel.
    - **Darker**: -20% HSV Value channel.

## 🧠 Algorithms & Models

### 1. Mudra Classification (Hand Gestures)

- **Algorithm**: Convolutional Neural Network (CNN) based on **EfficientNetB0** (Transfer Learning).
- **Input**: Pre-cropped 224x224 RGB images of the hand (extracted via MediaPipe bounding box).
- **Training**:
  - **Optimizer**: Adam with learning rate reduction on plateau.
  - **Loss Function**: Sparse Categorical Crossentropy.
  - **Metrics**: Accuracy, Top-3 Accuracy, Top-5 Accuracy.
- **Robustness**: Handles "Unknown" mudras and filters low-confidence predictions.

### 2. Dance Step Prediction (Sequences)

- **Algorithm**: Sequential Neural Network (LSTM/GRU).
- **Input**: Time-series buffer of feature vectors (Hand Landmarks + Pose Landmarks + Emotion).
- **Windowing**: Sliding window approach (e.g., 30 frames) to capture movement dynamics.

## 🚀 Training Pipeline

The training process is automated via `src/train_mudra_model.py`:

1.  **Load & Split**: Loads processed features (`.npy` files) and splits into Train/Val/Test (stratified).
2.  **Model Construction**: Builds the EfficientNet model with custom classification head.
3.  **Training Loop**: Trains with:
    - **Early Stopping**: Prevents overfitting by stopping when validation loss plateaus.
    - **Model Checkpointing**: Saves the best model weights.
    - **TensorBoard**: Logs metrics for visualization.
4.  **Evaluation**: Generates Classification Report (Precision/Recall/F1) and Confusion Matrix.

## 📁 Key Files

- `src/processing.py`: Augmentation logic and raw data processing.
- `src/train_mudra_model.py`: Main training script for the CNN.
- `src/mudra_predictor.py`: Inference engine handling model loading and prediction.
- `src/inference.py`: Orchestrates the video analysis pipeline.
