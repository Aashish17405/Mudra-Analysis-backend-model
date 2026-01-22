import argparse
import sys
import os

# CRITICAL: Disable oneDNN optimizations and Force CPU to prevent TF crashes
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import logging
from pathlib import Path
from src import config, utils
from src.extraction import FeatureExtractor
from src.processing import process_mudra_images, process_video_sequences
from src.inference import run_inference
from src.narrative import generate_storyline

def main():
    parser = argparse.ArgumentParser(description="Bharatanatyam Data Engineering Pipeline")
    parser.add_argument('--mode', type=str, 
                       choices=['mudra', 'steps', 'all', 'inference', 'verify_datasets', 
                               'process_mudras', 'train_mudra', 'train_steps'],
                       default='all',
                       help='Which pipeline to run.')
    parser.add_argument('--video_path', type=str, help='Path to video for inference mode.', default=None)
    parser.add_argument('--model_type', type=str, choices=['hybrid', 'landmark', 'image'], 
                       default='landmark', help='Model type for mudra classification')
    parser.add_argument('--no_mudra', action='store_true', help='Disable mudra detection in inference')
    
    args = parser.parse_args()
    
    logger = utils.setup_logging()
    logger.info(f"Initializing Pipeline in mode: {args.mode}")
    
    # Initialize Feature Extractor
    # Note: Static image mode is True for images, False for videos.
    # If running both, we might need two instances or handle it carefully.
    
    try:
        if args.mode == 'verify_datasets':
            logger.info("Running dataset verification...")
            from src.verify_datasets import DatasetVerifier
            verifier = DatasetVerifier()
            verifier.run()
            return
        
        if args.mode == 'process_mudras':
            logger.info("Processing mudra datasets...")
            from src.mudra_processor import MudraProcessor
            processor = MudraProcessor()
            processor.process_dataset(apply_augmentation=True, augmentation_multiplier=3)
            processor.close()
            logger.info("Mudra processing complete!")
            return
        
        if args.mode == 'train_mudra':
            logger.info(f"Training mudra model (type: {args.model_type})...")
            from src.train_mudra_model import MudraTrainer
            trainer = MudraTrainer(model_type=args.model_type)
            trainer.load_data()
            trainer.build_model()
            trainer.train()
            trainer.plot_training_history()
            trainer.evaluate()
            y_pred, cm, report = trainer.predict_and_analyze()
            trainer.plot_confusion_matrix(cm)
            trainer.save_model()
            logger.info("Training complete!")
            return
        
        if args.mode == 'inference':
            if not args.video_path:
                logger.error("Please provide --video_path for inference mode.")
                return
            
            abs_video_path = os.path.abspath(args.video_path)
            base_name = os.path.splitext(abs_video_path)[0]
            output_json = f"{base_name}_inferred.json"
            logger.info(f"Saving JSON to: {output_json}")
            
            # Run inference with mudra detection
            output = run_inference(abs_video_path, output_json, use_mudra_model=not args.no_mudra)
            
            # Generate mudra-focused narrative
            mudra_narrative = ""
            if output.get('mudra_detections'):
                from src.mudra_meanings import generate_mudra_narrative
                mudra_narrative = generate_mudra_narrative(output['mudra_detections'])
            
            # Print narrative
            print("\n" + "="*60)
            print("MUDRA DETECTION NARRATIVE")
            print("="*60)
            if mudra_narrative:
                print(mudra_narrative)
            else:
                print("No mudras were detected in this performance.")
            print("="*60 + "\n")
            
            # Save mudra narrative
            output_story = f"{os.path.basename(base_name)}_mudra_story.txt"
            logger.info(f"Saving mudra narrative to: {output_story}")
            with open(output_story, 'w') as f:
                f.write("MUDRA DETECTION NARRATIVE\n")
                f.write("="*60 + "\n")
                if mudra_narrative:
                    f.write(mudra_narrative)
                else:
                    f.write("No mudras were detected in this performance.")
            logger.info(f"Narrative saved to {output_story}")
            
        if args.mode in ['mudra', 'all']:
            logger.info("Initializing Extractor for Images...")
            extractor_img = FeatureExtractor(use_static_image_mode=True)
            process_mudra_images(extractor_img)
            extractor_img.close()
            
        if args.mode in ['steps', 'all']:
            logger.info("Initializing Extractor for Videos...")
            extractor_vid = FeatureExtractor(use_static_image_mode=False)
            process_video_sequences(extractor_vid)
            extractor_vid.close()
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)
        
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
