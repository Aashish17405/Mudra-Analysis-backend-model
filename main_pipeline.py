import argparse
import sys
import os
import logging
from src import config, utils
from src.extraction import FeatureExtractor
from src.processing import process_mudra_images, process_video_sequences
from src.inference import run_inference
from src.narrative import generate_storyline

def main():
    parser = argparse.ArgumentParser(description="Bharatanatyam Data Engineering Pipeline")
    parser.add_argument('--mode', type=str, choices=['mudra', 'steps', 'all', 'inference'], default='all',
                        help='Which pipeline to run: "mudra", "steps", "all", or "inference".')
    parser.add_argument('--video_path', type=str, help='Path to video for inference mode.', default=None)
    
    args = parser.parse_args()
    
    logger = utils.setup_logging()
    logger.info(f"Initializing Pipeline in mode: {args.mode}")
    
    # Initialize Feature Extractor
    # Note: Static image mode is True for images, False for videos.
    # If running both, we might need two instances or handle it carefully.
    
    try:
        if args.mode == 'inference':
            if not args.video_path:
                logger.error("Please provide --video_path for inference mode.")
                return
            
            abs_video_path = os.path.abspath(args.video_path)
            base_name = os.path.splitext(abs_video_path)[0]
            output_json = f"{base_name}_inferred.json"
            logger.info(f"Saving JSON to: {output_json}")
            predictions = run_inference(abs_video_path, output_json)
            
            # Generate and print story
            story = generate_storyline(predictions)
            print("\n" + "="*40)
            print("GENERATED DANCE NARRATIVE")
            print("="*40)
            print(story)
            print("="*40 + "\n")
            
            # Save Story
            # Fallback to simple filename in CWD to avoid path issues
            output_story = f"{os.path.basename(base_name)}_story.txt"
            logger.info(f"Attempting to save story to: {output_story}")
            with open(output_story, 'w') as f:
                f.write(story)
            logger.info(f"Story saved to {output_story}")
            
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
