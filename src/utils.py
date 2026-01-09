import logging
import sys

def setup_logging(log_file='pipeline.log'):
    """
    Configures logging to file and console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('Pipeline')
