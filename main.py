import subprocess
import sys
import logging
import os

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Pipeline")

def run_script(script_path: str):
    """Runs a python script as a subprocess and streams its output."""
    logger.info(f"==================================================")
    logger.info(f"Starting Step: {script_path}")
    logger.info(f"==================================================")
    
    if not os.path.exists(script_path):
        logger.error(f"Script not found: {script_path}")
        sys.exit(1)

    try:
        # Run the script using the current Python executable
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
        logger.info(f"✅ Successfully completed {script_path}\n")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed while running {script_path}.")
        logger.error(f"Exit code: {e.returncode}")
        logger.error("Pipeline stopped.")
        sys.exit(1)

def main():
    logger.info("🚀 Starting HKJC End-to-End ML Pipeline 🚀\n")
    
    # Define the sequence of scripts to run
    steps = [
        "scraping/scraper.py",
        "data_pipeline/feature_engineering.py",
        "modeling/model_training.py"  # This will also save the calibrator and model
    ]
    
    for step in steps:
        run_script(step)
        
    logger.info("🎉 All pipeline steps completed successfully! The model is now up to date.")

if __name__ == "__main__":
    main()
