import logging
from modeling.model_training import RacingPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting Walk-Forward Backtest with Simultaneous Kelly and Regression Margin Projections.")
    pipeline = RacingPipeline(features_csv="data/model_features.csv")
    pipeline.run_walk_forward_backtest()
