
"""
Main script.
"""

import os
import models
import argparse
from datetime import datetime as dt


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", type=str, required=True, help='Path to configuration file for model')
    ap.add_argument("--task", "-t", type=str, required=True, help='Task to perform, (train or plot)')
    ap.add_argument("--load", "-l", type=str, help='Path to directory containing trained model to be loaded')
    ap.add_argument("--output", "-o", type=str, default=dt.now().strftime("%H-%M_%Y-%m-%d"), help='Path to output directory')
    args = vars(ap.parse_args())

    # Initialize trainer
    trainer = models.StockMovementClassifier(args)

    if args['task'] == "train":
        trainer.train()

    elif args['task'] == "plot":
        if args['load'] is not None:
            print("[WARN] Chosen to plot attention probs but no trained model has been loaded!")
        trainer.get_stock_attention_probs()
