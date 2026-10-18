"""
SignVision - Model Training Entrypoint
======================================
Trains the SignVision Multi-Stream Temporal Attention Network (MTAN-SLR)
on the collected 100-word dataset in dataset/MP_Data.

Usage:
    python 2_train_model.py
    python 2_train_model.py --epochs 60 --batch_size 16
"""

import sys
import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SignVision Model Training")
    parser.add_argument("--data_dir", type=str, default="dataset/MP_Data", help="Path to landmark dataset directory")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
