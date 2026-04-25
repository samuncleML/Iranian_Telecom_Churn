"""
Main entry point for the Iranian Telecom Churn prediction pipeline.
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchmetrics import F1Score, Accuracy, Recall, Precision

from src.dataset import train_loader, val_loader, test_loader
from src.module import ChurnModule


def train_model(num_epochs=20, save_path='models/Iranian_Telecom_Churn_model.pth'):
    """
    Train the churn prediction model.
    
    Args:
        num_epochs: Number of training epochs
        save_path: Path to save the trained model
    """
    print("=" * 60)
    print("TRAINING IRANIAN TELECOM CHURN MODEL")
    print("=" * 60)
    
    # Initialize metrics
    f1_score = F1Score(task='binary')
    accuracy_score = Accuracy(task='binary')
    recall_score = Recall(task='binary')
    precision_score = Precision(task='binary')
    
    f1_score_val = F1Score(task='binary')
    accuracy_score_val = Accuracy(task='binary')
    recall_score_val = Recall(task='binary')
    precision_score_val = Precision(task='binary')
    
    # Initialize model, loss function, and optimizer
    model = ChurnModule()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # StepLR scheduler: gamma=0.02, step_size=7
    scheduler = StepLR(optimizer, step_size=7, gamma=0.02)
    
    print(f"\nConfiguration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Initial LR: 0.001")
    print(f"  - Scheduler: StepLR(gamma=0.02, step_size=7)")
    print(f"  - Loss: BCELoss")
    print(f"  - Optimizer: Adam")
    print()
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        
        f1_score.reset()
        accuracy_score.reset()
        recall_score.reset()
        precision_score.reset()
        
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            
            loss = criterion(outputs.squeeze(1), labels)
            f1_score.update(outputs.squeeze(1), labels)
            accuracy_score.update(outputs.squeeze(1), labels)
            recall_score.update(outputs.squeeze(1), labels)
            precision_score.update(outputs.squeeze(1), labels)
            
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        train_accuracy = accuracy_score.compute().item()
        train_recall = recall_score.compute().item()
        train_precision = precision_score.compute().item()
        train_f1 = f1_score.compute().item()
        train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        
        f1_score_val.reset()
        accuracy_score_val.reset()
        recall_score_val.reset()
        precision_score_val.reset()
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs.squeeze(1), labels)
                f1_score_val.update(outputs.squeeze(1), labels)
                accuracy_score_val.update(outputs.squeeze(1), labels)
                recall_score_val.update(outputs.squeeze(1), labels)
                precision_score_val.update(outputs.squeeze(1), labels)
                val_running_loss += loss.item()
        
        val_accuracy = accuracy_score_val.compute().item()
        val_recall = recall_score_val.compute().item()
        val_precision = precision_score_val.compute().item()
        val_f1 = f1_score_val.compute().item()
        val_loss = val_running_loss / len(val_loader)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:2d}/{num_epochs} | LR: {current_lr:.6f}")
        print(f"  Train -> Loss: {train_loss:.4f} | Acc: {train_accuracy*100:.2f}% | F1: {train_f1*100:.2f}%")
        print(f"  Val   -> Loss: {val_loss:.4f} | Acc: {val_accuracy*100:.2f}% | F1: {val_f1*100:.2f}%")
        
        scheduler.step()
    
    # Save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ Model saved to: {save_path}")
    
    return model


def evaluate_model(model_path='models/Iranian_Telecom_Churn_model.pth'):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model_path: Path to the saved model
    """
    print("=" * 60)
    print("EVALUATING MODEL ON TEST SET")
    print("=" * 60)
    
    # Load model
    model = ChurnModule()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Initialize metrics
    test_f1_score = F1Score(task='binary')
    test_accuracy_score = Accuracy(task='binary')
    test_recall_score = Recall(task='binary')
    test_precision_score = Precision(task='binary')
    
    with torch.no_grad():
        for features, labels in test_loader:
            outputs = model(features)
            test_f1_score.update(outputs.squeeze(1), labels)
            test_accuracy_score.update(outputs.squeeze(1), labels)
            test_recall_score.update(outputs.squeeze(1), labels)
            test_precision_score.update(outputs.squeeze(1), labels)
    
    print("\n" + "=" * 50)
    print("TEST SET RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {test_accuracy_score.compute().item()*100:.2f}%")
    print(f"F1 Score:  {test_f1_score.compute().item()*100:.2f}%")
    print(f"Recall:    {test_recall_score.compute().item()*100:.2f}%")
    print(f"Precision: {test_precision_score.compute().item()*100:.2f}%")
    print("=" * 50)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Iranian Telecom Churn Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                    Train the model (default 20 epochs)
  python main.py train --epochs 50       Train with 50 epochs
  python main.py train --save custom.pt  Save to custom path
  python main.py evaluate                 Evaluate trained model
  python main.py full                     Run full pipeline (train + evaluate)
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    train_parser.add_argument('--save', type=str, default='models/Iranian_Telecom_Churn_model.pth', 
                              help='Path to save model')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate the model on test set')
    eval_parser.add_argument('--model', type=str, default='models/Iranian_Telecom_Churn_model.pth',
                             help='Path to trained model')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline (train + evaluate)')
    full_parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    full_parser.add_argument('--save', type=str, default='models/Iranian_Telecom_Churn_model.pth',
                             help='Path to save model')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(num_epochs=args.epochs, save_path=args.save)
    
    elif args.command == 'evaluate':
        evaluate_model(model_path=args.model)
    
    elif args.command == 'full':
        model = train_model(num_epochs=args.epochs, save_path=args.save)
        evaluate_model(model_path=args.save)
    
    else:
        # Default: run full pipeline
        print("No command specified. Running full pipeline...\n")
        model = train_model(num_epochs=20)
        evaluate_model()


if __name__ == '__main__':
    main()