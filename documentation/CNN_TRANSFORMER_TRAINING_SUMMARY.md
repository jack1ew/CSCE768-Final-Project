# CNN + Transformer Training Summary

## Overview
Successfully trained the CNN + Transformer model on all three datasets using the same training protocol as the other 5 models (logistic, random_forest, knn, mlp, transformer).

## Training Configuration
- **Split Strategy**: Random (80/20 train/test split)
- **Epochs**: 12
- **Learning Rate**: 1e-3 with warmup (2 epochs)
- **Scheduler**: Cosine Annealing
- **Batch Size**: 64
- **Max Rounds**: 75
- **Gradient Clipping**: 1.0

## Training Results

### ESTA Dataset
- **Training Time**: 56.9s
- **Train Matches**: 1,246
- **Test Matches**: 312
- **Final Metrics**:
  - Accuracy: 0.9840
  - Log Loss: 0.0330
  - Brier Score: 0.0111
  - ROC AUC: 0.9993

### Kaggle Dataset
- **Training Time**: 19.0 minutes (1141.9s)
- **Train Matches**: 25,368
- **Test Matches**: 6,342
- **Final Metrics**:
  - Accuracy: 0.8468
  - Log Loss: 0.3945
  - Brier Score: 0.1181
  - ROC AUC: 0.8865

### Combined Dataset
- **Training Time**: 19.6 minutes (1173.7s)
- **Train Matches**: 26,614
- **Test Matches**: 6,654
- **Final Metrics**:
  - Accuracy: 0.8449
  - Log Loss: 0.3903
  - Brier Score: 0.1175
  - ROC AUC: 0.8899

## Total Training Time
**39.5 minutes** for all 3 datasets

## Model Artifacts Saved

### Model Weights
- `models/saved_models/cnn_transformer_esta.pt`
- `models/saved_models/cnn_transformer_esta_random.pt`
- `models/saved_models/cnn_transformer_kaggle.pt`
- `models/saved_models/cnn_transformer_kaggle_random.pt`
- `models/saved_models/cnn_transformer_combined.pt`
- `models/saved_models/cnn_transformer_combined_random.pt`

### Scalers
- `models/saved_models/cnn_transformer_esta_random_scaler.joblib`
- `models/saved_models/cnn_transformer_kaggle_random_scaler.joblib`
- `models/saved_models/cnn_transformer_combined_random_scaler.joblib`

### Metrics
All metrics appended to: `results/cnn_transformer_metrics.csv`

## Ready for Analysis
The CNN + Transformer model is now trained consistently across all datasets and ready for the same analysis pipeline as the other models:
- Round-by-round analysis
- Feature importance studies
- Calibration curves
- Checkpoint comparisons
- Economic momentum analysis
- Comeback analysis

## Notes
- The model uses a hybrid CNN-Transformer architecture that captures both local (via CNN) and global (via Transformer) temporal dependencies
- Training was performed on regulation rounds (1-30) with support for overtime rounds (up to 75)
- The ESTA dataset shows significantly better performance due to its smaller size and potentially less variance
- All models show no data leakage (0 overlapping match IDs between train and test sets)
