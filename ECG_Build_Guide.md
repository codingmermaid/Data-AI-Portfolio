# Building the ECG Arrhythmia Classifier — A Data Science / ML Guide

A step-by-step walkthrough of the CNN-LSTM arrhythmia project from an ML standpoint.

**1. Data acquisition.** Pull the MIT-BIH Arrhythmia Database from Kaggle: 87,554 training and 21,892 test heartbeats (47 patients), each a 187-point segment labeled across five classes.

**2. Data prep.** Split features from labels, normalize the 187 samples per beat with a scaler for stable gradients, and reshape into `(samples, 187, 1)` sequences. Note the severe imbalance — normal beats are ~83%, ventricular ~7%.

**3. Handling imbalance.** Rather than resampling, compute per-class weights and feed them into a weighted CrossEntropyLoss so minority arrhythmias aren't ignored.

**4. Feature learning + model.** Three 1D-conv blocks (1→32→64→128 filters, with batch-norm, ReLU, max-pool, 0.2 dropout) extract morphology and shrink 187→23 steps; two bidirectional LSTMs (64 units) capture temporal rhythm; dense layers (128→64, 0.5 dropout) output five classes.

**5. Training.** Adam (lr 0.001), batch 64, 50 epochs, ReduceLROnPlateau scheduling. Save the best checkpoint by validation performance.

**6. Evaluation.** Judge on macro metrics, not raw accuracy: confusion matrix, per-class precision/recall/F1, ROC-AUC, and Cohen's Kappa. Target ~95% accuracy, 0.98 macro-AUC, with strong ventricular recall for patient safety.
