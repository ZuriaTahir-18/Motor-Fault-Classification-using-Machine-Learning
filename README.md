# Motor Fault Detection using K-NN and PCA

This project implements a machine learning pipeline to detect faults in an induction motor using current signature data. The project uses the K-Nearest Neighbors (K-NN) algorithm implemented from scratch and applies Principal Component Analysis (PCA) to reduce dimensionality.

## Dataset Description

The dataset consists of time-domain three-phase current data acquired from an induction motor operating under various conditions:

- **Faults**: Inner and outer race bearing faults (0.7mm to 1.7mm), Broken Rotor Bar (BRB)
- **Loads**: 100W, 200W, 300W
- **Sampling rate**: 10 kHz with 1000 samples per channel
- **Classes**: 14 total (healthy, unhealthy, various fault types and severity levels)

Each row of the training data contains **1000 samples** and an associated class label.

##  Project Tasks

1. **Data Preprocessing**
   - Combine multiple datasets
   - Extract blocks of 1000 samples per row
   - Label each row with motor condition (healthy/unhealthy/fault type)

2. **K-NN Model (from scratch)**
   - Custom implementation of Euclidean distance
   - Manual train-test split (Hold-out)
   - 10-fold Cross Validation

3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - Sensitivity
   - Specificity
   - F1 Score

4. **Hyperparameter Tuning**
   - Optimize `k` (number of neighbors)
   - Plot accuracy vs. k for both test set and cross-validation

5. **Dimensionality Reduction with PCA**
   - Apply PCA to reduce the feature space
   - Repeat K-NN classification with reduced dimensions
   - Compare results with original data

## 🧪 Results

- Best accuracy and model performance are analyzed for both original and PCA-reduced datasets.
- The project discusses which approach gives the best results and why.

## Tools and Libraries

- Python
- NumPy, Pandas, Matplotlib
- Scikit-learn (only for PCA and metrics — KNN is implemented manually)

##  Files

- `motor_fault_knn_pca_classification.ipynb`: Main Jupyter Notebook
- `final_combined.csv`: Preprocessed dataset (not included due to size limits)
- `README.md`: Project overview (this file)

