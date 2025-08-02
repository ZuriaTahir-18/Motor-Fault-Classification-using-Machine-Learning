

# Motor Fault Classification using Machine Learning

This repository contains a set of machine learning experiments for classifying induction motor faults based on time-domain current signal data. Multiple ML models have been applied across different stages of binary and multiclass classification. The goal is to explore the effectiveness of various algorithms including K-NN, Logistic Regression, Naïve Bayes, and SVM on both raw and dimensionally reduced data.

## Dataset Description

The dataset contains instantaneous values of three-phase current signature data collected from an induction motor operating under both healthy and unhealthy conditions. The motor was subjected to various mechanical faults such as inner-race and outer-race bearing faults with severity levels of 0.7mm to 1.7mm, and broken rotor bar (BRB) faults. These experiments were performed under different load conditions (100W, 200W, and 300W).

* A total of **39 datasets** were collected.
* Each file contains more than **100,000 samples**.
* Data was acquired at a **10 kHz sampling rate** using non-invasive current sensors.
* Data is organized into **7 folders**, each representing different load levels.
* This is a **14-class classification** problem including healthy and various unhealthy states.

## Project Overview

The overall project aims to:

* Preprocess high-frequency motor current signal data into structured labeled datasets.
* Train and evaluate several machine learning models using both binary and multiclass labels.
* Perform dimensionality reduction using PCA and analyze its effect on classification performance.
* Visualize training and validation performance metrics to interpret model behavior.

## Notebooks & Tasks

### `motor_fault_knn_pca_classification.ipynb`

* Data preprocessing: segmenting signal blocks and labeling (healthy/unhealthy).
* Manual implementation of the **K-Nearest Neighbors (K-NN)** algorithm (no built-in ML libraries).
* Evaluation using both **Hold-out** and **10-fold Cross Validation**.
* Metrics: Accuracy, Recall, Precision, Sensitivity, Specificity, F1 Score.
* **Hyperparameter tuning**: optimized value of K.
* **PCA** applied for dimensionality reduction and the entire pipeline repeated.
* Final comparison of K-NN performance on original vs PCA-transformed data.

### `nb_svm_binary_multiclass_classification.ipynb`

* Data preprocessing to prepare labeled data.
* Applied **Logistic Regression** for:

  * **Binary classification** (healthy vs unhealthy)
  * **Multiclass classification** (14 motor conditions)
* Classification metrics computed for both settings.

### `logistic_regression_motor_fault_classification.ipynb`

* Applied both **Naïve Bayes** and **Support Vector Machine (SVM)** for:

  * Binary classification
  * Multiclass classification
* Evaluation using Accuracy, Precision, Recall, Sensitivity, Specificity, F1 Score.
* **Training and Validation curves** plotted for both classifiers to visualize accuracy and loss.

## Requirements

To run these notebooks smoothly, you need the following libraries installed:

```bash
numpy
pandas
matplotlib
scikit-learn
seaborn
```

## How to Run the Project

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install Required Libraries**
   (You can use pip or conda)

   ```bash
   pip install -r requirements.txt
   ```

   Or manually install the mentioned packages.

3. **Open Jupyter Notebook or any IDE**
   Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. **Run the Notebooks**

   * Open the desired `.ipynb` file.
   * Execute the cells in order to see results and visualizations.
   * Make sure your dataset is correctly placed if any path references are hardcoded.

5. **Optional: Modify Parameters**
   You can tune hyperparameters (e.g., number of neighbors in K-NN or PCA components) to explore different results.


