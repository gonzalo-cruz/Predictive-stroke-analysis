# 🧠 Stroke Risk Prediction: Linear Models, MLP & SVM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A comprehensive study comparing Logistic Regression, Artificial Neural Networks (MLP), and Support Vector Machines (SVM) for medical diagnosis.**

## 📋 Project Overview
Stroke is a leading cause of mortality worldwide. Early identification of high-risk patients is crucial for medical intervention. This project utilizes the **Healthcare Stroke Dataset** to build binary classification models capable of predicting stroke probability based on physiological and demographic biomarkers (age, glucose levels, BMI, etc.).

The study was conducted in two phases to address the **Stability vs. Complexity** trade-off:
1.  **Phase 1:** Establishing baselines with Linear Models (Logistic Regression) and Universal Approximators (MLP).
2.  **Phase 2:** Refining boundaries with Support Vector Machines (Linear vs. RBF Kernels) and Recursive Feature Elimination (RFE).

## Data Engineering & Methodology
This analysis follows a rigorous Data Science lifecycle, tackling the massive class imbalance inherent in medical datasets.

### Preprocessing Pipeline
* **Imputation:** KNN Imputer for missing BMI values.
* **Transformation:** Logarithmic transformation for skewed features (Glucose, BMI).
* **Encoding:** One-Hot Encoding for categorical variables.
* **Scaling:** `StandardScaler` (Critical for SVM performance).
* **Balancing:** Implementation of **SMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class (Stroke).

### Model Architectures Evaluated
1.  **Logistic Regression:** Implemented with ElasticNet regularization and Wrapper methods.
2.  **Multi-Layer Perceptron (MLP):** Explored to capture non-linear patterns.
3.  **Support Vector Machines (SVM):**
    * **LinearSVC:** Optimized for high-dimensional linear separation.
    * **Kernel SVM (RBF):** Tested for non-linear decision boundaries.
4.  **Wrapper Methods (Feature Selection):** Utilized `RFECV` to reduce the dataset from 20+ features to the **12-15 most significant predictors**, adhering to the Principle of Parsimony.

## Key Findings & Results

The primary metric for evaluation was **Recall (Sensitivity)**. In a clinical triage setting, the cost of a False Negative (failing to identify a patient at risk) is unacceptable.

### Performance Comparison (Validation Set)

| Model | Recall (Sensitivity) | Specificity | AUC | Observations |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (Base)** | 0.78 | 0.74 | 0.76 | Solid baseline. |
| **Logistic Regression + Wrapper** | **0.88** | **0.73** | **0.84** | **Best Overall Model.** Achieved the highest sensitivity with reduced complexity. |
| **SVM RBF (Non-Linear)** | 0.08 | 0.95 | 0.66 | **Overfitting**. The model prioritized specificity, failing to detect the disease. |
| **SVM Linear + Wrapper** | 0.80 | 0.72 | 0.83 | **Best SVM Model.** Excellent performance, slightly lower recall than Logistic but highly robust. |

### Lessons Learned
1.  **Linearity Wins:** Despite the theoretical power of non-linear kernels (RBF) and Deep MLPs, the **Linear SVM** and **Logistic Regression** proved significantly more robust for this specific dataset.
2.  **Feature Selection is Critical:** The Wrapper method improved the performance of the Logistic Regression model, boosting Recall to 0.88 and AUC to 0.84, confirming that a parsimonious model (fewer variables) generalizes better.
3.  **The "Medical" Metric:** Optimizing for Accuracy would have been misleading. By focusing on Recall, we ensured the model is actually useful for screening patients.

## 💻 Tech Stack
* **Language:** Python 3.8+
* **Libraries:** Scikit-Learn, Pandas, NumPy, Imbalanced-learn (SMOTE).
* **Visualization:** Matplotlib, Seaborn.

## 🚀 Installation & Usage

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/yourusername/predictive-stroke-analysis.git](https://github.com/yourusername/predictive-stroke-analysis.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    * Phase 1: `jupyter notebook notebooks/Tarea1_Logistic_MLP.ipynb`
    * Phase 2: `jupyter notebook notebooks/Tarea2_SVM.ipynb`

## 👤 Author
**Gonzalo Cruz Gómez**
