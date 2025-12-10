# Predictive Stroke Analysis & Risk Modeling

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A comparative study of Linear Models vs. Artificial Neural Networks (MLP) for medical diagnosis.**

## Project Overview
Stroke is a leading cause of mortality worldwide. Early identification of high-risk patients is crucial for medical intervention. This project utilizes the **Healthcare Stroke Dataset** to build a binary classification model capable of predicting stroke probability based on physiological and demographic biomarkers (age, glucose levels, BMI, etc.).

The goal was not just to predict, but to **compare architectural approaches**: interpreting the transparency of Logistic Regression against the non-linear capabilities of a Multi-Layer Perceptron (MLP).

[**📄 View Full Technical Report (PDF)**](reports/technical-report.pdf)

## Key Features & Methodology
This analysis follows a rigorous Data Science lifecycle:

* **Exploratory Data Analysis (EDA):** Analyzed distributions of key variables to understand the imbalance in the target class.
* **Preprocessing:** Implemented sampling with replacement (bootstrap) to handle dataset variability and create robust test subsets.
* **Feature Selection:** Utilized **Wrapper Methods** to identify the most significant predictors, reducing noise and computational cost.
* **Model 1: Logistic Regression:** Implemented with Regularization to prevent overfitting and establish a baseline.
* **Model 2: Multi-Layer Perceptron (MLP):** Explored the "Universal Approximation" capabilities of Neural Networks to capture non-linear patterns in patient data.
* **Evaluation:** Models were evaluated using **F1-Score** (prioritized over accuracy due to class imbalance) and validation curves as well as other performance metrics.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (LogisticRegression, MLPClassifier)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Jupyter Notebooks

**Key Findings:**
* **Logistic Approach:** Provided high interpretability regarding which factors (e.g., Age, Hypertension) most strongly correlate with stroke risk.
* **MLP Approach:** demonstrated the complexity of model selection in medical fields, where slight gains in F1-score must be weighed against computational cost and "black box" opacity.
* *See the `notebooks/` folder for the detailed validation curves and confusion matrices.*

## 🚀 Installation & Usage

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/yourusername/predictive-stroke-analysis.git
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis:**
    ```bash
    jupyter notebook notebooks/01_stroke_prediction_analysis.ipynb
    ```

## 🧠 Engineering Decisions & Lessons Learned

### Why F1-Score over Accuracy?
In medical diagnosis, False Negatives (predicting "Healthy" when a patient is at risk) are dangerous. Accuracy is misleading in imbalanced datasets (where most people *don't* have strokes). I optimized for F1-Score to balance Precision and Recall.

### The Trade-off: Linearity vs. Non-Linearity
While the **MLP** offered the potential to model complex interactions between biomarkers, the **Logistic Regression** (with regularization) offered robustness and interpretability. In a clinical setting, knowing *why* a model predicts a stroke is often as important as the prediction itself.

## 👤 Author
**Gonzalo Cruz Gómez**

---
*This project was originally developed as part of the subject Machine Learning II.*
