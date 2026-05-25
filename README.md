<div align="center">

# 😴 Sleep Disorder Prediction

**A supervised ML classification project that predicts sleep disorders- Insomnia, Sleep Apnea or None from health and lifestyle features including age, BMI, stress level, sleep duration and blood pressure, using Logistic Regression, Random Forest and SVM.**

<p align="center">

<img src="https://img.shields.io/badge/Python-3.9+-0f172a?style=for-the-badge&logo=python&logoColor=FFD43B" />
<img src="https://img.shields.io/badge/Notebook-Jupyter-0f172a?style=for-the-badge&logo=jupyter&logoColor=F97316" />
<img src="https://img.shields.io/badge/ML-Scikit--Learn-0f172a?style=for-the-badge&logo=scikitlearn&logoColor=F7931E" />
<img src="https://img.shields.io/badge/Data-Pandas-0f172a?style=for-the-badge&logo=pandas&logoColor=38BDF8" />
<img src="https://img.shields.io/badge/License-MIT-0f172a?style=for-the-badge&logo=opensourceinitiative&logoColor=22C55E" />
<img src="https://img.shields.io/github/stars/MusaIslamFahad/sleep-disorder-prediction?style=for-the-badge&logo=github&label=GitHub%20Stars&color=0f172a" />

</p>

![Sleep Disorder Prediction Banner](https://raw.githubusercontent.com/MusaIslamFahad/sleep-disorder-prediction/main/assets/banner.png)

</div>

---

## 📖 Overview
 
**Sleep Health Classifier** is an end-to-end machine learning project that predicts whether a person has **Insomnia**, **Sleep Apnea**, or **No Disorder** based on real-world health and lifestyle data. The notebook covers the full supervised classification pipeline- EDA, feature engineering, multi-model benchmarking with 5-fold stratified cross-validation, hyperparameter tuning via GridSearchCV, and a final test-set evaluation.
 
Six classifiers are trained and compared. **Logistic Regression and SVM both achieve 97.33% test accuracy**, with SVM leading on cross-validation (89.62%). The project also includes clinically meaningful engineered features like Pulse Pressure, Sleep Efficiency, and Activity-to-Stress Ratio that go beyond the raw dataset columns.
 
> 💤 Sleep disorders affect over **1 billion people** worldwide and are linked to cardiovascular disease, obesity, and mental health issues. Early detection through lifestyle data can enable timely intervention.
 
---
 
## 📓 Open the Notebook
 
| Viewer | Link |
|--------|------|
| **NBViewer** (static render) | [View on NBViewer](https://nbviewer.org/github/MusaIslamFahad/sleep-disorder-prediction/blob/main/sleep-disorder-prediction.ipynb) |
| **Google Colab** (run in browser) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MusaIslamFahad/sleep-disorder-prediction/blob/main/sleep-disorder-prediction.ipynb) |
 
---

# 🖼️ Results & Visualizations
This section showcases the key visual outputs, evaluation results, and model insights generated during the **Sleep Disorder Prediction** project.

---

## 📊 Exploratory Data Analysis (EDA)

| Numeric Feature Distributions                               | Key Feature Relationships with Sleep Disorder               |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| <img src="Result image/__results___16_0.png" width="100%"/> | <img src="Result image/__results___18_0.png" width="100%"/> |

| Occupation vs Sleep Disorder (sorted by disorder prevalence) | Pairplot of Key Features                                    |
| ------------------------------------------------------------ | ----------------------------------------------------------- |
| <img src="Result image/__results___20_0.png" width="100%"/>  | <img src="Result image/__results___22_0.png" width="100%"/> |

| Correlation Matrix (Lower Triangle)                         |
| ----------------------------------------------------------- |
| <img src="Result image/__results___24_0.png" width="100%"/> |

---

## 🤖 Model Performance & Evaluation

| Model Comparison - 5-Fold Cross-Validation Accuracy        | Tuned XGBoost - Test Set Performance                        |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| <img src="Result image/__results___30_0.png" width="100%"/> | <img src="Result image/__results___35_0.png" width="100%"/> |

| All Models - Test Accuracy & F1 Score                       | Permutation Feature Importance - XGBoost (Tuned)            |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| <img src="Result image/__results___39_0.png" width="100%"/> | <img src="Result image/__results___41_0.png" width="100%"/> |

---

## 🔍 Feature Importance & Explainability

| SHAP Feature Importance - XGBoost Tuned (Mean \|SHAP\|)    | SHAP Summary Plot - Class: Insomnia                              |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| <img src="Result image/__results___45_2.png" width="100%"/> | <img src="Result image/__results___46_1.png" width="100%"/> |

| SHAP Summary Plot - Class: Sleep Apnea                      | SHAP Summary Plot - Class: None                             |
| ----------------------------------------------------------- | ----------------------------------------------------------- |
| <img src="Result image/__results___46_3.png" width="100%"/> | <img src="Result image/__results___46_5.png" width="100%"/> |

| SHAP Force Plot - Single Prediction Example                 |
| ----------------------------------------------------------- |
| <img src="Result image/__results___47_1.png" width="100%"/> |

---

## ✨ Highlights

* Comprehensive **Exploratory Data Analysis (EDA)**
* Multi-model machine learning benchmarking
* Hyperparameter tuning for performance optimization
* Feature importance visualization
* SHAP-based explainability analysis
* Interactive prediction system for real-world usage

---

> _All visualizations and results were generated from the Sleep Disorder Prediction notebook._
 
---
 
## 🎯 Target Classes
 
| Class | Description |
|-------|-------------|
| `None` | No sleep disorder detected |
| `Insomnia` | Difficulty falling or staying asleep |
| `Sleep Apnea` | Breathing interruptions during sleep |
 
---
 
## 📊 Dataset
 
**Source:** [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
**Size:** 374 records · 13 raw features · included in repo (no separate download needed)
 
| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Integer | Age in years |
| `Gender` | Categorical | Male / Female |
| `Occupation` | Categorical | Job type |
| `Sleep Duration` | Float | Hours of sleep per night |
| `Quality of Sleep` | Integer | Self-rated quality (1–10) |
| `Physical Activity Level` | Integer | Daily activity minutes |
| `Stress Level` | Integer | Self-rated stress (1–10) |
| `BMI Category` | Categorical | Normal / Overweight / Obese |
| `Blood Pressure` | String | `"Systolic/Diastolic"` — split into two numeric columns |
| `Heart Rate` | Integer | Resting heart rate (bpm) |
| `Daily Steps` | Integer | Steps per day |
| `Sleep Disorder` | Categorical | **Target**- None / Insomnia / Sleep Apnea |
 
---
 
## ⚙️ Feature Engineering
 
Four clinically meaningful features are derived from the raw columns to improve model performance:
 
| Engineered Feature | Formula | Clinical Meaning |
|--------------------|---------|-----------------|
| `Pulse_Pressure` | `Systolic_BP − Diastolic_BP` | Cardiovascular risk marker |
| `Sleep_Efficiency` | `Quality of Sleep ÷ Sleep Duration` | Quality per hour of sleep |
| `Activity_Stress_Ratio` | `Physical Activity ÷ (Stress Level + 1)` | Balance of activity vs. stress |
| `High_BP_Flag` | `1 if Systolic ≥ 130 else 0` | Binary hypertension indicator |
 
---
 
## 🔬 ML Pipeline
 
```
Raw CSV (Sleep_health_and_lifestyle_dataset.csv)
        │
        ▼
① Load & Inspect
        │  → Shape, dtypes, null counts, class distribution
        ▼
② Data Preprocessing
        │  → Fill Sleep Disorder NaN → "None"
        │  → Split "Blood Pressure" string → Systolic_BP + Diastolic_BP (numeric)
        │  → Drop Person ID
        ▼
③ Feature Engineering
        │  → Pulse_Pressure, Sleep_Efficiency
        │  → Activity_Stress_Ratio, High_BP_Flag
        ▼
④ Exploratory Data Analysis (EDA)
        │  → Disorder distribution (pie + countplot)
        │  → Univariate histograms (9 numeric features)
        │  → Disorder vs Gender, BMI, Age, Occupation
        │  → Pairplot (Age, Sleep Duration, Stress, Systolic_BP)
        │  → Correlation heatmap
        ▼
⑤ Encode & Scale
        │  → Label encode: Gender, Occupation, BMI Category, Sleep Disorder
        │  → Train / Test split — 80/20, stratified, random_state=42
        │  → StandardScaler for distance-based models (LR, KNN, SVM)
        ▼
⑥ Multi-Model Benchmarking
        │  → 6 classifiers × 5-fold Stratified K-Fold CV
        │  → Ranked by CV Mean Accuracy
        ▼
⑦ Hyperparameter Tuning
        │  → GridSearchCV on SVM (C, gamma, kernel)
        │  → GridSearchCV on XGBoost (n_estimators, max_depth)
        ▼
⑧ Final Test-Set Evaluation
        │  → Confusion Matrix, Classification Report
        │  → ROC Curve (multi-class One-vs-Rest)
        ▼
⑨ Full Model Comparison Summary
           → All 7 models ranked by Test Accuracy & Weighted F1
```
 
---
 
## 📈 Results
 
### Cross-Validation Ranking (5-Fold Stratified)
 
| Rank | Model | CV Mean Accuracy | CV Std |
|------|-------|-----------------|--------|
| 🥇 1 | **Support Vector Machine** | **89.62%** | ±2.01% |
| 2 | XGBoost | 88.63% | ±2.22% |
| 3 | Logistic Regression | 88.60% | ±3.83% |
| 4 | Random Forest | 88.29% | ±2.11% |
| 5 | K-Nearest Neighbors | 87.60% | ±4.89% |
| 6 | Decision Tree | 85.63% | ±4.99% |
 
---
 
### Test Set Performance (Final Evaluation)
 
| Rank | Model | Test Accuracy | F1 (Weighted) |
|------|-------|--------------|---------------|
| 🥇 1 | **Logistic Regression** | **97.33%** | **97.35%** |
| 🥇 1 | **Support Vector Machine** | **97.33%** | 97.32% |
| 3 | K-Nearest Neighbors | 96.00% | 96.00% |
| 3 | Random Forest | 96.00% | 96.06% |
| 3 | ★ XGBoost (Tuned) | 96.00% | 96.00% |
| 6 | Decision Tree | 94.67% | 94.78% |
| 7 | XGBoost (untuned) | 93.33% | 93.45% |
 
---
 
### Key Takeaways
 
- **Logistic Regression and SVM tie at 97.33% test accuracy**: the top performers on unseen data
- **SVM leads cross-validation (89.62%)**: most consistent across all 5 folds
- **Hyperparameter tuning matters**: XGBoost jumps from 93.33% → 96.00% after GridSearchCV
- **Decision Tree overfits**: highest variance (CV Std ±4.99%), lowest generalization
- The strong overall performance reflects how well `Stress Level`, `BMI Category`, `Sleep Duration`, and the engineered `Pulse_Pressure` separate the three disorder classes
---
 
## 🛠️ Tech Stack
 
| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations |
| `matplotlib` / `seaborn` | EDA visualizations and results plots |
| `scikit-learn` | Preprocessing, all classical models, GridSearchCV, evaluation |
| `xgboost` | Gradient boosting classifier + hyperparameter tuning |
| `Jupyter Notebook` | Interactive development environment |
 
---
 
## ⚙️ Requirements
 
- Python 3.9+
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `jupyter`
---
 
## 🚀 Getting Started
 
**1. Clone the repository**
```bash
git clone https://github.com/MusaIslamFahad/sleep-disorder-prediction.git
cd sleep-disorder-prediction
```
 
**2. (Optional) Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```
 
**3. Install dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```
 
**4. Launch the notebook**
```bash
jupyter notebook sleep-disorder-prediction.ipynb
```
 
> The dataset `Sleep_health_and_lifestyle_dataset.csv` is already included; no external download needed. If running on Kaggle, the notebook auto-detects the Kaggle dataset path.
 
---
 
## 📂 Project Structure
 
```
sleep-health-classifier/
│
├── sleep-disorder-prediction.ipynb          # Main notebook — full 9-step pipeline
├── Sleep_health_and_lifestyle_dataset.csv   # Dataset (included)
│
├── images/
│   ├── confusion_matrix.png                 # Confusion matrix of the best model
│   ├── roc_curve.png                        # Multi-class ROC curve (OvR)
│   └── feature_importance.png              # Feature importances plot
│
└── README.md
```
 
---
 
## 📚 What You'll Learn
 
This notebook is a strong portfolio reference for:
 
- **Parsing mixed-format strings**: splitting `"120/80"` Blood Pressure into two numeric columns
- **Clinical feature engineering**: creating Pulse Pressure, Sleep Efficiency, and Activity-Stress Ratio from domain knowledge
- **Multi-class classification**: handling 3 target labels (None / Insomnia / Sleep Apnea) cleanly
- **Stratified K-Fold CV**: ensuring each fold preserves class balance for fair comparison
- **GridSearchCV hyperparameter tuning**: applied to both SVM and XGBoost
- **Multi-class ROC curves**: One vs Rest strategy for three-class AUC
- **Full model comparison table**: ranking all 7 models by test accuracy and weighted F1 in one view
---
 
## 🔮 Future Enhancements
 
- 🧠 **SHAP explainability**: show which features drive each individual prediction
- 🌐 **Streamlit web app**: let users input their own health data and get a live prediction
- 📊 **Extended cross-validation**: repeated stratified K-Fold for tighter confidence intervals
- 📱 **REST API**: Flask endpoint for integration into health monitoring apps
- 📈 **Larger & more diverse dataset**: improve generalization across demographics
- 🏥 **Clinical validation**: test predictions against clinician-diagnosed ground truth
  
---
 
## 🤝 Contributing
 
Contributions are welcome!
 
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request
---
 
## 🙏 Acknowledgments
 
- [Kaggle - Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) for the data
- [Scikit-learn](https://scikit-learn.org/) for the complete ML toolkit
- [XGBoost](https://xgboost.readthedocs.io/) for the gradient boosting implementation
---
 
## 👨‍💻 Author
 
**Musa Islam Fahad**
- GitHub: [@MusaIslamFahad](https://github.com/MusaIslamFahad)
---
 
> ⭐ If this project helped your learning or research, a star would mean a lot. Thank you!
 
