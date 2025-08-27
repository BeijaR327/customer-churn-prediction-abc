[README.md](https://github.com/user-attachments/files/22000357/README.md)
# Customer Churn Prediction – ABC Corporation

## 📌 Project Overview
This project builds predictive models to identify customers at **high risk of attrition** (leaving ABC Corporation).  
The goal is to assign each customer a **churn probability (0–1)** so ABC can take proactive retention actions.  
Models compared: Logistic Regression, Random Forest, and Gradient Boosting with hyperparameter tuning.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BeijaR327/customer-churn-prediction-abc/blob/main/notebooks/Churn_Dashboard_Colab.ipynb)

## 🚀 Installation & Usage

1. Clone this repository:
```bash
git clone https://github.com/BeijaR327/customer-churn-prediction-abc.git
cd customer-churn-prediction-abc
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate    # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the notebook:
```bash
jupyter notebook notebooks/modeling_pipeline.ipynb
```

## 📊 Results & Insights
- Logistic Regression provides a strong baseline with interpretable coefficients.
- Random Forest improves recall and highlights top predictors of churn.
- Gradient Boosting achieved the **best ROC-AUC** and balanced precision/recall.
- Feature importance shows customer service calls, contract type, and tenure as the strongest churn signals.

## ☁️ Run in Google Colab
Click below to launch the notebook without local setup:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BeijaR327/customer-churn-prediction-abc/blob/main/notebooks/modeling_pipeline.ipynb)

## 📂 Repository Structure
```
customer-churn-prediction-abc/
├── data/                # input dataset(s)
├── notebooks/
│   └── modeling_pipeline.ipynb
├── reports/
│   └── banner.png
├── requirements.txt
└── README.md
```

## ✨ Business Value
ABC Corporation can now:
- Identify high-risk customers before they churn.
- Prioritize retention campaigns toward those most likely to leave.
- Use interpretable features (tenure, contract type, call frequency) to inform strategy.

---
Made with ❤️ by Beija Richardson
