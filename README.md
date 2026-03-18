# 🏦 Universal Bank — Personal Loan Intelligence Dashboard

A full-stack Streamlit analytics application for predicting personal loan acceptance and enabling hyper-personalised marketing campaigns.

---

## 🚀 Live Demo — Deploy on Streamlit.io

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Universal Bank Loan Intelligence App"
git branch -M main
git remote add origin https://github.com/<your-username>/universal-bank-loan-app.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"New app"**
3. Select your GitHub repo and branch
4. Set **Main file path** to `app.py`
5. Click **Deploy**

---

## 📁 Project Structure
```
universal_bank_app/
├── app.py                  # Main Streamlit application
├── UniversalBank.csv       # Training dataset (5,000 records)
├── sample_test_data.csv    # Sample test file for predictions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 📊 Dashboard Sections

| Tab | Description |
|-----|-------------|
| 📋 Data Overview | Dataset preview, feature dictionary, descriptive stats |
| 📊 Descriptive Analytics | Distribution charts, class balance analysis |
| 🔍 Exploratory Analysis | Conversion rates, scatter plots, correlation heatmap |
| 🤖 ML Models & Performance | Decision Tree, Random Forest, Gradient Boosted Tree — metrics table, ROC curve, confusion matrices, feature importance |
| 🎯 Prescriptive Analytics | Customer segmentation, campaign playbook, budget allocation |
| 🔮 Predict New Data | Upload CSV → get predictions → download results |

---

## 🔧 Local Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/universal-bank-loan-app.git
cd universal-bank-loan-app

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

---

## 📦 Dependencies
- `streamlit` — Web app framework
- `pandas` / `numpy` — Data processing
- `scikit-learn` — ML models (Decision Tree, Random Forest, Gradient Boosted Tree)
- `imbalanced-learn` — SMOTE for class imbalance handling
- `plotly` — Interactive charts

---

## 🎯 Key Insights Built Into the App
- **Income** is the strongest predictor of loan acceptance (top-quartile: 35.6% conversion)
- **CD Account holders** convert at 46.4% — prime campaign targets
- **SMOTE oversampling** handles the 9.4:1 class imbalance
- **4 customer segments** ranked by conversion probability for budget-optimised targeting

---

## 📝 Notes
- Negative `Experience` values (52 records) are clipped to 0 automatically
- `ID` and `ZIP Code` are excluded from model features
- All models evaluated on a 20% stratified holdout test set

---

*Built for Universal Bank Marketing Team · 2024*
