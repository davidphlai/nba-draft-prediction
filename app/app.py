import streamlit as st
from pathlib import Path
from utils import setup_logo

st.set_page_config(
    page_title='NBA Draft Predictor',
    page_icon='🏀',
    layout='wide'
)

setup_logo()

st.title('🏀 NBA Draft Predictor')

st.markdown('''
Predict whether an NCAA basketball player will be selected in the **NBA Draft** —
trained on 2009–2021 college basketball statistics from the Kaggle
*College Basketball 2009–2021 + NBA Advanced Stats* dataset.
''')

st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    st.markdown('''
### Navigate
Use the sidebar to access:

- **Data Overview** — Explore the dataset, feature statistics, and draft trends
- **Draft Projection** — Search historical players and inspect model-generated draft probabilities
- **What-if Simulator** — Design a hypothetical prospect and explore how player statistics affect draft probability
- **Model Evaluation** — Compare model performance, confusion matrices, and evaluation metrics
''')

with col2:
    st.markdown('### Team Model Performance')
    st.markdown('''
The team developed three classifiers from scratch using NumPy:

| Model | Test Macro-F1 | Multiclass AUROC |
|---|---:|---:|
| Logistic Regression | 0.4427 | 0.9416 |
| KNN (k=3) | 0.4702 | 0.7220 |
| **MLP (selected)** | **0.5306** | **0.9520** |

The MLP achieved the strongest overall performance and was selected for the final application.
Because the dataset is heavily imbalanced toward undrafted players, Macro-F1 and AUROC
were emphasized over accuracy alone.
''')

st.markdown('---')
st.markdown('''
**Target classes:** `0` Undrafted · `1` 1st Round · `2` 2nd Round  
**Features:** NCAA player statistics, recruiting information, and engineered basketball features  
**Course:** INFO 5368-030: Practical Applications in Machine Learning · Cornell Tech
''')
