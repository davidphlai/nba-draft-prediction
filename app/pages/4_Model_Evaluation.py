import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title='Model Evaluation', page_icon='📈', layout='wide')

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import setup_logo
setup_logo()

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = ['Undrafted', '1st Round', '2nd Round']
CLASS_COLORS = ['#6c757d', '#28a745', '#ffc107']

#Get every results
@st.cache_data
def load_lr_preds():
    return pd.read_csv(ROOT / 'outputs/LogisticRegression/lr_test_predictions.csv')

@st.cache_data
def load_mlp_loss():
    return pd.read_csv(ROOT / 'outputs/mlp/mlp_training_validation_loss.csv')

@st.cache_data
def load_mlp_preds():
    return pd.read_csv(ROOT / 'outputs/mlp/mid_checkpoint/mlp_predictions_test.csv')

@st.cache_data
def load_knn_preds():
    return pd.read_csv(ROOT / 'outputs/knn/knn_predictions_test.csv')

def confusion_matrix_from_preds(y_true, y_pred, n_classes=3):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if (0 <= int(t) < n_classes) and (0 <= int(p) < n_classes):
            cm[int(t)][int(p)] += 1
    return cm

def plot_confusion_matrix(cm, title, class_names=CLASS_NAMES):
    z = cm.astype(float)
    total = cm.sum()
    text = [[f'{cm[i][j]}<br>({100*cm[i][j]/max(cm[i].sum(),1):.1f}%)'
              for j in range(len(class_names))] for i in range(len(class_names))]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=[f'Pred: {c}' for c in class_names],
        y=[f'True: {c}' for c in class_names],
        text=text,
        texttemplate='%{text}',
        colorscale='Blues',
        showscale=True
    ))
    fig.update_layout(
        title=title,
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    return fig

def per_class_metrics(cm):
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        rows.append({
            'Class': cls,
            'Support': int(cm[i, :].sum()),
            'Precision': round(prec, 3),
            'Recall': round(recall, 3),
            'F1': round(f1, 3)
        })
    return pd.DataFrame(rows).set_index('Class')

#Page
st.title('📈 Model Evaluation')
st.markdown(
    'Compare all three from-scratch classifiers on the **2021 test set**. '
    'All models use NumPy only — no scikit-learn, TensorFlow, or PyTorch.'
)

#Summary table
st.markdown('---')
st.subheader('Overall Performance Summary')

summary = pd.DataFrame([
    {'Model': 'Logistic Regression', 'Test Accuracy': '94.71%', 'Test Macro-F1': '0.4427', 'Test AUROC': '0.9416', 'Selected': ''},
    {'Model': 'KNN (k=3)', 'Test Accuracy': '96.81%', 'Test Macro-F1': '0.4702', 'Test AUROC': '0.7220', 'Selected': ''},
    {'Model': 'MLP (1 hidden, 64 units)', 'Test Accuracy': '97.86%', 'Test Macro-F1': '0.5306', 'Test AUROC': '0.9520', 'Selected': 'Best'}
]).set_index('Model')

st.dataframe(summary, use_container_width=True)
st.info(
    '**Why MLP?** The MLP achieves the highest Macro-F1 (0.5306) and AUROC (0.9520), '
    'providing the strongest overall balance across the three draft classes. '
    'KNN performs worse on the rare classes, while logistic regression is limited by its linear decision boundary.'
)

#Confusion matrices
st.markdown('---')
st.subheader('Confusion Matrices (Test Set)')

#Load predictions
lr_preds = load_lr_preds()
knn_preds = load_knn_preds()
mlp_preds = load_mlp_preds()

cm_lr = confusion_matrix_from_preds(lr_preds['y_true'], lr_preds['y_pred'])
cm_knn = confusion_matrix_from_preds(knn_preds['y'], knn_preds['y_pred'])

#MLP confusion matrix columns
if 'prob_undrafted' in mlp_preds.columns:
    mlp_true = mlp_preds['true_draft_status'] if 'true_draft_status' in mlp_preds.columns else None
    mlp_pred_col = 'pred_draft_status' if 'pred_draft_status' in mlp_preds.columns else None
    if mlp_true is not None and mlp_pred_col is not None:
        cm_mlp = confusion_matrix_from_preds(mlp_true, mlp_preds[mlp_pred_col])
    else:
        #Hardcod
        cm_mlp = np.array([[4788, 32, 86], [5, 13, 8], [6, 2, 16]])
else:
    cm_mlp = np.array([[4788, 32, 86], [5, 13, 8], [6, 2, 16]])

col1, col2, col3 = st.columns(3)
with col1:
    st.plotly_chart(plot_confusion_matrix(cm_lr, 'Logistic Regression'), use_container_width=True)
with col2:
    st.plotly_chart(plot_confusion_matrix(cm_knn, 'KNN (k=3)'), use_container_width=True)
with col3:
    st.plotly_chart(plot_confusion_matrix(cm_mlp, 'MLP (selected)'), use_container_width=True)

st.caption(
    'Each cell shows the raw count and row-wise recall %. '
    'All models correctly classify most undrafted players but struggle with the rare drafted classes.'
)

#Per-class metrics
st.markdown('---')
st.subheader('Per-Class Metrics (Test Set)')

tab_lr, tab_knn, tab_mlp = st.tabs(['Logistic Regression', 'KNN (k=3)', 'MLP (selected)'])

with tab_lr:
    st.dataframe(per_class_metrics(cm_lr), use_container_width=True)

with tab_knn:
    st.dataframe(per_class_metrics(cm_knn), use_container_width=True)

with tab_mlp:
    st.dataframe(per_class_metrics(cm_mlp), use_container_width=True)

#MLP training curves
st.markdown('---')
st.subheader('MLP Training Curves')

loss_df = load_mlp_loss()

left, right = st.columns(2)

with left:
    fig_loss = go.Figure()
    fig_loss.add_trace(go.Scatter(
        x=loss_df['epoch'], y=loss_df['train_loss'],
        mode='lines', name='Train Loss',
        line=dict(color='#4e79a7', width=2)
    ))
    fig_loss.add_trace(go.Scatter(
        x=loss_df['epoch'], y=loss_df['val_loss'],
        mode='lines', name='Validation Loss',
        line=dict(color='#e15759', width=2, dash='dash')
    ))
    fig_loss.update_layout(
        title='Cross-Entropy Loss over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_loss, use_container_width=True)

with right:
    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Scatter(
        x=loss_df['epoch'], y=loss_df['train_macro_f1'],
        mode='lines', name='Train Macro-F1',
        line=dict(color='#4e79a7', width=2)
    ))
    fig_f1.add_trace(go.Scatter(
        x=loss_df['epoch'], y=loss_df['val_macro_f1'],
        mode='lines', name='Val Macro-F1',
        line=dict(color='#e15759', width=2, dash='dash')
    ))
    if 'val_binary_f1' in loss_df.columns:
        fig_f1.add_trace(go.Scatter(
            x=loss_df['epoch'], y=loss_df['val_binary_f1'],
            mode='lines', name='Val Binary F1 (any-drafted)',
            line=dict(color='#f28e2b', width=1.5, dash='dot')
        ))
    fig_f1.update_layout(
        title='Macro-F1 over Epochs',
        xaxis_title='Epoch',
        yaxis_title='F1 Score',
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_f1, use_container_width=True)

st.caption(
    'Best validation epoch: **45** · '
    'Early stopping patience: **30 epochs** · '
    'Best validation Macro-F1: **0.6557**'
)

#Draft-class score distributions
st.markdown('---')
st.subheader('MLP Predicted Probability Distributions')

if 'prob_undrafted' in mlp_preds.columns and 'true_draft_status' in mlp_preds.columns:
    prob_cols = {'Undrafted': 'prob_undrafted', '1st Round': 'prob_1st_round', '2nd Round': 'prob_2nd_round'}
    existing  = {k: v for k, v in prob_cols.items() if v in mlp_preds.columns}

    score_col = st.selectbox(
        'Score to visualize',
        list(existing.keys()),
        help="Distribution of the MLP's predicted probability for each true class"
    )
    col_name = existing[score_col]

    fig_dist = go.Figure()
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        subset = mlp_preds[mlp_preds['true_draft_status'] == cls_id][col_name].dropna()
        if len(subset) > 0:
            fig_dist.add_trace(go.Histogram(
                x=subset, name=cls_name,
                marker_color=CLASS_COLORS[cls_id],
                opacity=0.7,
                nbinsx=40
            ))
    fig_dist.update_layout(
        barmode='overlay',
        title=f'Distribution of P({score_col}) by True Class',
        xaxis_title=f'P({score_col})',
        yaxis_title='Count',
        height=380,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=-0.2)
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    st.caption(
        'Ideal separation: drafted players should cluster near 1.0, undrafted near 0.0. '
        'The severe class imbalance makes this challenging — note the large undrafted bar.'
    )