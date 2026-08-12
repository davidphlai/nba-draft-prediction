import sys
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title='What-if Simulator', page_icon='🧪', layout='wide')

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import setup_logo
setup_logo()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'models'))

from mlp_from_scratch import CLASS_NAMES, forward
from mlp_inference import load_preprocessor

CLASS_COLORS = {0: '#6c757d', 1: '#28a745', 2: '#ffc107'}
CLASS_EMOJI = {0: '⬜', 1: '🟢', 2: '🟡'}
ARTIFACT_DIR = ROOT / 'outputs/mlp/mid_checkpoint'

TEMPLATE_OPTIONS = {
    'Average College Player': ('undrafted', 0),
    'Borderline Prospect (2nd Round range)': ('2nd_round', 2),
    'Top Prospect (1st Round range)': ('1st_round', 1)
}

@st.cache_resource
def load_model_artifacts():
    preprocessor = load_preprocessor(ARTIFACT_DIR / 'mlp_preprocessing.json')
    model = np.load(ARTIFACT_DIR / 'mlp_model.npz')
    results = json.loads((ARTIFACT_DIR / 'mlp_results.json').read_text())
    params = {k: model[k] for k in ['W1', 'b1', 'W2', 'b2']}
    activation = results['best_run']['hyperparameters']['activation']
    return preprocessor, params, activation

@st.cache_data
def get_class_baselines(numeric_cols_tuple):
    '''Median feature row for each draft class, computed from training data.'''
    train   = pd.read_csv(ROOT / 'dataset/NBA_Train.csv')
    numeric_cols = list(numeric_cols_tuple)
    baselines, default_cats = {}, {}
    for key, cls in [('undrafted', 0), ('2nd_round', 2), ('1st_round', 1)]:
        subset = train[train['draft_status'] == cls]
        baselines[key] = subset[numeric_cols].median().to_dict()
        default_cats[key] = {
            'conf': subset['conf'].mode()[0],
            'role': subset['role'].mode()[0]
        }
    return baselines, default_cats

@st.cache_data
def get_categories():
    raw = json.loads((ARTIFACT_DIR / 'mlp_preprocessing.json').read_text())
    return raw['categories']

# Use the trained MLP for inference
def predict(preprocessor, params, activation, row_df):
    x = preprocessor.transform(row_df)
    return forward(params, x, activation)['probs'][0]

def show_prediction_card(probs):
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    color = CLASS_COLORS[pred_class]

    st.markdown(
        f'''
        <div style='background-color:{color}22; border:2px solid {color};
             border-radius:12px; padding:20px; text-align:center; margin-bottom:16px;'>
            <div style='font-size:3rem;'>{CLASS_EMOJI[pred_class]}</div>
            <div style='font-size:2rem; font-weight:bold; color:{color};'>
                {CLASS_NAMES[pred_class]}
            </div>
            <div style='font-size:1.2rem; color:#888;'>Confidence: {confidence:.1%}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    fig = go.Figure(go.Bar(
        x=[probs[1], probs[2], probs[0]],
        y=['1st Round', '2nd Round', 'Undrafted'],
        orientation='h',
        marker_color=[CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[0]],
        text=[f'{p:.1%}' for p in [probs[1], probs[2], probs[0]]],
        textposition='outside'
    ))
    fig.update_layout(
        title='Draft Probability',
        xaxis=dict(range=[0, 1.1], tickformat='.0%'),
        height=200,
        margin=dict(l=10, r=70, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

# Page
st.title('🧪 What-if Simulator')
st.markdown(
    'Design a hypothetical prospect. Choose a **template** to set realistic baseline '
    'stats for that tier, then fine-tune the per-game numbers you care about.'
)

try:
    preprocessor, params, activation = load_model_artifacts()
    baselines, default_cats = get_class_baselines(tuple(preprocessor.numeric_cols))
    categories = get_categories()
except Exception as e:
    st.error(f'Could not load model: {e}')
    st.stop()

st.markdown('---')

template_label = st.selectbox(
    'Player Template',
    options=list(TEMPLATE_OPTIONS.keys()),
    index=0,
    help='Sets ALL baseline stats (including cumulative season totals) to the '
         'real-data median for that draft tier. You can override individual stats below.'
)
template_key, _ = TEMPLATE_OPTIONS[template_label]

st.markdown('---')

# Sliders
base = baselines[template_key]
dcat = default_cats[template_key]

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader('Player Profile')
    yr_num = st.selectbox('Class Year', [1, 2, 3, 4],
                           format_func=lambda x: {1:'Freshman',2:'Sophomore',3:'Junior',4:'Senior'}[x],
                           index=int(base.get('yr_num', 2)) - 1)
    ht_inches = st.slider('Height (inches)', 66, 90, int(base.get('ht_inches', 78)))
    rec_rank = st.slider('Recruiting Rank (out of high school)', 1, 999,
                          int(base.get('Rec Rank', 500)),
                          help='1 = top recruit. 1st Round median ≈ 96 · Undrafted median ≈ 999')
    conf = st.selectbox('Conference', sorted(categories['conf']),
                        index=sorted(categories['conf']).index(dcat['conf'])
                        if dcat['conf'] in categories['conf'] else 0)
    role = st.selectbox('Role / Position', categories['role'],
                        index=categories['role'].index(dcat['role'])
                        if dcat['role'] in categories['role'] else 0)

    st.subheader('Playing Time')
    gp = st.slider('Games Played', 0, 40, int(base.get('GP', 28)))
    min_per = st.slider('Minutes % (team share)', 0.0, 100.0,
                        round(float(base.get('Min_per', 37)), 1), step=0.5)

with col_right:
    st.subheader('Scoring')
    pts = st.slider('Points per game', 0.0, 40.0, round(float(base.get('pts', 14)), 1), step=0.5)
    efg = st.slider('Effective FG%', 0.0, 80.0, round(float(base.get('eFG', 50)), 1), step=0.5)
    ts = st.slider('True Shooting%', 0.0, 80.0, round(float(base.get('TS_per', 55)), 1), step=0.5)
    tp = st.slider('3-Point%', 0.0, 60.0, round(float(base.get('TP_per', 35)), 1), step=0.5)
    ft = st.slider('Free Throw%', 0.0, 100.0, round(float(base.get('FT_per', 72)), 1), step=0.5)
    usg = st.slider('Usage Rate', 0.0, 40.0, round(float(base.get('usg', 20)), 1), step=0.5)

    st.subheader('Other Skills')
    treb = st.slider('Rebounds per game', 0.0, 20.0, round(float(base.get('treb', 6)), 1), step=0.5)
    ast = st.slider('Assists per game', 0.0, 15.0, round(float(base.get('ast', 3)), 1), step=0.5)
    stl = st.slider('Steals per game', 0.0, 5.0, round(float(base.get('stl', 1)), 1), step=0.1)
    blk = st.slider('Blocks per game', 0.0, 5.0, round(float(base.get('blk', 1)), 1), step=0.1)
    bpm = st.slider('Box Plus/Minus', -10.0, 15.0, round(float(base.get('bpm', 2)), 1), step=0.5,
                     help='Per-100-possession value above average. 1st Round median ≈ 8')

# Run prediction
run_btn = st.button('▶ Run Prediction', type='primary', use_container_width=True)

if run_btn:
    row = dict(base)
    row.update({
        'GP': float(gp), 'Min_per': float(min_per), 'pts': float(pts),
        'treb': float(treb), 'ast': float(ast), 'stl': float(stl), 'blk': float(blk),
        'eFG': float(efg), 'TS_per': float(ts), 'TP_per': float(tp),
        'FT_per': float(ft), 'usg': float(usg), 'yr_num': float(yr_num),
        'ht_inches': float(ht_inches), 'bpm': float(bpm),
        'obpm': float(bpm) * 0.6, 'dbpm': float(bpm) * 0.4,
        'Rec Rank': float(rec_rank)
    })

    row_df = pd.DataFrame([row])
    row_df['team'] = '__OTHER__'
    row_df['conf'] = conf
    row_df['role'] = role

    st.markdown('---')
    st.subheader('Prediction Result')

    try:
        probs = predict(preprocessor, params, activation, row_df)
        show_prediction_card(probs)
    except Exception as e:
        st.error(f'Prediction failed: {e}')
        st.exception(e)