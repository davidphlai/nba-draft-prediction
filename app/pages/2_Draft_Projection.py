import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title='Draft Projection', page_icon='🏀', layout='wide')

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import setup_logo
setup_logo()

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {0: 'Undrafted', 1: '1st Round', 2: '2nd Round'}
CLASS_COLORS = {0: '#6c757d', 1: '#28a745', 2: '#ffc107'}
CLASS_EMOJI  = {0: '⬜', 1: '🟢', 2: '🟡'}

RADAR_STATS = ['pts', 'treb', 'ast', 'stl', 'blk', 'eFG', 'usg', 'bpm']
RADAR_LABELS = ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks', 'eFG%', 'Usage', 'BPM']

#normalizing radar to 0–100
RADAR_RANGE = {
    'pts': (0.0, 20.0),
    'treb': (0.0, 8.0),
    'ast': (0.0, 4.5),
    'stl': (0.0, 1.8),
    'blk': (0.0, 1.5),
    'eFG': (30.0, 65.0),
    'usg': (8.0, 30.0),
    'bpm': (-16.0, 10.0)
}

#Training-data medians per class (pre-computed)
CLASS_MEDIANS = {
    'Undrafted': {'pts':4.1,'treb':2.1,'ast':0.6,'stl':0.4,'blk':0.1,'eFG':47.0,'usg':18.1,'bpm':-2.4},
    '2nd Round': {'pts':15.0,'treb':5.8,'ast':1.9,'stl':0.9,'blk':0.6,'eFG':53.8,'usg':24.4,'bpm':6.6},
    '1st Round': {'pts':16.0,'treb':6.2,'ast':2.0,'stl':1.1,'blk':0.8,'eFG':53.8,'usg':25.1,'bpm':8.1}
}

def normalize(val, lo, hi):
    #normalize val to 0–100 for radar chart.
    return float(np.clip((val - lo) / (hi - lo) * 100, 0, 100))

@st.cache_data
def load_data():
    preds = pd.read_csv(ROOT / 'outputs/mlp/mid_checkpoint/mlp_predictions_test.csv')
    stats = pd.read_csv(ROOT / 'dataset/NBA_Test.csv')
    merged = stats.merge(preds, on=['player_name', 'year'], how='left')
    merged = merged.drop_duplicates(subset=['player_name', 'year']).reset_index(drop=True)
    return merged

def player_label(row):
    return f"{row['player_name']} ({int(row['year'])})"

def show_prediction_card(prob_undrafted, prob_1st, prob_2nd):
    probs = np.array([prob_undrafted, prob_1st, prob_2nd])
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    color = CLASS_COLORS[pred_class]

    st.markdown(
        f'''
        <div style='background-color:{color}22; border:2px solid {color};
             border-radius:12px; padding:20px; text-align:center; margin-bottom:16px;'>
            <div style='font-size:3rem;'>{CLASS_EMOJI[pred_class]}</div>
            <div style='font-size:2rem; font-weight:bold; color:{color};'>{CLASS_NAMES[pred_class]}</div>
            <div style='font-size:1.2rem; color:#888;'>Confidence: {confidence:.1%}</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

    fig = go.Figure(go.Bar(
        x=[prob_1st, prob_2nd, prob_undrafted],
        y=['1st Round', '2nd Round', 'Undrafted'],
        orientation='h',
        marker_color=[CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[0]],
        text=[f'{p:.1%}' for p in [prob_1st, prob_2nd, prob_undrafted]],
        textposition='outside'
    ))
    fig.update_layout(
        title='Draft Probability',
        xaxis=dict(range=[0, 1.05], tickformat='.0%'),
        height=200,
        margin=dict(l=10, r=70, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

def show_radar(row):
    vals_player = [normalize(float(row[s]), *RADAR_RANGE[s]) if s in row.index and pd.notna(row[s]) else 0
                   for s in RADAR_STATS]

    fig = go.Figure()

    #Reference traces: class medians — drawn first so player sits on top
    ref_styles = [
        ('Undrafted avg', CLASS_COLORS[0], 'dot', 1.5, 0.5),
        ('2nd Round avg', CLASS_COLORS[2], 'solid', 3.0, 0.8),
        ('1st Round avg', CLASS_COLORS[1], 'solid', 3.0, 0.9)
    ]
    for label, color, dash, width, opacity in ref_styles:
        cls_name = label.replace(' avg', '')
        vals_ref = [normalize(CLASS_MEDIANS[cls_name][s], *RADAR_RANGE[s])
                    for s in RADAR_STATS]
        fig.add_trace(go.Scatterpolar(
            r=vals_ref + [vals_ref[0]],
            theta=RADAR_LABELS + [RADAR_LABELS[0]],
            mode='lines', name=label,
            line=dict(color=color, dash=dash, width=width),
            opacity=opacity
        ))

    #Player trace (filled, drawn last so it appears on top)
    fig.add_trace(go.Scatterpolar(
        r=vals_player + [vals_player[0]],
        theta=RADAR_LABELS + [RADAR_LABELS[0]],
        mode='lines+markers', name=row['player_name'],
        line=dict(color='#4e79a7', width=3),
        marker=dict(size=6),
        fill='toself',
        fillcolor='rgba(78,121,167,0.2)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], ticksuffix='%', tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=12))
        ),
        showlegend=True,
        height=480,
        margin=dict(l=60, r=60, t=60, b=60),
        title='Stat Profile vs Class Averages  (double-click to reset zoom)',
        legend=dict(orientation='h', yanchor='bottom', y=-0.18, xanchor='center', x=0.5),
        dragmode=False
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={'scrollZoom': False, 'doubleClick': 'reset', 'displayModeBar': False}
    )

def show_bar_profile(row):
    display = ['GP', 'pts', 'treb', 'ast', 'stl', 'blk', 'eFG', 'TS_per', 'FT_per', 'TP_per', 'usg', 'bpm']
    labels_map = {
        'GP':'Games','pts':'Points','treb':'Rebounds','ast':'Assists',
        'stl':'Steals','blk':'Blocks','eFG':'eFG%','TS_per':'TS%',
        'FT_per':'FT%','TP_per':'3P%','usg':'Usage','bpm':'BPM'
    }
    cols = [c for c in display if c in row.index and pd.notna(row[c])]
    labels = [labels_map.get(c, c) for c in cols]
    values = [round(float(row[c]), 1) for c in cols]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color='#4e79a7',
        text=values, textposition='outside'
    ))
    fig.update_layout(
        title='Raw Stats',
        height=300,
        margin=dict(l=10, r=10, t=40, b=40),
        yaxis_title='Value'
    )
    st.plotly_chart(fig, use_container_width=True)

#Page
st.title('🏀 Draft Projection')
st.markdown("Search any player from the **2021** test season to see the MLP's draft prediction.")

df = load_data()

st.markdown('---')

#Autocomplete search
search = st.text_input(
    'Search player name',
    placeholder='Start typing a name...',
    label_visibility='collapsed'
)

if search.strip():
    mask = df['player_name'].str.contains(search.strip(), case=False, na=False)
    results = df[mask]
else:
    results = pd.DataFrame()

if results.empty and search.strip():
    st.warning(f"No players found for '{search}'. Try a shorter or different name.")
    st.stop()

if not results.empty:
    options = [player_label(r) for _, r in results.iterrows()]
    selected_label = st.selectbox(
        'Matching players',
        options,
        label_visibility='collapsed'
    )
    idx = options.index(selected_label)
    row = results.iloc[idx]

    st.markdown('---')

    #Player info + prediction card
    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"### {row['player_name']}")
        yr_label = {1:'Freshman',2:'Sophomore',3:'Junior',4:'Senior'}.get(
            int(row['yr_num']) if 'yr_num' in row.index and pd.notna(row.get('yr_num')) else 0, '')
        for label, col in [('Year', 'year'), ('Team', 'team'), ('Conference', 'conf'), ('Role', 'role')]:
            if col in row.index and pd.notna(row[col]):
                st.markdown(f'**{label}:** {row[col]}')
        if yr_label:
            st.markdown(f'**Class:** {yr_label}')
        if 'true_draft_status' in row.index and pd.notna(row.get('true_draft_status')):
            actual = CLASS_NAMES.get(int(row['true_draft_status']), 'Unknown')
            emoji  = CLASS_EMOJI.get(int(row['true_draft_status']), '')
            st.markdown(f'**Actual outcome:** {emoji} {actual}')

    with right:
        if pd.notna(row.get('prob_undrafted')):
            show_prediction_card(row['prob_undrafted'], row['prob_1st_round'], row['prob_2nd_round'])
        else:
            st.warning('Prediction not available for this player.')

    st.markdown('---')

    #Visualizations - 3 charts
    st.markdown('#### 📡 Radar Chart')
    show_radar(row)

    st.markdown('#### 📊 Key Stats')
    show_bar_profile(row)

    st.markdown('#### 🗂 Full Stats')
    st.dataframe(row.to_frame().T, use_container_width=True)

else:
    st.info('Type a player name above to search. The dropdown will show matching results.')
    st.markdown('**Example searches:** Cade Cunningham, Jalen Suggs, Franz Wagner, Evan Mobley')