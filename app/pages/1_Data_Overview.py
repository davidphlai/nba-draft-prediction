import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title='Data Overview', page_icon='📊', layout='wide')

import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import setup_logo
setup_logo()

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = {0: 'Undrafted', 1: '1st Round', 2: '2nd Round'}
CLASS_COLORS = {0: '#6c757d', 1: '#28a745', 2: '#ffc107'}

SPLIT_COUNTS = {
    'Train (2009–2018)': {'Undrafted': 16398, '1st Round': 239, '2nd Round': 209},
    'Validation (2019–2020)': {'Undrafted': 3548, '1st Round': 48, '2nd Round': 47},
    'Test (2021)': {'Undrafted': 4906, '1st Round': 26, '2nd Round': 24}
}

@st.cache_data
def load_train():
    return pd.read_csv(ROOT / 'dataset/NBA_Train.csv')

st.title('📊 Data Overview & EDA')
st.markdown(
    'Explore the dataset used to train and evaluate the NBA Draft classifiers. '
    'The data covers NCAA players from **2009 to 2021** with college statistics, '
    'recruiting rankings, and draft outcomes.'
)

df = load_train()

# Dataset summary
st.markdown('---')
st.subheader('Dataset Summary')

m1, m2, m3, m4 = st.columns(4)
m1.metric('Total Players (train)', f'{len(df):,}')
m2.metric('Features', '63 numeric + 3 categorical')
m3.metric('1st Round Players', f'{(df['draft_status']==1).sum()}')
m4.metric('2nd Round Players', f'{(df['draft_status']==2).sum()}')

# Dataset split
st.markdown('''
| Split | Years | Rows | Undrafted | 1st Round | 2nd Round |
|---|---|---|---|---|---|
| Train | 2009–2018 | 16,846 | 16,398 (97.3%) | 239 (1.4%) | 209 (1.2%) |
| Validation | 2019–2020 | 3,643 | 3,548 (97.4%) | 48 (1.3%) | 47 (1.3%) |
| Test | 2021 | 4,956 | 4,906 (99.0%) | 26 (0.5%) | 24 (0.5%) |
''')

# Class distribution
st.markdown('---')
st.subheader('Class Distribution')

left, right = st.columns([1, 1])

with left:
    # Stacked bar across splits
    splits = list(SPLIT_COUNTS.keys())
    classes = ['Undrafted', '1st Round', '2nd Round']
    colors = [CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]]

    fig = go.Figure()
    for cls, color in zip(classes, colors):
        totals = [SPLIT_COUNTS[s][cls] for s in splits]
        fig.add_trace(go.Bar(name=cls, x=splits, y=totals, marker_color=color))
    fig.update_layout(
        barmode='stack',
        title='Player Count by Split & Draft Class',
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation='h', y=-0.18)
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    # Pie chart — train split only
    counts = [16398, 239, 209]
    fig2 = go.Figure(go.Pie(
        labels=classes,
        values=counts,
        marker_colors=colors,
        textinfo='label+percent',
        hole=0.3
    ))
    fig2.update_layout(
        title='Training Set Class Balance',
        height=350,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.info(
    '**Severe class imbalance**: over 97% of college players go undrafted each year. '
    'All three models address this with class-weighted losses.'
)

# Key stats by draft class
st.markdown('---')
st.subheader('Key Stats by Draft Class')

stat_options = {
    'Points per game': 'pts',
    'Rebounds per game': 'treb',
    'Assists per game': 'ast',
    'Box Plus/Minus (BPM)': 'bpm',
    'Effective FG%': 'eFG',
    'Usage Rate': 'usg',
    'Recruiting Rank': 'Rec Rank',
    'Height (inches)': 'ht_inches'
}

stat_label = st.selectbox('Select stat to compare', list(stat_options.keys()), index=0)
stat_col   = stat_options[stat_label]

if stat_col in df.columns:
    fig3 = go.Figure()
    for cls_id, cls_name in CLASS_NAMES.items():
        subset = df[df['draft_status'] == cls_id][stat_col].dropna()
        fig3.add_trace(go.Box(
            y=subset,
            name=cls_name,
            marker_color=CLASS_COLORS[cls_id],
            boxmean=True
        ))
    fig3.update_layout(
        title=f'{stat_label} Distribution by Draft Class (Training Set)',
        yaxis_title=stat_label,
        height=400,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Median table
    rows = []
    for cls_id, cls_name in CLASS_NAMES.items():
        subset = df[df['draft_status'] == cls_id][stat_col].dropna()
        rows.append({
            'Draft Class': cls_name,
            'Median': round(float(subset.median()), 2),
            'Mean': round(float(subset.mean()), 2),
            'Std Dev': round(float(subset.std()), 2),
            'Count': len(subset)
        })
    st.dataframe(pd.DataFrame(rows).set_index('Draft Class'), use_container_width=True)

# Multi-stat median comparison
st.markdown('---')
st.subheader('Median Stat Profile by Draft Class')

PROFILE_STATS = {
    'pts': 'Points', 'treb': 'Rebounds', 'ast': 'Assists',
    'stl': 'Steals', 'blk': 'Blocks', 'eFG': 'eFG%',
    'usg': 'Usage', 'bpm': 'BPM'
}

stats_in_df = [c for c in PROFILE_STATS if c in df.columns]
medians_by_class = {}
for cls_id, cls_name in CLASS_NAMES.items():
    subset = df[df['draft_status'] == cls_id]
    medians_by_class[cls_name] = {c: round(float(subset[c].median()), 2) for c in stats_in_df}

fig4 = go.Figure()
for cls_name in CLASS_NAMES.values():
    fig4.add_trace(go.Bar(
        name=cls_name,
        x=[PROFILE_STATS[c] for c in stats_in_df],
        y=[medians_by_class[cls_name][c] for c in stats_in_df],
        marker_color=CLASS_COLORS[list(CLASS_NAMES.values()).index(cls_name)]
    ))
fig4.update_layout(
    barmode='group',
    title='Median Per-Game Stats by Draft Class',
    height=380,
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation='h', y=-0.18)
)
st.plotly_chart(fig4, use_container_width=True)

# Recruiting rank & class year
st.markdown('---')
left2, right2 = st.columns(2)

with left2:
    st.subheader('Recruiting Rank by Draft Class')
    if 'Rec Rank' in df.columns:
        fig5 = go.Figure()
        for cls_id, cls_name in CLASS_NAMES.items():
            vals = df[df['draft_status'] == cls_id]['Rec Rank'].dropna()
            fig5.add_trace(go.Box(
                y=vals, name=cls_name,
                marker_color=CLASS_COLORS[cls_id], boxmean=True
            ))
        fig5.update_layout(
            yaxis_title='Recruiting Rank (lower = better)',
            height=350, margin=dict(l=10, r=10, t=30, b=10)
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.caption('Rank 999 = unranked. Top recruits (low rank) are far more likely to be drafted.')

with right2:
    st.subheader('Draft Rate by College Class Year')
    if 'yr_num' in df.columns:
        year_data = []
        for yr in [1, 2, 3, 4]:
            sub = df[df['yr_num'] == yr]
            n = len(sub)
            if n > 0:
                drafted = (sub['draft_status'] > 0).sum()
                year_data.append({
                    'Class': {1:'Freshman',2:'Sophomore',3:'Junior',4:'Senior'}[yr],
                    'Total': n,
                    'Drafted': drafted,
                    'Rate (%)': round(100 * drafted / n, 2)
                })
        yr_df = pd.DataFrame(year_data)
        fig6 = go.Figure(go.Bar(
            x=yr_df['Class'], y=yr_df['Rate (%)'],
            marker_color=['#4e79a7','#f28e2b','#e15759','#76b7b2'],
            text=yr_df['Rate (%)'].apply(lambda v: f'{v:.2f}%'),
            textposition='outside'
        ))
        fig6.update_layout(
            yaxis_title='Draft Rate (%)',
            height=350, margin=dict(l=10, r=10, t=30, b=50)
        )
        st.plotly_chart(fig6, use_container_width=True)
        st.caption('Freshmen and sophomores have the highest draft rates — they leave early when talented.')

# Conference & role
st.markdown('---')
left3, right3 = st.columns(2)

with left3:
    st.subheader('Top Conferences by Draft Count')
    if 'conf' in df.columns:
        conf_drafted = (
            df[df['draft_status'] > 0]
            .groupby('conf')
            .size()
            .sort_values(ascending=False)
            .head(12)
        )
        fig7 = go.Figure(go.Bar(
            x=conf_drafted.values,
            y=conf_drafted.index,
            orientation='h',
            marker_color='#4e79a7'
        ))
        fig7.update_layout(
            xaxis_title='Drafted Players',
            yaxis_autorange='reversed',
            height=380, margin=dict(l=80, r=10, t=30, b=10)
        )
        st.plotly_chart(fig7, use_container_width=True)

with right3:
    st.subheader('Draft Rate by Position Role')
    if 'role' in df.columns:
        role_stats = []
        for role, sub in df.groupby('role'):
            n = len(sub)
            drafted = (sub['draft_status'] > 0).sum()
            role_stats.append({'Role': role, 'Total': n, 'Drafted': drafted,
                                'Rate (%)': round(100 * drafted / n, 2)})
        role_df = pd.DataFrame(role_stats).sort_values('Rate (%)', ascending=False)
        fig8 = go.Figure(go.Bar(
            x=role_df['Rate (%)'],
            y=role_df['Role'],
            orientation='h',
            marker_color='#f28e2b',
            text=role_df['Rate (%)'].apply(lambda v: f'{v:.1f}%'),
            textposition='outside'
        ))
        fig8.update_layout(
            xaxis_title='Draft Rate (%)',
            yaxis_autorange='reversed',
            height=380, margin=dict(l=100, r=60, t=30, b=10)
        )
        st.plotly_chart(fig8, use_container_width=True)