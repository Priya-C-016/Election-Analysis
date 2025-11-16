# app.py
# Streamlit dashboard: Statistical Analysis of Electoral Patterns in Indian Democracy (1977-2015)
# Advanced version (tabs) — Mobile-friendly
# Data sources (remote CSVs):
#  - Vidhan Sabha: https://samatrix-data.s3.ap-south-1.amazonaws.com/Statistics-Project/ind-vidhan-sabha.csv
#  - Lok Sabha:   https://samatrix-data.s3.ap-south-1.amazonaws.com/Statistics-Project/ind-lok-sabha.csv
#
# Instructions:
# 1) Run locally: `streamlit run app.py`
# 2) Deploy: push to a GitHub repo and deploy to Streamlit Cloud (link your repo, it will auto-run)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from functools import lru_cache

st.set_page_config(page_title="Electoral Patterns (1977-2015)", layout="wide")

# ---------------------- Cached data loading ----------------------
@st.cache_data(show_spinner=False)
def load_data():
    vidhan_url = 'https://samatrix-data.s3.ap-south-1.amazonaws.com/Statistics-Project/ind-vidhan-sabha.csv'
    lok_url = 'https://samatrix-data.s3.ap-south-1.amazonaws.com/Statistics-Project/ind-lok-sabha.csv'
    df_v = pd.read_csv(vidhan_url)
    df_l = pd.read_csv(lok_url)

    # normalize column names (make consistent)
    df_v = df_v.rename(columns={
        'ac_no':'const_no', 'ac_name':'const_name', 'ac_type':'const_type'
    })
    df_l = df_l.rename(columns={
        'pc_no':'const_no', 'pc_name':'const_name', 'pc_type':'const_type'
    })

    # tag dataset
    df_v['election_type'] = 'Vidhan Sabha'
    df_l['election_type'] = 'Lok Sabha'

    # unify columns order
    common_cols = ['st_name','year','const_no','const_name','const_type','cand_name',
                   'cand_sex','partyname','partyabbre','totvotpoll','electors','election_type']
    df_v = df_v[common_cols]
    df_l = df_l[common_cols]

    # clean simple issues
    df_v['partyname'] = df_v['partyname'].fillna('Independent')
    df_l['partyname'] = df_l['partyname'].fillna('Independent')

    # ensure numeric
    for col in ['totvotpoll','electors','year']:
        for df in (df_v, df_l):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # append
    df = pd.concat([df_l, df_v], axis=0, ignore_index=True)

    # add columns
    df['turnout_pct'] = (df['totvotpoll'] / df['electors']) * 100
    # normalize candidate sex values
    df['cand_sex'] = df['cand_sex'].astype(str).str.strip().replace({'nan':'Unknown', '': 'Unknown'})

    return df, df_v, df_l

with st.spinner('Loading data...'):
    df, df_vidhan, df_lok = load_data()

# ---------------------- Helper functions ----------------------
@st.cache_data
def compute_overview(df):
    total_rows = len(df)
    years = sorted(df['year'].dropna().unique())
    years_range = (int(min(years)), int(max(years))) if years else (None, None)
    total_candidates = df['cand_name'].nunique()
    total_parties = df['partyname'].nunique()
    male = (df['cand_sex']=='M').sum() if 'M' in df['cand_sex'].unique() else (df['cand_sex'].str.upper().str.startswith('M')).sum()
    female = (df['cand_sex']=='F').sum() if 'F' in df['cand_sex'].unique() else (df['cand_sex'].str.upper().str.startswith('F')).sum()
    return {
        'rows': total_rows,
        'years_range': years_range,
        'candidates': total_candidates,
        'parties': total_parties,
        'male': int(male),
        'female': int(female)
    }

@st.cache_data
def compute_winners(df):
    # A simple way: per election_type, year, state, constituency -> candidate with max votes
    df_valid = df.dropna(subset=['totvotpoll'])
    winners = df_valid.loc[df_valid.groupby(['election_type','year','st_name','const_no'])['totvotpoll'].idxmax()].copy()
    winners['won'] = 1
    return winners

@st.cache_data
def yearly_turnout(df, election_type=None):
    d = df.copy()
    if election_type:
        d = d[d['election_type']==election_type]
    # aggregate by year
    agg = d.groupby('year').apply(lambda x: (x['totvotpoll'].sum() / x['electors'].sum())*100).reset_index()
    agg.columns = ['year','turnout_pct']
    agg = agg.sort_values('year')
    return agg

# winners precomputed
winners = compute_winners(df)
overview = compute_overview(df)

# ---------------------- Sidebar controls ----------------------
st.sidebar.title('Controls')
selected_election = st.sidebar.radio('Election type', options=['Both','Lok Sabha','Vidhan Sabha'], index=0)
selected_state = st.sidebar.selectbox('State (All)', options=['All'] + sorted(df['st_name'].dropna().unique().tolist()))
selected_year = st.sidebar.selectbox('Year (All)', options=['All'] + sorted(df['year'].dropna().unique().astype(int).tolist()))
selected_party = st.sidebar.selectbox('Party (All)', options=['All'] + sorted(df['partyname'].dropna().unique().tolist()))

# quick filter apply
def apply_filters(df):
    d = df.copy()
    if selected_election != 'Both':
        d = d[d['election_type']==selected_election]
    if selected_state != 'All':
        d = d[d['st_name']==selected_state]
    if selected_year != 'All':
        d = d[d['year']==int(selected_year)]
    if selected_party != 'All':
        d = d[d['partyname']==selected_party]
    return d

filtered_df = apply_filters(df)
filtered_winners = apply_filters(winners)

# ---------------------- Top-level layout ----------------------
st.title('Statistical Analysis of Electoral Patterns in Indian Democracy (1977–2015)')
st.markdown('**Interactive dashboard** — use the sidebar to filter elections, state, year and party. Mobile friendly layout.')

# Tabs for advanced dashboard
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Trends","Party Analysis","Gender","State Insights"])

# ---------------------- OVERVIEW TAB ----------------------
with tab1:
    st.header('Overview')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Rows (records)', f"{overview['rows']:,}")
    yrs = overview['years_range']
    col2.metric('Years covered', f"{yrs[0]} - {yrs[1]}")
    col3.metric('Unique candidates', f"{overview['candidates']:,}")
    col4.metric('Unique parties', f"{overview['parties']:,}")

    st.subheader('Gender distribution (all data)')
    gcol1, gcol2 = st.columns([1,2])
    sex_counts = df['cand_sex'].value_counts().reset_index()
    sex_counts.columns = ['sex','count']
    fig_pie = px.pie(sex_counts, names='sex', values='count', title='Candidate sex distribution')
    gcol1.plotly_chart(fig_pie, use_container_width=True)

    st.subheader('Top insights from current filters')
    # quick numbers from filtered df
    fcol1, fcol2, fcol3 = st.columns(3)
    fcol1.metric('Records (filtered)', f"{len(filtered_df):,}")
    avg_turnout = filtered_df['turnout_pct'].mean()
    fcol2.metric('Avg turnout (%)', f"{avg_turnout:.2f}" if not np.isnan(avg_turnout) else 'N/A')
    top_party = filtered_df['partyname'].value_counts().idxmax() if len(filtered_df)>0 else 'N/A'
    fcol3.metric('Top party (filtered)', top_party)

    with st.expander('View sample data (first 10 rows)'):
        st.dataframe(filtered_df.head(10))

# ---------------------- TRENDS TAB ----------------------
with tab2:
    st.header('Trends (1977–2015)')
    st.markdown('Year-wise turnout and overall time trends.')

    colA, colB = st.columns([2,1])
    # turnout for both
    agg_both = yearly_turnout(df)
    fig = px.line(agg_both, x='year', y='turnout_pct', markers=True, title='Year-wise overall turnout (%) — All Elections')
    colA.plotly_chart(fig, use_container_width=True)

    # separate by election type
    agg_lok = yearly_turnout(df, election_type='Lok Sabha')
    agg_vid = yearly_turnout(df, election_type='Vidhan Sabha')
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=agg_lok['year'], y=agg_lok['turnout_pct'], mode='lines+markers', name='Lok Sabha'))
    fig2.add_trace(go.Scatter(x=agg_vid['year'], y=agg_vid['turnout_pct'], mode='lines+markers', name='Vidhan Sabha'))
    fig2.update_layout(title='Turnout (%) by Election Type', xaxis_title='Year', yaxis_title='Turnout (%)')
    colA.plotly_chart(fig2, use_container_width=True)

    # show top years by turnout
    with colB:
        st.subheader('Top turnout years')
        top_years = agg_both.sort_values('turnout_pct', ascending=False).head(8)
        st.table(top_years.round(2))

    st.subheader('Turnout distribution across states (box plot)')
    # boxplot per state for filtered election type
    sel_type = st.selectbox('Choose election type for state distribution', options=['Both','Lok Sabha','Vidhan Sabha'], key='box_elec')
    dplot = df.copy()
    if sel_type != 'Both':
        dplot = dplot[dplot['election_type']==sel_type]
    # select states with sufficient data
    state_counts = dplot['st_name'].value_counts()
    good_states = state_counts[state_counts>30].index.tolist()
    dplot = dplot[dplot['st_name'].isin(good_states)]
    fig_box = px.box(dplot, x='st_name', y='turnout_pct', points='outliers', title='Turnout % distribution by state (states with >30 records)')
    fig_box.update_layout(xaxis={'categoryorder':'total descending'}, height=500)
    st.plotly_chart(fig_box, use_container_width=True)

# ---------------------- PARTY ANALYSIS TAB ----------------------
with tab3:
    st.header('Party Analysis')
    st.markdown('Examine party-level performance, dominance and vote shares.')

    party = st.selectbox('Select party (for detailed view)', options=['All'] + sorted(df['partyname'].unique().tolist()), key='party_select')
    # total votes by party over years
    df_party = df.copy()
    votes_by_party_year = df_party.groupby(['year','partyname'])['totvotpoll'].sum().reset_index()

    if party!='All':
        data_party = votes_by_party_year[votes_by_party_year['partyname']==party]
        figp = px.line(data_party, x='year', y='totvotpoll', title=f'Total votes for {party} by year', markers=True)
        st.plotly_chart(figp, use_container_width=True)

        st.write('Top performing states for this party (by total votes)')
        st.table(df_party[df_party['partyname']==party].groupby('st_name')['totvotpoll'].sum().sort_values(ascending=False).head(10))
    else:
        top_parties = df_party.groupby('partyname')['totvotpoll'].sum().sort_values(ascending=False).head(15)
        fig_bar = px.bar(top_parties.reset_index(), x='partyname', y='totvotpoll', title='Top 15 parties by total votes (1977-2015)')
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader('Party dominance heatmap (by year & state)')
    st.markdown('This shows which party got the highest total votes in a state in a given year (simplified view).')
    # simplified dominance: party with max votes in state-year
    grp = df.groupby(['year','st_name','partyname'])['totvotpoll'].sum().reset_index()
    idx = grp.groupby(['year','st_name'])['totvotpoll'].idxmax()
    dom = grp.loc[idx][['year','st_name','partyname']]
    # pivot for a small sample (limit states to top 12 by records)
    states_small = df['st_name'].value_counts().head(12).index.tolist()
    dom_small = dom[dom['st_name'].isin(states_small)]
    pivot_dom = dom_small.pivot(index='st_name', columns='year', values='partyname')
    st.dataframe(pivot_dom.fillna('-'))

# ---------------------- GENDER TAB ----------------------
with tab4:
    st.header('Gender Analysis')
    st.markdown('How male and female participation evolved over years and across election types.')

    gender_time = df.groupby(['year','cand_sex'])['cand_name'].count().reset_index()
    gender_time = gender_time[gender_time['cand_sex'].notna()]
    fig_gender = px.line(gender_time, x='year', y='cand_name', color='cand_sex', title='Number of candidates by sex over years', markers=True)
    st.plotly_chart(fig_gender, use_container_width=True)

    st.subheader('Female candidate share (%) by year')
    year_tot = df.groupby('year')['cand_name'].count().reset_index(name='total')
    year_fem = df[df['cand_sex'].str.upper().str.startswith('F')].groupby('year')['cand_name'].count().reset_index(name='female')
    comb = pd.merge(year_tot, year_fem, on='year', how='left').fillna(0)
    comb['female_share'] = (comb['female'] / comb['total']) * 100
    fig_fs = px.line(comb, x='year', y='female_share', title='Female candidate share (%) by year', markers=True)
    st.plotly_chart(fig_fs, use_container_width=True)

    st.write('States with highest female participation (overall)')
    fem_states = df[df['cand_sex'].str.upper().str.startswith('F')].groupby('st_name')['cand_name'].count().sort_values(ascending=False).head(10)
    st.table(fem_states)

# ---------------------- STATE INSIGHTS TAB ----------------------
with tab5:
    st.header('State Insights')
    st.markdown('Choose a state to see constituency-level patterns, winners and turnout.')
    st_state = st.selectbox('Select state for detailed view', options=sorted(df['st_name'].dropna().unique().tolist()), index=0, key='state_insights')

    st.subheader(f'State summary: {st_state}')
    sd = df[df['st_name']==st_state]
    s_winners = winners[winners['st_name']==st_state]

    c1, c2, c3 = st.columns(3)
    c1.metric('Total records', f"{len(sd):,}")
    c2.metric('Total constituencies', sd['const_no'].nunique())
    avg_turn = sd['turnout_pct'].mean()
    c3.metric('Avg turnout (%)', f"{avg_turn:.2f}" if not np.isnan(avg_turn) else 'N/A')

    st.subheader('Top parties by total votes in this state')
    st.table(sd.groupby('partyname')['totvotpoll'].sum().sort_values(ascending=False).head(10))

    st.subheader('Winning parties timeline (by year)')
    win_seq = s_winners.groupby(['year','partyname'])['won'].count().reset_index()
    fig_win = px.bar(win_seq, x='year', y='won', color='partyname', title=f'Winning party counts by year — {st_state}')
    st.plotly_chart(fig_win, use_container_width=True)

    with st.expander('Show constituency-wise winners (latest year)'):
        latest_year = int(sd['year'].max())
        latest_wins = s_winners[s_winners['year']==latest_year][['year','const_no','const_name','cand_name','partyname','totvotpoll']]
        st.write(f'Winners in {latest_year} ({len(latest_wins)} constituencies)')
        st.dataframe(latest_wins.sort_values('const_no'))

# ---------------------- Footer / Notes ----------------------
with st.expander('Notes and methodology'):
    st.markdown('''
    - Winners are determined by the highest `totvotpoll` within each election_type-year-state-constituency (a simplification).
    - Turnout is computed as total votes recorded (`totvotpoll`) divided by `electors` in that row; aggregated turnout sums votes and electors across groups.
    - Data may contain imperfect `cand_sex` labels; we used string-start checks for M/F to be inclusive.
    - For deployment: upload this file to a GitHub repo and connect the repo to Streamlit Cloud (https://share.streamlit.io) — Streamlit Cloud will install dependencies from requirements.txt and run `streamlit run app.py`.
    ''')

st.caption('Built with Streamlit — designed to work on mobile browsers. Resize the browser or open on phone to check responsiveness.')
