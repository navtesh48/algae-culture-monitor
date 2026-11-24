import requests, pandas as pd, plotly.graph_objects as go, streamlit as st
API = 'http://localhost:8000'
st.set_page_config(page_title='Algae Monitor', layout='wide')
cid = st.sidebar.selectbox('Culture', [f'pbr{k}' for k in range(20)])
hours = st.sidebar.slider('History (hours)', 48, 720, 168)

df = pd.DataFrame(requests.get(f'{API}/cultures/{cid}/readings', params={'hours': hours}).json())
df['ts'] = pd.to_datetime(df['ts'])
fc = requests.get(f'{API}/cultures/{cid}/forecast').json()

fig = go.Figure(go.Scatter(x=df.ts, y=df.od750, name='OD750'))
t = pd.to_datetime(fc['target_ts'])
fig.add_trace(go.Scatter(x=[t, t], y=[fc['p10'], fc['p90']], mode='lines', name='80% interval', line=dict(width=6)))
fig.add_trace(go.Scatter(x=[t], y=[fc['p50']], mode='markers', name='24h forecast'))
st.plotly_chart(fig, use_container_width=True)

for name in ('temp_c', 'ph', 'do_mgl'):
    st.line_chart(df.set_index('ts')[name])

st.subheader('Open alerts')
for a in requests.get(f'{API}/alerts').json():
    if a['culture_id'] == cid:
        with st.expander(f"{a['ts']}  {a['kind']}  {a['severity']}"):
            if st.button('Explain', key=a['id']):
                st.json(requests.post(f"{API}/alerts/{a['id']}/explain").json())

up = st.file_uploader('Microscope image', type=['jpg', 'png'])
if up:
    st.json(requests.post(f'{API}/cultures/{cid}/images', files={'file': up.getvalue()}).json())
