"""Writes 20 simulated cultures into SQLite (and their ground-truth event labels to CSV)."""
import sys; sys.path.insert(0, 'src')
from algae.db import init_db, get_conn
from algae.sim.simulator import simulate

init_db()
with get_conn() as c:
    for k in range(20):
        df = simulate(days=30, culture_id=f'pbr{k}', seed=k)
        df['ts'] = df['ts'].dt.strftime('%Y-%m-%dT%H:%M:%SZ')
        df[['ts','culture_id','temp_c','ph','do_mgl','par','od750']].to_sql(
            'readings', c, if_exists='append', index=False)
        df[['ts','culture_id','event']].to_csv(f'data/events_pbr{k}.csv', index=False)
print('seeded')
