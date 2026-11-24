import json, pathlib, datetime as dt
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from algae.db import get_conn, init_db
from algae.anomaly.detector import Detector

class Reading(BaseModel):
    ts: str; culture_id: str
    temp_c: float; ph: float; do_mgl: float; par: float; od750: float

detector: Detector | None = None

@asynccontextmanager
async def lifespan(app):
    global detector
    init_db(); detector = Detector().fit()
    yield

app = FastAPI(title='Algae Monitor', lifespan=lifespan)

def _recent(cid: str, hours: int) -> pd.DataFrame:
    with get_conn() as c:
        df = pd.read_sql('SELECT * FROM readings WHERE culture_id=? ORDER BY ts DESC LIMIT ?', c, params=(cid, hours))
    df['ts'] = pd.to_datetime(df['ts']); return df.sort_values('ts').reset_index(drop=True)

@app.post('/readings')
def post_readings(rows: list[Reading]):
    with get_conn() as c:
        c.executemany('INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?)',
                      [(r.ts, r.culture_id, r.temp_c, r.ph, r.do_mgl, r.par, r.od750) for r in rows])
        for cid in {r.culture_id for r in rows}:
            df = _recent(cid, 72)
            if len(df) >= 24 and bool(detector.alert_hours(df).iloc[-1]):
                flags = detector.residual_flags(df).iloc[-1].to_dict()
                c.execute('INSERT INTO alerts (ts, culture_id, kind, severity, detail_json) VALUES (?,?,?,?,?)',
                          (df.ts.iloc[-1].isoformat(), cid, 'anomaly', 'warn',
                           json.dumps({'triggers': [k for k, v in flags.items() if v]})))
    return {'inserted': len(rows)}

@app.get('/cultures/{cid}/readings')
def get_readings(cid: str, hours: int = 168):
    return json.loads(_recent(cid, hours).to_json(orient='records', date_format='iso'))

@app.get('/cultures/{cid}/forecast')
def get_forecast(cid: str):
    from algae.forecast.predict import forecast_latest
    df = _recent(cid, 72)
    if len(df) < 48: raise HTTPException(422, 'need at least 48 hourly rows')
    return forecast_latest(df)

@app.post('/cultures/{cid}/images')
async def post_image(cid: str, file: UploadFile = File(...)):
    from algae.vision.infer import analyze
    p = pathlib.Path('data/uploads'); p.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc)
    path = p / f'{cid}_{now:%Y%m%dT%H%M%S}_{pathlib.Path(file.filename).name}'
    path.write_bytes(await file.read())
    res = analyze(str(path))
    with get_conn() as c:
        c.execute('INSERT INTO images (ts, culture_id, path, n_cells, cells_per_ml, contaminants_json) VALUES (?,?,?,?,?,?)',
                  (now.isoformat(), cid, str(path), res['n_cells'], res['cells_per_ml'], json.dumps(res['contaminants'])))
        if res['contaminants']:
            c.execute('INSERT INTO alerts (ts, culture_id, kind, severity, detail_json) VALUES (?,?,?,?,?)',
                      (now.isoformat(), cid, 'contamination', 'critical', json.dumps(res)))
    return res

@app.get('/alerts')
def list_alerts(open_only: bool = True):
    q = 'SELECT * FROM alerts' + (' WHERE acknowledged=0' if open_only else '') + ' ORDER BY ts DESC LIMIT 100'
    with get_conn() as c:
        return [dict(r) for r in c.execute(q)]

@app.post('/alerts/{aid}/explain')
def explain_alert(aid: int):
    from algae.explain.explain import explain
    with get_conn() as c:
        a = c.execute('SELECT * FROM alerts WHERE id=?', (aid,)).fetchone()
        if not a: raise HTTPException(404)
        e = explain(dict(a), _recent(a['culture_id'], 48))
        c.execute('UPDATE alerts SET explanation=? WHERE id=?', (e.model_dump_json(), aid))
    return e
