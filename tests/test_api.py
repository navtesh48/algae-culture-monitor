from fastapi.testclient import TestClient
from algae.api.main import app

def test_post_and_read():
    with TestClient(app) as c:
        r = c.post('/readings', json=[{'ts': '2030-01-01T00:00:00Z', 'culture_id': 'test', 'temp_c': 28, 'ph': 7.6,
                                       'do_mgl': 6, 'par': 0, 'od750': 1.0}])
        assert r.status_code == 200
        assert len(c.get('/cultures/test/readings').json()) >= 1
