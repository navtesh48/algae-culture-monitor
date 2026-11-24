from ultralytics import YOLO

# Neubauer hemocytometer large square = 1 mm x 1 mm x 0.1 mm = 1e-4 mL.
# If each image covers exactly one large square, cells/mL = count / 1e-4 * dilution.
FIELD_VOLUME_ML = 1e-4
DILUTION = 1.0
_model = None

def analyze(path: str, conf: float = 0.25) -> dict:
    global _model
    if _model is None:
        _model = YOLO('data/models/yolo_best.pt')
    r = _model(path, conf=conf, verbose=False)[0]
    cells, contam = 0, []
    for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        name = r.names[int(c)]
        if name == 'algae_cell':
            cells += 1
        elif name != 'debris' and p >= 0.4:
            contam.append({'cls': name, 'conf': round(p, 3)})
    return {'n_cells': cells, 'cells_per_ml': cells / FIELD_VOLUME_ML * DILUTION,
            'contaminants': contam}
