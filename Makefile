seed:   ; python scripts/seed_db.py
images: ; python -m algae.sim.synth_microscope
train:  ; python src/algae/vision/train.py && python -m algae.forecast.train
api:    ; uvicorn algae.api.main:app --reload
ui:     ; streamlit run src/algae/dashboard/app.py
test:   ; pytest -q
