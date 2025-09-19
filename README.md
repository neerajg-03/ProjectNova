# ProjectNova - Deployment

Run locally (dev):

```bash
pip install -r requirements.txt
python app.py
```

Serve with Gunicorn (prod-like):

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 2 --threads 4 --timeout 120
```

Deploy on Render/Railway/Heroku (Procfile):

```text
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120
```

Docker build and run:

```bash
docker build -t projectnova .
docker run -p 5000:5000 projectnova
```

Notes:
- App writes/reads dataset `nova_dataset.csv` and model `nova_model.pkl` in the container filesystem.
- For persistent data across restarts, mount a volume to `/app` or specific files.
- Set `PORT` env var if your platform requires it (Procfile handles it automatically).

