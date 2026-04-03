# Anomaly Detection Dashboard

An AI-powered smart home monitoring dashboard that detects anomalies in elderly residents' daily behavior using an **LSTM Autoencoder** model trained on the **REFIT** dataset.

## What It Does

- Processes minute-level energy/activity data from smart home sensors
- Runs an **LSTM Autoencoder** to learn each household’s “normal” daily pattern
- Flags days where behavior deviates significantly from the norm
- Displays results in an interactive dashboard with alerts, statistics, and comparisons

---

## Tech Stack

| Layer | Technology |
|------|------------|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Backend | FastAPI (Python) |
| ML Model | PyTorch (LSTM Autoencoder) |
| Preprocessing | scikit-learn (`StandardScaler`) |
| Charts | Recharts |

---

## Project Structure

```text
Dashboard/
├── backend/
│   ├── main.py                  # FastAPI entry point
│   ├── requirements.txt
│   ├── data/                    # Place CSV files here (not included)
│   ├── model/                   # Place model files here (not included)
│   │   ├── lstm_autoencoder_model.pth
│   │   └── scaler (1).pkl
│   ├── routers/
│   │   ├── anomalies.py
│   │   ├── days.py
│   │   ├── houses.py
│   │   └── stats.py
│   ├── schemas/
│   │   └── responses.py
│   └── services/
│       ├── data_service.py
│       ├── model_service.py
│       └── scoring_service.py
├── frontend/
│   └── src/
│       ├── pages/               # Overview, Alerts, DayDetail, Statistics, Comparison
│       ├── components/          # Sidebar, Charts, KPI Cards
│       ├── services/api.ts      # Axios API calls
│       └── types/index.ts
├── docker-compose.yml
└── README.md
```

---

## Setup & Installation

### Prerequisites

- Python **3.10+**
- Node.js **18+**
- Git

### 1) Clone the repository

```bash
git clone https://github.com/AhmedBenRahma/PCD.git
cd PCD
```

### 2) Add the data files

Download the **REFIT Smart Home Dataset**:

- https://pureportal.strath.ac.uk/en/datasets/refit-electrical-load-measurements-cleaned

Place the cleaned CSV files inside:

```text
backend/data/
├── CLEAN_House1.csv
├── CLEAN_House2.csv
├── CLEAN_House3.csv
├── CLEAN_House4.csv
└── CLEAN_House5.csv
```

### 3) Add the model files

Place the trained model files inside:

```text
backend/model/
├── lstm_autoencoder_model.pth
└── scaler (1).pkl
```

### 4) Run the Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

- Backend: http://127.0.0.1:8000  
- API Docs: http://127.0.0.1:8000/docs

**Note:** The first startup may take several minutes because the model scores all CSV files. Subsequent startups may be faster if caching is enabled.

### 5) Run the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

Frontend: http://localhost:5173

### 6) (Optional) Run with Docker

If you have Docker installed:

```bash
docker-compose up
```

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| Overview | Summary of anomaly scores across all houses |
| Alerts | Filterable table of flagged anomaly days |
| Day Detail | Signal vs reconstruction chart for a specific day |
| Statistics | Trends and distributions over time |
| Comparison | Compare behavior patterns across houses |

---

## Notes

- The model was trained on **scikit-learn 1.6.1** — a version warning on startup is normal and can be ignored.
- CSV files and model files are excluded from this repository due to size.
- The scaler file uses **joblib** format and must be loaded using `joblib.load()` (not `pickle.load()`).

---

## Author

**Ahmed Ben Rahma** — PCD Project  
Ministère de l'Enseignement Supérieur et de la Recherche Scientifique
