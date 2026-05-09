# Plan 05-05 — FastAPI + Vite/React/Tailwind Demo UI

**Phase:** 05 — Demo Interface  
**Status:** PLANNED — not started  
**Rationale:** Gradio'nun CSS/layout kısıtları production-grade UI için yetersiz. Mevcut `app.py` (Gradio) HF Spaces için korunuyor. Bu plan paralel olarak tam kontrollü bir web UI ekler.

---

## Goal

`ONNXPredictor` inference mantığını değiştirmeden, FastAPI backend + Vite/React/Tailwind frontend ile modern bir demo UI sunmak.

---

## Architecture

```
autolens_ai/
├── app.py                  ← Gradio (HF Spaces için korunuyor, dokunma)
├── api.py                  ← FastAPI backend (YENİ)
└── frontend/               ← Vite + React + Tailwind (YENİ)
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    └── src/
        ├── App.tsx
        ├── components/
        │   ├── ImageUpload.tsx
        │   ├── ResultCard.tsx
        │   ├── SoftmaxChart.tsx
        │   └── ModelSelector.tsx
        └── api.ts
```

Production build: `npm run build` → `frontend/dist/` → FastAPI `StaticFiles` ile serve eder. Tek process, tek port.

---

## Step 1 — FastAPI Backend (`api.py`)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/models` | Mevcut export artifact'larını listele |
| `GET` | `/api/models/active` | Aktif modelin metadata'sını döndür |
| `POST` | `/api/models/active` | `{"model_dir": "..."}` — active_model.json güncelle |
| `POST` | `/api/predict` | `multipart/form-data` image → prediction JSON |
| `GET` | `/` | `frontend/dist/index.html` serve et |

### `/api/predict` Response Shape

```json
{
  "predicted_class": "SUV",
  "confidence": 0.923,
  "latency_ms": 18.4,
  "probabilities": {
    "SUV": 0.923,
    "SEDAN": 0.041,
    "HATCHBACK": 0.018,
    ...
  }
}
```

### Implementation Notes

- `_discover_models()` ve `_get_predictor()` mantığı mevcut `app.py`'deki multi-model versiyondan taşınır.
- CORS middleware ekle (dev'de Vite `localhost:5173` → FastAPI `localhost:8000`).
- Image upload: `UploadFile` → `PIL.Image` → `ONNXPredictor.predict()`.
- Model switch: `active_model.json`'u güncelle, predictor cache'i temizle.

---

## Step 2 — Frontend (`frontend/`)

### Setup

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install -D tailwindcss @tailwindcss/vite
npm install recharts axios
```

### Components

**`ModelSelector.tsx`**
- `GET /api/models` ile model listesini çek
- Dropdown — her item'da model adı, accuracy, F1 macro
- Seçim → `POST /api/models/active`

**`ImageUpload.tsx`**
- Drag & drop + click to upload
- Preview thumbnail
- Upload sonrası otomatik `POST /api/predict` tetikle

**`ResultCard.tsx`**
- Predicted class (büyük, bold)
- Confidence chip (renk kodlu: yeşil/turuncu/kırmızı)
- Latency chip

**`SoftmaxChart.tsx`**
- Recharts `BarChart` (horizontal)
- 8 class, argmax highlight (#6366f1), diğerleri gri
- Yüzde label'ları

**`App.tsx`** — 3 sütun layout:
```
[Model Selector + Upload] | [Result Card + Chart] | [Model Info Panel]
```
Model info panel: architecture, temperature, test metrics, per-class accuracy mini bar'ları — sağda sabit, model değişince güncellenir.

### Vite Config (dev proxy)

```ts
// vite.config.ts
server: {
  proxy: {
    '/api': 'http://localhost:8000'
  }
}
```

---

## Step 3 — Production Build & Serve

```bash
cd frontend && npm run build   # → frontend/dist/
```

`api.py`'de:
```python
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

Tek komutla çalışır:
```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Step 4 — Makefile Targets

```makefile
frontend-install:
    cd frontend && npm install

frontend-dev:
    cd frontend && npm run dev

frontend-build:
    cd frontend && npm run build

demo-web:
    uv run uvicorn api:app --host 0.0.0.0 --port 8000 --reload

demo-gradio:          # HF Spaces / mevcut Gradio
    uv run python app.py
```

---

## Dependencies

**Python (pyproject.toml'a ekle):**
```
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
```

**Node (frontend/package.json):**
```
vite, react, react-dom, typescript
@tailwindcss/vite, tailwindcss
recharts
axios
```

---

## Validation Criteria

- [ ] `uv run uvicorn api:app` ayağa kalkıyor
- [ ] `GET /api/models` 3+ model döndürüyor
- [ ] `POST /api/predict` geçerli bir araç görseli için doğru JSON döndürüyor
- [ ] Frontend build hatasız tamamlanıyor (`npm run build`)
- [ ] Production modda tek port (8000) üzerinden çalışıyor
- [ ] Model selector'dan farklı model seçince `active_model.json` güncelleniyor ve sonraki predict yeni modeli kullanıyor
- [ ] `app.py` (Gradio) bu değişikliklerden etkilenmiyor, bağımsız çalışıyor

---

## What This Plan Does NOT Touch

- `app.py` — Gradio versiyonu olduğu gibi kalır, HF Spaces deploy için
- `ONNXPredictor` — inference kodu değişmez
- `active_model.json` contract — aynı format
- Mevcut test suite

---

*Plan created: 2026-05-09*  
*Author: kiro*
