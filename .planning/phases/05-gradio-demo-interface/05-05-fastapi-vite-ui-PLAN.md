# Plan 05-05 — FastAPI + Vite/React/Tailwind Demo UI (Revised)

**Phase:** 05 — Demo Interface  
**Status:** PLANNED — not started  
**Revised:** 2026-05-09  
**Skill:** `frontend-design` (`.agents/skills/frontend-design/SKILL.md`) — production-grade, distinctive UI

---

## Goal

`ONNXPredictor` inference mantığını değiştirmeden, FastAPI backend + Vite/React/Tailwind frontend ile production-grade, animasyonlu, model-switching destekli bir demo UI sunmak.

---

## Aesthetic Direction

**Tone:** Refined dark-tech / precision instrument. Bir ML inference aracı gibi hissettirmeli — bilimsel ama soğuk değil. Karanlık arka plan, keskin tipografi, veri görselleştirmesinde hassasiyet.

**Font:** `DM Mono` (monospace, teknik hissi) + `Syne` (display, başlıklar için geometrik). Google Fonts CDN veya `@fontsource`.

**Color palette:**
```
bg:        #080810   (neredeyse siyah, hafif mavi ton)
surface:   #0f0f1e
border:    #1e2035
accent:    #6366f1   (indigo — argmax highlight, CTA)
accent2:   #a78bfa   (violet — secondary)
success:   #34d399
warn:      #fbbf24
danger:    #f87171
text:      #e2e8f0
muted:     #475569
```

**Motion:** `framer-motion` — sayfa yüklenince staggered reveal, classify sırasında scanning animasyonu, sonuçlar slide-in ile gelir, softmax bar'ları width animate eder.

---

## Architecture

```
autolens_ai/
├── app.py                      ← Gradio (HF Spaces, dokunma)
├── api.py                      ← FastAPI backend (YENİ)
└── frontend/                   ← Vite + React + Tailwind (YENİ)
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.ts
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── store.ts            ← Zustand global store
        ├── api.ts              ← typed API client
        ├── types.ts            ← shared types
        └── components/
            ├── Header.tsx
            ├── ModelSelector.tsx
            ├── ImageUpload.tsx
            ├── AnalyzingOverlay.tsx
            ├── ResultCard.tsx
            ├── SoftmaxChart.tsx
            └── ModelDetailPanel.tsx
```

---

## Step 1 — FastAPI Backend (`api.py`)

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/models` | Export artifact'larını listele (metadata dahil) |
| `GET` | `/api/models/active` | ~~Aktif model full metadata~~ — `/api/models` `is_active:true` ile karşılanır, **kaldırıldı** |
| `POST` | `/api/models/active` | `{"model_dir": "..."}` → active_model.json güncelle |
| `POST` | `/api/predict` | `multipart/form-data` image → prediction JSON |
| `GET` | `/api/health` | Liveness check — model yüklü mü, cache kaç model tutuyor |
| `GET` | `/` | `frontend/dist/index.html` |

### `GET /api/models` Response

```json
[
  {
    "id": "efficientnet_b2_current",
    "dir": "artifacts/export/efficientnet_b2_current",
    "model_name": "efficientnet_b2",
    "accuracy": 0.9202,
    "f1_macro": 0.9004,
    "f1_weighted": 0.9205,
    "val_loss": 0.2835,
    "temperature": 1.7799,
    "num_classes": 8,
    "input_size": 224,
    "pretrained": true,
    "source_run_id": "baseline_0_efficientnet_b2_20260509_135313",
    "is_active": true,
    "per_class": { "SUV": {"accuracy": 0.909, "support": 670}, ... },
    "training": {
      "best_epoch": 4,
      "total_epochs_run": 17,
      "learning_rate": 0.001,
      "weight_decay": 0.0001
    }
  },
  ...
]
```

### `POST /api/predict` Response

```json
{
  "predicted_class": "SUV",
  "confidence": 0.923,
  "latency_ms": 18.4,
  "probabilities": {
    "SUV": 0.923, "SEDAN": 0.041, "HATCHBACK": 0.018,
    "VAN": 0.007, "PICK UP": 0.005, "STATION WAGON": 0.003,
    "OPEN WHEEL / F1": 0.002, "MICRO": 0.001
  }
}
```

### Lifespan & Predictor Cache

```python
# Tüm modeller için predictor cache — model_dir → ONNXPredictor
_predictor_cache: dict[str, ONNXPredictor] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: aktif modeli önceden yükle (ilk istek soğuk başlamaz)
    _predictor_cache[_active_model_dir()] = _build_predictor(_active_model_dir())
    yield
    # Shutdown: ONNX session'lar GC'ye bırakılır, temizlik gerekmez

app = FastAPI(lifespan=lifespan)
```

**Cache davranışı:**
- `_get_predictor(model_dir)` → cache'de varsa döndür, yoksa yükle + cache'e ekle
- Model switch (`POST /api/models/active`) → `active_model.json` günceller, cache'i **silmez** — aynı modele geri dönünce yeniden yükleme olmaz
- Uygulama ömrü boyunca tüm yüklenen modeller bellekte kalır (3 model × ~30-100MB = makul)
- Restart gerekirse `uvicorn --reload` lifespan'i yeniden tetikler

### `GET /api/health` Response

```json
{ "status": "ok", "active_model": "efficientnet_b2_current", "cached_models": 2 }
```

### File Validation (`POST /api/predict`)

```python
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

if file.content_type not in ALLOWED_TYPES:
    raise HTTPException(415, "Unsupported file type")
data = await file.read()
if len(data) > MAX_SIZE_BYTES:
    raise HTTPException(413, "File too large (max 10 MB)")
```

### Implementation Notes

- `_discover_models()` mevcut multi-model `app.py` versiyonundan taşınır, `training` ve `per_class` alanları eklenir.
- CORS middleware: dev'de `localhost:5173` origin'e izin ver.
- Image: `UploadFile` → bytes → `PIL.Image.open(BytesIO(...))` → `ONNXPredictor.predict()`.
- Model switch: `active_model.json` güncelle, cache'e dokunma.

---

## Step 2 — Frontend

### Setup

```bash
cd frontend
npm create vite@latest . -- --template react-ts
npm install -D tailwindcss @tailwindcss/vite
npm install framer-motion recharts axios zustand
npm install @fontsource/syne @fontsource/dm-mono
```

### State Management — Zustand (`src/store.ts`)

Tek global store, prop drilling yok:

```ts
interface AppStore {
  // Model state
  models: ModelInfo[]
  activeModel: ModelInfo | null
  fetchModels: () => Promise<void>           // GET /api/models → store populate
  setActiveModel: (model: ModelInfo) => Promise<void>

  // Prediction state
  status: 'idle' | 'analyzing' | 'done' | 'error'
  result: PredictResult | null
  error: string | null
  selectedFile: File | null
  setFile: (file: File) => void
  predict: () => Promise<void>              // selectedFile'ı kullanır

  // UI state
  detailPanelOpen: boolean
  detailTab: 'config' | 'training'
  toggleDetailPanel: () => void
  setDetailTab: (tab: 'config' | 'training') => void
}
```

`setActiveModel` → `POST /api/models/active` → store güncellenir → `ModelDetailPanel` otomatik re-render.  
`predict` → `status: 'analyzing'` set et → `POST /api/predict` → `status: 'done'` + `result` set et. Hata durumunda `status: 'error'` + `error` mesajı set et.  
`AnalyzingOverlay` sadece `status === 'analyzing'`'i subscribe eder, gereksiz re-render yok.

**Initial load — `App.tsx`:**
```tsx
useEffect(() => {
  useAppStore.getState().fetchModels()
}, [])
```
`fetchModels` → `GET /api/models` → `models` + `activeModel` (`is_active: true` olan) store'a yazılır.

### Layout — `App.tsx`

```
┌─────────────────────────────────────────────────────────┐
│  Header: AutoLens AI  ·  model badge  ·  [model switcher]│
├──────────────────┬──────────────────┬────────────────────┤
│                  │                  │                    │
│  ImageUpload     │  ResultCard      │  ModelDetailPanel  │
│  (drag & drop,   │  (pred class,    │  (collapsible,     │
│   preview,       │   confidence,    │   config values,   │
│   classify btn)  │   latency)       │   metrics,         │
│                  │                  │   per-class bars,  │
│                  │  SoftmaxChart    │   training details)│
│                  │  (animated bars) │                    │
└──────────────────┴──────────────────┴────────────────────┘
```

---

### `ImageUpload.tsx`

- Drag & drop zone — dashed border, hover'da glow efekti (`box-shadow: 0 0 0 2px #6366f1`)
- Click to upload fallback
- Yüklenen görsel preview (object-fit: cover, rounded)
- "Classify →" CTA butonu — indigo gradient, hover'da scale(1.02) + brightness
- Görsel yüklenince buton aktif olur

---

### `AnalyzingOverlay.tsx`

Classify tetiklenince görsel üzerine overlay:
- Yarı saydam dark overlay
- Yatay tarama çizgisi animasyonu (top→bottom loop, `framer-motion`)
- "Analyzing…" yazısı pulse ile
- Tamamlanınca fade-out, sonuçlar slide-in

---

### `ResultCard.tsx`

`framer-motion` ile `initial={{ opacity:0, y:20 }}` → `animate={{ opacity:1, y:0 }}`:

- **Predicted class** — büyük monospace font, accent rengi
- **Confidence chip** — renk kodlu:
  - ≥85% → `#34d399` "High confidence"
  - 60–85% → `#fbbf24` "Moderate"
  - <60% → `#f87171` "Low"
- **Latency chip** — `{n} ms · ONNX Runtime`
- **Error state** (`status === 'error'`) — kırmızı border, `error` mesajı, "Try again" butonu (`setFile(null)` + `status: 'idle'`)

---

### `SoftmaxChart.tsx`

Recharts `BarChart` (horizontal), `framer-motion` ile bar width animasyonu:

- Argmax bar: `#6366f1` + label bold
- Diğerleri: `#1e2035`
- Her bar sağında `{prob:.1%}` label
- Sıralama: yüksekten düşüğe (argmax en üstte)
- Arka plan: `#080810` (sayfa rengiyle aynı, çerçeve yok)

---

### `ModelDetailPanel.tsx`

Sağ sütun, varsayılan **kapalı**, başlığa tıklayınca `framer-motion` ile slide-down açılır.

**Kapalı halde** (her zaman görünür):
```
⚙ efficientnet_b2   Acc 92.0%   F1 0.900   ›
```

**Açık halde** iki sekme:

**Config sekmesi:**
- Input size, resize, crop
- Mean / std
- Temperature (calibrated)
- Backend (ONNX Runtime)
- Source run ID

**Training sekmesi:**
- Best epoch / total epochs run
- Learning rate, weight decay
- Val loss
- Per-class accuracy mini bar chart (Recharts veya pure CSS)
- Confusion matrix özeti (en çok karıştırılan 3 çift)

Model değişince panel içeriği `AnimatePresence` ile cross-fade.

---

### `ModelSelector.tsx`

Header'da compact dropdown (custom, Tailwind styled):
- Her option: `{model_name}  ·  acc {acc:.1%}  ·  f1 {f1:.3f}`
- Aktif model badge (indigo dot)
- Seçim → `POST /api/models/active` → predictor cache güncellenir → panel güncellenir

---

### `api.ts` — Typed Client

```ts
export const getModels = (): Promise<ModelInfo[]>
export const setActiveModel = (dir: string): Promise<void>
export const predict = (file: File): Promise<PredictResult>
export const healthCheck = (): Promise<{ status: string; active_model: string; cached_models: number }>
```

---

### `types.ts`

```ts
interface ModelInfo {
  id: string; model_name: string; accuracy: number; f1_macro: number;
  f1_weighted: number; val_loss: number; temperature: number;
  is_active: boolean;
  per_class: Record<string, { accuracy: number; support: number }>;
  training: { best_epoch: number; total_epochs_run: number; learning_rate: number; weight_decay: number };
}

interface PredictResult {
  predicted_class: string; confidence: number; latency_ms: number;
  probabilities: Record<string, number>;
}
```

---

## Step 3 — Production Build & Serve

```bash
cd frontend && npm run build   # → frontend/dist/
```

`api.py`:
```python
app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
```

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```

---

## Step 4 — Makefile Targets

```makefile
frontend-install:
	cd frontend && npm install

frontend-dev:
	cd frontend && npm run dev &
	uv run uvicorn api:app --reload

frontend-build:
	cd frontend && npm run build

demo-web:
	uv run uvicorn api:app --host 0.0.0.0 --port 8000

demo-gradio:
	uv run python app.py
```

---

## Dependencies

**Python — `pyproject.toml`:**
```toml
fastapi>=0.115
uvicorn[standard]>=0.30
python-multipart>=0.0.9
```

**Node — `frontend/package.json`:**
```
vite, react, react-dom, typescript
@tailwindcss/vite, tailwindcss
framer-motion
recharts
axios
zustand
@fontsource/syne, @fontsource/dm-mono
```

---

## Validation Criteria

- [ ] `uv run uvicorn api:app` hatasız ayağa kalkıyor
- [ ] `GET /api/health` `{"status":"ok"}` dönüyor
- [ ] `GET /api/models` 3 model, `per_class` ve `training` alanlarıyla dönüyor, `is_active` doğru set
- [ ] `POST /api/predict` geçerli araç görseli için doğru JSON dönüyor
- [ ] `POST /api/predict` PDF veya 10MB+ dosyada 415/413 hatası dönüyor
- [ ] Frontend `npm run build` hatasız tamamlanıyor
- [ ] Production modda tek port (8000) çalışıyor
- [ ] Sayfa açılınca `fetchModels` çalışır, model listesi ve aktif model yüklenir
- [ ] Model selector'dan farklı model seçince sonraki predict yeni modeli kullanıyor
- [ ] Classify sırasında scanning animasyonu görünüyor, sonuçlar slide-in ile geliyor
- [ ] Softmax bar'ları animate ederek genişliyor
- [ ] ModelDetailPanel açılıp kapanıyor, Config ve Training sekmeleri çalışıyor
- [ ] Predict hatası `status:'error'` + hata mesajı + "Try again" butonu gösteriyor
- [ ] `app.py` (Gradio) bu değişikliklerden etkilenmiyor

---

## What This Plan Does NOT Touch

- `app.py` — Gradio, HF Spaces için olduğu gibi kalır
- `ONNXPredictor` — inference kodu değişmez
- `active_model.json` contract — aynı format
- Mevcut test suite

---

*Plan created: 2026-05-09 · Revised: 2026-05-09*  
*Skill: frontend-design*
