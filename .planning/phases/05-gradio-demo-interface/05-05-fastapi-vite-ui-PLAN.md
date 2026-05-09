# Plan 05-05 — FastAPI + Vite/React/Tailwind Demo UI

**Phase:** 05 — Demo Interface  
**Status:** PLANNED — not started  
**Skill:** `frontend-design` (`.agents/skills/frontend-design/SKILL.md`)

---

## Goal

`ONNXPredictor` inference mantığını değiştirmeden, FastAPI backend + Vite/React/Tailwind frontend ile production-grade, animasyonlu, model-switching destekli bir demo UI sunmak. `app.py` (Gradio) HF Spaces için dokunulmadan korunur.

---

## Aesthetic Direction

**Tone:** Refined dark-tech / precision instrument. Bilimsel ama soğuk değil.  
**Font:** `DM Mono` (monospace, teknik) + `Syne` (display, başlıklar).  
**Colors:** `#080810` bg, `#0f0f1e` surface, `#6366f1` accent (indigo), `#a78bfa` secondary, `#34d399` success, `#fbbf24` warn, `#f87171` danger.  
**Motion:** `framer-motion` — staggered page reveal, scanning overlay, slide-in results. Recharts kendi animasyon sistemiyle, framer-motion sadece container mount/unmount için.

---

## Architecture

```
autolens_ai/
├── app.py          ← Gradio (HF Spaces, dokunma)
├── api.py          ← FastAPI backend
└── frontend/       ← Vite + React + Tailwind
    └── src/
        ├── App.tsx
        ├── store.ts          ← Zustand
        ├── api.ts            ← typed API client
        ├── types.ts
        └── components/
            ├── Header.tsx
            ├── ModelSelector.tsx
            ├── ImageUpload.tsx
            ├── AnalyzingOverlay.tsx
            ├── ResultCard.tsx
            ├── SoftmaxChart.tsx
            └── ModelDetailPanel.tsx
```

Production: `npm run build` → `frontend/dist/` → FastAPI StaticFiles ile serve. Tek process, tek port (8000).

---

## Step 1 — FastAPI Backend (`api.py`)

### Endpoints

| Method | Path | Açıklama |
|--------|------|----------|
| `GET` | `/api/health` | Liveness check — aktif model adı, cache'deki model sayısı |
| `GET` | `/api/models` | Tüm export artifact'ları — metadata, metrics, `is_active` flag |
| `POST` | `/api/models/active` | `id` ile aktif modeli değiştir → `active_model.json` güncelle |
| `POST` | `/api/predict` | Görsel yükle → prediction JSON dön |
| `GET` | `/` | `frontend/dist/index.html` |

### Önemli Kararlar

- **`GET /api/models/active` yok** — `is_active: true` olan `/api/models` response'undan alınır.
- **`POST /api/models/active` body'si `id` alır**, `dir` değil — filesystem path frontend'e expose edilmez, backend `id → dir` mapping'i kendisi yapar.
- **File validation** — sadece `image/jpeg`, `image/png`, `image/webp` kabul edilir, max 10MB. Aksi halde 415/413 döner.
- **Lifespan** — startup'ta aktif model preload edilir, ilk istek soğuk başlamaz.
- **Predictor cache** — `model_id → ONNXPredictor` dict. Model switch cache'e dokunmaz; aynı modele geri dönünce yeniden yükleme olmaz. 3 model bellekte tutmak makul (~30-100MB her biri).

### `/api/models` Response Alanları

Her model için: `id`, `model_name`, `accuracy`, `f1_macro`, `f1_weighted`, `val_loss`, `temperature`, `num_classes`, `input_size`, `pretrained`, `source_run_id`, `is_active`, `per_class` (her class için accuracy + support), `training` (best_epoch, total_epochs_run, learning_rate, weight_decay).

---

## Step 2 — Frontend

### State — Zustand (`store.ts`)

Tek global store, prop drilling yok. Alanlar:

- **Model:** `models[]`, `activeModel`, `fetchModels()`, `setActiveModel()` — API call + state update + result reset
- **Prediction:** `status` (idle/analyzing/done/error), `result`, `error`, `selectedFile`, `setFile()` — file set edilince status/result reset, `predict()` — selectedFile'ı kullanır, null guard var
- **UI:** `detailPanelOpen`, `detailTab` (config/training), `toggleDetailPanel()`, `setDetailTab()`

### Layout

3 sütun: **Sol** (model selector + image upload) | **Orta** (result card + softmax chart) | **Sağ** (model detail panel).

### Bileşenler

**`Header`** — Logo, aktif model badge, model selector dropdown. Her model option'ında: model adı, accuracy, F1 macro. Seçim → `setActiveModel()`.

**`ImageUpload`** — Drag & drop + click. Preview thumbnail. Classify butonu sadece `selectedFile` varken aktif. Yeni dosya seçilince önceki result temizlenir.

**`AnalyzingOverlay`** — `status === 'analyzing'` iken görsel üzerine: yarı saydam overlay + yatay tarama çizgisi animasyonu (loop) + "Analyzing…" pulse. Tamamlanınca fade-out.

**`ResultCard`** — `status === 'done'`: predicted class (büyük monospace), confidence chip (renk kodlu), latency chip. `status === 'error'`: kırmızı border, hata mesajı, "Try again" butonu (state reset). `framer-motion` slide-in.

**`SoftmaxChart`** — Recharts horizontal bar chart. Argmax indigo, diğerleri gri. Yüksekten düşüğe sıralı. Recharts kendi animasyonuyla açılır. Çerçeve/arka plan yok.

**`ModelDetailPanel`** — Sağ sütun, varsayılan kapalı. Kapalıyken: model adı + acc + F1 + açma oku. Açıkken iki sekme:
- *Config:* input size, resize/crop, mean/std, temperature, backend, run ID
- *Training:* best epoch, total epochs, LR, weight decay, val loss, per-class accuracy bar'ları

Model değişince `AnimatePresence` cross-fade.

**`App.tsx`** — Mount'ta `fetchModels()` çağrılır, modeller ve aktif model yüklenir.

### Dev Ortamı

Vite dev server ve uvicorn ayrı terminallerde çalıştırılır (`concurrently` veya iki terminal). Vite `/api/*` isteklerini `localhost:8000`'e proxy'ler.

---

## Step 3 — Makefile Targets

- `frontend-install` — npm bağımlılıklarını yükle
- `frontend-build` — production build
- `demo-web` — production modda tek port (8000)
- `demo-gradio` — Gradio versiyonu (HF Spaces)

---

## Dependencies

**Python:** `fastapi`, `uvicorn[standard]`, `python-multipart`  
**Node:** `vite`, `react`, `react-dom`, `typescript`, `tailwindcss`, `framer-motion`, `recharts`, `axios`, `zustand`, `@fontsource/syne`, `@fontsource/dm-mono`

---

## Validation Criteria

- [ ] `GET /api/health` çalışıyor
- [ ] `GET /api/models` 3 model, per_class ve training alanlarıyla dönüyor
- [ ] `POST /api/predict` doğru JSON dönüyor
- [ ] Geçersiz dosya tipinde 415, 10MB+ dosyada 413 dönüyor
- [ ] Sayfa açılınca modeller yükleniyor, aktif model set ediliyor
- [ ] Model switch sonraki predict'te yeni modeli kullanıyor, önceki result temizleniyor
- [ ] Classify sırasında scanning animasyonu görünüyor
- [ ] Softmax bar'ları animasyonla açılıyor
- [ ] ModelDetailPanel açılıp kapanıyor, iki sekme çalışıyor
- [ ] Predict hatası error state + "Try again" gösteriyor
- [ ] `npm run build` hatasız tamamlanıyor
- [ ] Production modda tek port (8000) çalışıyor
- [ ] `app.py` bu değişikliklerden etkilenmiyor

---

## What This Plan Does NOT Touch

- `app.py` — Gradio, HF Spaces için olduğu gibi kalır
- `ONNXPredictor` — inference kodu değişmez
- `active_model.json` contract — aynı format
- Mevcut test suite

---

*Plan created: 2026-05-09 · Revised: 2026-05-09*  
*Skill: frontend-design*
