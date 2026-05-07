"""Import AutoLens AI project structure into a Plane project.

Usage:
  cp .env.plane.local.example .env.plane.local
  # fill PLANE_API_KEY and, optionally, PLANE_PROJECT_ID
  uv run python scripts/import_project_to_plane.py

The script follows /Users/enesdemir/Documents/plane_project_template.md:
- reads .env.plane.local itself
- uses X-API-Key authentication
- reuses modules/work items by name
- patches existing work item descriptions/states
- can create the Plane project when PLANE_PROJECT_ID is missing
"""

from __future__ import annotations

import html
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class ModuleSpec:
    name: str
    description: str
    status: str = "planned"


@dataclass(frozen=True)
class WorkItemSpec:
    name: str
    module: str
    priority: str
    description: str
    state_group: str = "unstarted"
    parent_phase: str | None = None


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    module: str
    priority: str
    description: str
    state_group: str


PROJECT_NAME = "autolens-ai"
PROJECT_IDENTIFIER = "AUTO"
PROJECT_DESCRIPTION = (
    "AutoLens AI, Kocaeli Üniversitesi Yazılım Laboratuvarı-II Proje III için geliştirilen "
    "8 sınıflı araç gövde tipi sınıflandırma projesidir. Python 3.12, uv, PyTorch/Lightning, "
    "DINOv3/CNN karşılaştırmaları ve Gradio arayüzü kullanarak yüklenen araç görselinden hızlı, "
    "açıklanabilir ve raporlanabilir tahmin üretir. Ana hedef, görülmemiş sunum/test görsellerinde "
    "makro F1 öncelikli doğru sınıflandırma ve 95 MB altı teslim edilebilir model sağlamaktır."
)

MODULES = [
    ModuleSpec("Product / UX", "Gradio demo deneyimi, upload/preview/prediction akışı ve sunum hazırlığı."),
    ModuleSpec("Dataset / Curation", "Açık kaynak Kaggle/Hugging Face veri kaynakları, 8 sınıf etiket eşleme, EDA, denge ve split yönetimi."),
    ModuleSpec("Training Pipeline", "PyTorch Lightning eğitim döngüsü, MPS/CPU fallback, augmentasyon, checkpoint ve baseline modeller."),
    ModuleSpec("Model Comparison / Evaluation", "DINOv3 ana yol, baseline karşılaştırmaları, Accuracy/Precision/Recall/F1, confusion matrix, boyut ve hız ölçümü."),
    ModuleSpec("Inference / Demo", "Final model yükleme, preprocessing parity, sınıf olasılıkları ve canlı demo inference adaptörü."),
    ModuleSpec("Reports / Docs", "IEEE LaTeX raporu, dataset/model/metric/UI kanıtları, README ve faz raporları."),
    ModuleSpec("Infrastructure / DevEx", "uv/Python 3.12 proje temeli, pyproject, kalite kapıları, repo yapısı ve geliştirici komutları."),
    ModuleSpec("Testing & Quality", "pytest, ruff, mypy, veri sızıntısı kontrolleri, smoke testleri ve doğrulama kayıtları."),
    ModuleSpec("Backlog / Optimization", "ONNX, Optuna, Grad-CAM, LoRA ve zaman kalırsa yapılacak optimizasyonlar."),
]

PHASE_ITEMS = [
    PhaseSpec(
        "Phase 1 — Project Foundation",
        "Infrastructure / DevEx",
        "high",
        "Tamamlandı: uv/Python 3.12 proje iskeleti ve kalite kapıları kuruldu.\n\n"
        "- pyproject, src/autolens_ai ve tests temeli oluşturuldu.\n"
        "- Runtime/dev bağımlılıkları docs/dependencies.md içinde belgelendi.\n"
        "- ruff, mypy ve pytest komutları Makefile üzerinden doğrulandı.\n"
        "- ENV-01, ENV-02 ve ENV-03 gereksinimleri 2026-05-06 tarihinde valide edildi.",
        "completed",
    ),
    PhaseSpec(
        "Phase 2 — Dataset Research and Curation",
        "Dataset / Curation",
        "high",
        "Planlandı: açık kaynaklardan dokümante, yaklaşık dengeli 8 sınıflı araç gövde tipi veri seti üretilecek.\n\n"
        "- Kaggle/Hugging Face ve metadata-backed kaynaklar kaynak kimliği/URL/lisans notuyla kataloglanacak.\n"
        "- Raw etiketler SUV, VAN, STATION WAGON, MICRO, OPEN WHEEL, SEDAN, HATCHBACK, PICK UP sınıflarına eşlenecek.\n"
        "- Manifest, EDA, kalite/outlier/duplicate/corrupt-image inceleme listeleri üretilecek.\n"
        "- Final test sızıntısı olmadan train/validation/internal-test split oluşturulacak.",
        "unstarted",
    ),
    PhaseSpec(
        "Phase 3 — Baseline Training Pipeline",
        "Training Pipeline",
        "high",
        "Planlandı: yeniden kullanılabilir Lightning eğitim/evaluation hattı ve CNN baseline modelleri çalışır hale getirilecek.\n\n"
        "- MobileNetV4 Conv Medium, EfficientNet-B2 ve bir ResNet baseline/comparison modeli eğitilecek.\n"
        "- MPS varsa MPS, yoksa CPU fallback desteklenecek.\n"
        "- Augmentation/normalization parity, early stopping ve checkpointing eklenecek.\n"
        "- Başarı kriteri: checkpoint + validation metric + açıklanabilir küçük kod yolu.",
        "unstarted",
    ),
    PhaseSpec(
        "Phase 4 — Main DINOv3 Model and Selection",
        "Model Comparison / Evaluation",
        "high",
        "Planlandı: DINOv3 ViT-S/16 ana model yolu uygulanıp baseline modellerle F1/size/speed temelli seçilecek.\n\n"
        "- facebook/dinov3-vits16-pretrain-lvd1689m erişimi erken doğrulanacak; bloklanırsa fallback yazılacak.\n"
        "- LoRA opsiyonel kalacak; önce non-LoRA ana yol değerlendirilecek.\n"
        "- Accuracy, Precision, Recall, F1, per-class/macro/weighted metrikler üretilecek.\n"
        "- Loss/accuracy eğrileri, normalize confusion matrix, artifact size ve latency raporlanacak.",
        "unstarted",
    ),
    PhaseSpec(
        "Phase 5 — Gradio Demo Interface",
        "Product / UX",
        "high",
        "Planlandı: sunuma hazır modern Gradio web demo arayüzü teslim edilecek.\n\n"
        "- Upload/drag-drop, image preview ve tahmin butonu/akışı olacak.\n"
        "- Predicted class, confidence ve 8 sınıf probability chart gösterilecek.\n"
        "- Validation preprocessing ile inference preprocessing aynı tutulacak.\n"
        "- Başarı kriteri: canlı demo için kabul edilebilir latency ve temiz görsel düzen.",
        "unstarted",
    ),
    PhaseSpec(
        "Phase 6 — Final Evidence and IEEE Report",
        "Reports / Docs",
        "high",
        "Planlandı: teslim kanıtları dondurulup IEEE formatında final rapor üretilecek.\n\n"
        "- Final model/checkpoint, class mapping, metrics, plots, confusion matrix ve UI kanıtları paketlenecek.\n"
        "- awesome-ieee-report / LaTeX report creator skill ile minimum 4 sayfa IEEE rapor hazırlanacak.\n"
        "- README install/train/evaluate/run UI/reproduce akışını açıklayacak.\n"
        "- Rapor dataset kaynakları, preprocessing, mimari gerekçe, model karşılaştırma ve UI açıklamasını içerecek.",
        "unstarted",
    ),
]

WORK_ITEMS = [
    WorkItemSpec("Kaynak katalog ve lisans manifestini tamamla", "Dataset / Curation", "high", "Kaggle/Hugging Face/metadata kaynakları ID, URL, lisans, sınıf kapsama ve indirme notlarıyla tek manifestte topla.", parent_phase="Phase 2 — Dataset Research and Curation"),
    WorkItemSpec("8 sınıf label mapping kararlarını kilitle", "Dataset / Curation", "high", "SUV, VAN, STATION WAGON, MICRO, OPEN WHEEL, SEDAN, HATCHBACK, PICK UP mapping tablosu ve ambiguous-label exclusion kararlarını belgele.", parent_phase="Phase 2 — Dataset Research and Curation"),
    WorkItemSpec("EDA ve veri kalite raporlarını üret", "Dataset / Curation", "high", "Class/source distribution, image quality, duplicate/corrupt/outlier listeleri ve zayıf sınıf raporları üret.", parent_phase="Phase 2 — Dataset Research and Curation"),
    WorkItemSpec("Leakage-safe split manifestini oluştur", "Dataset / Curation", "high", "Train/validation/internal-test splitlerini final instructor test verisi kullanılmadan ve source/domain bias notlarıyla oluştur.", parent_phase="Phase 2 — Dataset Research and Curation"),
    WorkItemSpec("Lightning DataModule ve transform parity ekle", "Training Pipeline", "high", "Dataset manifestinden çalışan dataloader, train/val augmentation ve deterministic validation preprocessing katmanını kur.", parent_phase="Phase 3 — Baseline Training Pipeline"),
    WorkItemSpec("Baseline model factory kur", "Training Pipeline", "high", "timm/torchvision tabanlı MobileNetV4/EfficientNet-B2/ResNet model factory ve substitution notlarını ekle.", parent_phase="Phase 3 — Baseline Training Pipeline"),
    WorkItemSpec("Training loop, checkpoint ve metric logging tamamla", "Training Pipeline", "high", "MPS/CPU fallback, early stopping, checkpointing, torchmetrics ve W&B/yerel log kanıtlarını çalıştır.", parent_phase="Phase 3 — Baseline Training Pipeline"),
    WorkItemSpec("DINOv3 erişim ve fallback kararını belgeleyip uygula", "Model Comparison / Evaluation", "high", "DINOv3 gated erişimini doğrula; bloke olursa raporlanabilir fallback mimarisini seç ve uygulamaya bağla.", parent_phase="Phase 4 — Main DINOv3 Model and Selection"),
    WorkItemSpec("Model benchmark tablosu ve seçim notunu üret", "Model Comparison / Evaluation", "high", "Makro F1 öncelikli, accuracy/precision/recall, size <95 MB ve latency karşılaştırma tablosuyla final modeli seç.", parent_phase="Phase 4 — Main DINOv3 Model and Selection"),
    WorkItemSpec("Zorunlu grafik ve confusion matrix üretimini otomatikleştir", "Model Comparison / Evaluation", "high", "Loss/accuracy eğrileri ve 8x8 normalized confusion matrix çıktıları tekrar üretilebilir script/komutla oluşmalı.", parent_phase="Phase 4 — Main DINOv3 Model and Selection"),
    WorkItemSpec("Inference adapter ve class probability çıktısını bağla", "Inference / Demo", "high", "Final artifact + class mapping yüklenerek predicted class, confidence ve 8 sınıf probability dict/array üret.", parent_phase="Phase 5 — Gradio Demo Interface"),
    WorkItemSpec("Gradio Blocks UI akışını tasarla", "Product / UX", "high", "Upload/preview/result/probability chart layout modern, sade ve sunum-ready olacak şekilde kur.", parent_phase="Phase 5 — Gradio Demo Interface"),
    WorkItemSpec("Demo smoke test ve örnek kullanım dokümantasyonu ekle", "Testing & Quality", "medium", "UI import/smoke testleri ve README demo komutları ile canlı sunum öncesi hızlı doğrulama sağla.", parent_phase="Phase 5 — Gradio Demo Interface"),
    WorkItemSpec("Final evidence freeze checklist oluştur", "Reports / Docs", "high", "Metric CSV/JSON, plots, class mapping, model artifact size, UI screenshot ve dataset manifest kanıtlarını tek yerde listele.", parent_phase="Phase 6 — Final Evidence and IEEE Report"),
    WorkItemSpec("IEEE LaTeX report source üret", "Reports / Docs", "high", "awesome-ieee-report skill çıktısı olarak .tex/.bib ve gerekli figür referanslarını üret; minimum 4 sayfa hedefini kontrol et.", parent_phase="Phase 6 — Final Evidence and IEEE Report"),
    WorkItemSpec("v2 Backlog — ONNX / Optuna / Grad-CAM / LoRA", "Backlog / Optimization", "low", "Backlog: ana teslim tamamlandıktan sonra ONNX export, Optuna tuning, Grad-CAM ve DINOv3 LoRA opsiyonlarını değerlendir.", "backlog"),
]

PROJECT_PAGE_MD = """# AutoLens AI Overview

AutoLens AI, Kocaeli Üniversitesi Yazılım Laboratuvarı-II Proje III için geliştirilen 8 sınıflı araç gövde tipi sınıflandırma projesidir. Python 3.12/uv, PyTorch Lightning, DINOv3/CNN karşılaştırmaları ve Gradio web demo akışı kullanır.

## Ana hedef

Görülmemiş sunum/test görsellerinde doğru gövde tipi tahmini yapmak; makro F1'i birinci öncelik olarak optimize etmek; final model artifact'ını 95 MB altında tutmak; Accuracy, Precision, Recall, F1 ve normalized confusion matrix ile raporlanabilir kanıt üretmek.

## Sınıflar

- SUV
- VAN
- STATION WAGON
- MICRO
- OPEN WHEEL / F1
- SEDAN
- HATCHBACK
- PICK UP

## Ana modüller

- Product / UX
- Dataset / Curation
- Training Pipeline
- Model Comparison / Evaluation
- Inference / Demo
- Reports / Docs
- Infrastructure / DevEx
- Testing & Quality
- Backlog / Optimization

## Kısa teslim hedefi

30.05.2026 deadline öncesinde çalışan dataset pipeline, karşılaştırmalı model sonuçları, sunuma hazır Gradio demo ve IEEE LaTeX raporu üretmek.
"""


def plane_project_name(value: str) -> str:
    """Plane self-host rejects hyphens in project name; keep local target as autolens-ai and use API-safe display."""
    return value.replace("-", " ")


def slugify(value: str) -> str:
    text = value.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "autolens-ai"


def load_dotenv_file(path: str = ".env.plane.local") -> None:
    """Load simple KEY=VALUE dotenv files without requiring shell-specific source syntax."""
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for raw_line in dotenv.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env(name: str, default: str | None = None, *, required: bool = True) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise SystemExit(f"Missing env: {name}")
    return (value or "").rstrip("/")


def html_desc(text: str) -> str:
    escaped = html.escape(text)
    paragraphs = [p.strip().replace("\n", "<br>") for p in escaped.split("\n\n") if p.strip()]
    return "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)


class PlaneClient:
    def __init__(self) -> None:
        self.base_url = env("PLANE_BASE_URL", "http://localhost:8080")
        self.workspace = env("PLANE_WORKSPACE")
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": env("PLANE_API_KEY"), "Content-Type": "application/json"},
            timeout=20,
        )
        self.project_id = env("PLANE_PROJECT_ID", required=False)

    def workspace_path(self, suffix: str) -> str:
        return f"/api/v1/workspaces/{self.workspace}/{suffix.lstrip('/')}"

    def path(self, suffix: str) -> str:
        if not self.project_id:
            raise SystemExit("PLANE_PROJECT_ID is still missing after project resolution")
        return self.workspace_path(f"projects/{self.project_id}/{suffix.lstrip('/')}")

    def get_json(self, path: str) -> Any:
        response = self.client.get(path)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "results" in data:
            return data["results"]
        return data

    def post_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = self.client.post(path, json=payload)
        if response.status_code >= 400:
            print(f"POST failed {response.status_code}: {path}\n{response.text[:1000]}", file=sys.stderr)
        response.raise_for_status()
        return response.json() if response.content else {}

    def patch_json(self, path: str, payload: dict[str, Any]) -> Any:
        response = self.client.patch(path, json=payload)
        if response.status_code >= 400:
            print(f"PATCH failed {response.status_code}: {path}\n{response.text[:1000]}", file=sys.stderr)
        response.raise_for_status()
        return response.json() if response.content else {}

    def resolve_project(self) -> None:
        if self.project_id:
            return

        desired_name = env("PLANE_PROJECT_NAME", PROJECT_NAME, required=False) or PROJECT_NAME
        api_name = plane_project_name(desired_name)
        projects = self.get_json(self.workspace_path("projects/"))
        for project in projects:
            if isinstance(project, dict) and project.get("name") in {desired_name, api_name}:
                self.project_id = str(project["id"])
                print(f"= project exists: {desired_name} ({self.project_id})")
                return

        payload = {
            "name": api_name,
            "identifier": env("PLANE_PROJECT_IDENTIFIER", PROJECT_IDENTIFIER, required=False) or PROJECT_IDENTIFIER,
            "description": PROJECT_DESCRIPTION,
            "description_html": html_desc(PROJECT_DESCRIPTION),
            "network": 2,
            "module_view": True,
            "issue_views_view": True,
            "page_view": True,
            "intake_view": False,
        }
        created = self.post_json(self.workspace_path("projects/"), payload)
        self.project_id = str(created["id"])
        print(f"+ project created: {api_name} ({self.project_id})")

    def update_project_description(self) -> None:
        payload = {
            "name": plane_project_name(PROJECT_NAME),
            "identifier": env("PLANE_PROJECT_IDENTIFIER", PROJECT_IDENTIFIER, required=False) or PROJECT_IDENTIFIER,
            "description": PROJECT_DESCRIPTION,
            "description_html": html_desc(PROJECT_DESCRIPTION),
            "module_view": True,
            "issue_views_view": True,
            "page_view": True,
            "intake_view": False,
        }
        self.patch_json(self.workspace_path(f"projects/{self.project_id}/"), payload)
        print("= project description updated")

    def state_ids_by_group(self) -> dict[str, str]:
        states = self.get_json(self.path("states/"))
        return {
            state.get("group"): state.get("id")
            for state in states
            if isinstance(state, dict) and state.get("group") and state.get("id")
        }


def write_project_id_hint(project_id: str) -> None:
    env_path = Path(".env.plane.local")
    if not env_path.exists():
        return
    lines = env_path.read_text().splitlines()
    for index, line in enumerate(lines):
        if line.startswith("PLANE_PROJECT_ID=") or line.startswith("# PLANE_PROJECT_ID="):
            lines[index] = f"PLANE_PROJECT_ID={project_id}"
            break
    else:
        lines.append(f"PLANE_PROJECT_ID={project_id}")
    env_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    load_dotenv_file()
    plane = PlaneClient()
    plane.resolve_project()
    write_project_id_hint(plane.project_id)
    print(f"Importing into {plane.base_url}/{plane.workspace}/projects/{plane.project_id}")
    plane.update_project_description()

    existing_modules_raw = plane.get_json(plane.path("modules/"))
    existing_modules = {module.get("name"): module for module in existing_modules_raw if isinstance(module, dict)}

    module_ids: dict[str, str] = {}
    for spec in MODULES:
        current = existing_modules.get(spec.name)
        if current:
            module_ids[spec.name] = current["id"]
            print(f"= module exists: {spec.name}")
            continue
        created = plane.post_json(
            plane.path("modules/"),
            {"name": spec.name, "description": spec.description, "status": spec.status},
        )
        module_ids[spec.name] = created["id"]
        print(f"+ module created: {spec.name}")

    state_ids = plane.state_ids_by_group()
    default_state = state_ids.get("unstarted") or state_ids.get("backlog")

    existing_by_title: dict[str, dict[str, Any]] = {}
    try:
        existing_items_raw = plane.get_json(plane.path("work-items/"))
        existing_by_title = {
            item.get("name"): item for item in existing_items_raw if isinstance(item, dict) and item.get("name")
        }
    except httpx.HTTPStatusError as exc:
        print(f"! Could not list existing work items; will create without duplicate check: {exc.response.status_code}")

    def upsert_work_item(
        name: str,
        module: str,
        priority: str,
        description: str,
        state_group: str,
        parent_id: str | None = None,
    ) -> str:
        module_id = module_ids.get(module)
        state_id = state_ids.get(state_group) or default_state
        payload: dict[str, Any] = {
            "name": name,
            "description_html": html_desc(description),
            "priority": priority,
        }
        if module_id:
            payload["module"] = module_id
        if state_id:
            payload["state"] = state_id
        if parent_id:
            payload["parent"] = parent_id

        existing = existing_by_title.get(name)
        if existing:
            plane.patch_json(plane.path(f"work-items/{existing['id']}/"), payload)
            print(f"= work item updated: {name} [{state_group}]" + (" as sub-item" if parent_id else ""))
            return str(existing["id"])
        created = plane.post_json(plane.path("work-items/"), payload)
        if isinstance(created, dict) and created.get("id"):
            existing_by_title[name] = created
        print(f"+ work item created: {name} [{state_group}]" + (" as sub-item" if parent_id else ""))
        return str(created["id"])

    phase_ids: dict[str, str] = {}
    for phase in PHASE_ITEMS:
        phase_ids[phase.name] = upsert_work_item(
            phase.name, phase.module, phase.priority, phase.description, phase.state_group
        )

    for item in WORK_ITEMS:
        parent_id = phase_ids.get(item.parent_phase or "")
        upsert_work_item(item.name, item.module, item.priority, item.description, item.state_group, parent_id)

    print("\nPages API self-host sürümünde dokümante olmayabilir. UI > Pages içine şunu kopyalayabilirsin:\n")
    print(PROJECT_PAGE_MD)


if __name__ == "__main__":
    main()
