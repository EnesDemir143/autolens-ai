"""Scrape MICRO class car images from Bing with negative filters.

Output: datasets/scraped-micro/{model_slug}/
Each subfolder maps to an approved MICRO_MODEL_PATTERNS entry.
Images need manual review before adding to the training pipeline.
"""

import time
from pathlib import Path

from icrawler.builtin import BingImageCrawler

# (display_name, search_query, max_images)
QUERIES: list[tuple[str, str, int]] = [
    ("smart_fortwo",   "Smart Fortwo car exterior -toy -diecast -hotwheels -model -interior -parts", 200),
    ("citroen_ami",    "Citroen Ami electric car exterior -toy -diecast -model -interior", 150),
    ("toyota_iq",      "Toyota iQ city car exterior -toy -diecast -model -interior", 150),
    ("renault_twizy",  "Renault Twizy car exterior -toy -diecast -model -interior", 120),
    ("tata_nano",      "Tata Nano car exterior -toy -diecast -model -interior", 120),
    ("bmw_isetta",     "BMW Isetta car exterior -toy -diecast -model -interior -replica", 80),
    ("aixam_city",     "Aixam City microcar exterior -toy -diecast -model", 80),
    ("ligier_js50",    "Ligier JS50 microcar exterior -toy -diecast -model", 60),
    ("peel_p50",       "Peel P50 car exterior -toy -diecast -model -replica -miniature", 60),
    ("smart_forfour",  "Smart Forfour car exterior -toy -diecast -model -interior", 80),
]

OUT_ROOT = Path("datasets/scraped-micro")


def scrape() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for slug, query, max_num in QUERIES:
        out_dir = OUT_ROOT / slug
        out_dir.mkdir(exist_ok=True)

        existing = sum(1 for _ in out_dir.glob("*.jpg")) + sum(1 for _ in out_dir.glob("*.png"))
        if existing >= max_num:
            print(f"⏭  {slug}: zaten {existing} görsel var, atlanıyor")
            continue

        print(f"\n🔍 {slug}: '{query}' ({max_num} görsel hedef)")

        crawler = BingImageCrawler(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": str(out_dir)},
        )
        crawler.crawl(
            keyword=query,
            filters={"type": "photo", "size": "large"},
            max_num=max_num,
            file_idx_offset="auto",
        )

        count = sum(1 for _ in out_dir.glob("*.jpg")) + sum(1 for _ in out_dir.glob("*.png"))
        print(f"✓ {slug}: {count} görsel")
        time.sleep(3)  # rate limit koruması


def summary() -> None:
    print("\n" + "=" * 50)
    print("ÖZET")
    print("=" * 50)
    total = 0
    for slug, _, _ in QUERIES:
        d = OUT_ROOT / slug
        count = sum(1 for _ in d.glob("*.jpg")) + sum(1 for _ in d.glob("*.png")) if d.exists() else 0
        total += count
        print(f"  {slug:<20} {count:>4} görsel")
    print(f"  {'TOPLAM':<20} {total:>4} görsel")
    print(f"\n⚠️  Manuel review gerekli: {OUT_ROOT}/")
    print("   Oyuncak, iç mekan, parça fotoğraflarını sil.")
    print("   Onaylananları build_phase2_dataset.py pipeline'ına ekle.")


if __name__ == "__main__":
    scrape()
    summary()
