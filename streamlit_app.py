# testV2.py
# Run simply with:  python testV2.py

from __future__ import annotations
import sys, json
from pathlib import Path

# ------------------------------------------------------
# Make 'genaids' (which contains 'libs') importable
# Folder layout:
#   .../new/genaids/
#       libs/pdfExtractor/...
#       apps/drawingValidation/backend/testV2.py  <-- this file
# ------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
# parents: [0]=backend, [1]=drawingValidation, [2]=apps, [3]=genaids, [4]=new
GENAIDS_DIR = THIS_FILE.parents[3]  # .../genaids
if str(GENAIDS_DIR) not in sys.path:
    sys.path.insert(0, str(GENAIDS_DIR))

# ------------------------------------------------------
# Now these imports will work
# ------------------------------------------------------
from libs.pdfExtractor.PDFExtractor import PDFExtractor
from libs.symbolDetector.DimensionDetectorV2 import CleanDimensionDetector

try:
    import yaml
except Exception:
    yaml = None

# -----------------------------
# Defaults (no CLI)
# -----------------------------
BACKEND_DIR = THIS_FILE.parent
DRAWINGVAL_DIR = BACKEND_DIR.parent
APPS_DIR = DRAWINGVAL_DIR.parent
PROJECT_ROOT = GENAIDS_DIR.parent

DEFAULT_PDF = BACKEND_DIR / "exampleP1.pdf"   # change if needed
OUT_DIR = BACKEND_DIR / "dimension_results"

# Candidate config locations (checked in order)
CONFIG_CANDIDATES = [
    BACKEND_DIR / "config" / "default.yaml",
    DRAWINGVAL_DIR / "config" / "default.yaml",
    GENAIDS_DIR / "config" / "default.yaml",
    PROJECT_ROOT / "config" / "default.yaml",
    BACKEND_DIR / "config" / "default.json",
    DRAWINGVAL_DIR / "config" / "default.json",
    GENAIDS_DIR / "config" / "default.json",
    PROJECT_ROOT / "config" / "default.json",
]

MINIMAL_CONFIG_YAML = """\
extractor:
  zone_grouping:
    enabled: false
    zone_size: 120
  shape_grouping:
    enabled: false
    distance_threshold: 12
    tile_size: 100
  use_pdfplumber_tables: true
"""

def load_config(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml") and yaml is not None:
        return yaml.safe_load(text)
    return json.loads(text)

def find_or_create_config(out_dir: Path) -> Path:
    for p in CONFIG_CANDIDATES:
        if p.is_file():
            return p
    conf_dir = out_dir / "autoconfig"
    conf_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = conf_dir / "default.yaml"
    cfg_path.write_text(MINIMAL_CONFIG_YAML, encoding="utf-8")
    return cfg_path

def flatten_elements(maybe_grouped):
    if isinstance(maybe_grouped, dict):
        flat = []
        for zone_elems in maybe_grouped.values():
            if zone_elems:
                flat.extend(zone_elems)
        return flat
    return maybe_grouped

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_path = DEFAULT_PDF
    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF not found at {pdf_path}\n"
            f"Put a sample PDF there, or edit DEFAULT_PDF in testV2.py"
        )

    cfg_path = find_or_create_config(OUT_DIR)
    config = load_config(cfg_path)

    extractor = PDFExtractor(config=config)
    elements = extractor.extract_elements(str(pdf_path))
    elements = flatten_elements(elements)

    detector = CleanDimensionDetector(confidence_threshold=0.80)
    detections = detector.detect(elements)

    json_path, annotated_path = detector.save_results(str(OUT_DIR), pdf_path.stem, str(pdf_path))

    print("\n=== Dimension Detection ===")
    print(f"PDF: {pdf_path}")
    print(f"Config: {cfg_path}")
    print(f"Detections: {len(detections)}")
    print(f"JSON: {json_path}")
    print(f"Annotated PDF: {annotated_path or '(none)'}")

if __name__ == "__main__":
    main()
