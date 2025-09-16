# --- Robust import of utiles.trending with diagnostics ---
import os, sys, traceback, importlib.util
import streamlit as st

ROOT_DIR = os.path.dirname(__file__)
UTIL_DIR = os.path.join(ROOT_DIR, "utiles")
PKG_INIT = os.path.join(UTIL_DIR, "__init__.py")
TREND_PATH = os.path.join(UTIL_DIR, "trending.py")

# Make sure project root is on sys.path
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# Ensure utiles is a proper package
if not os.path.isdir(UTIL_DIR):
    st.error(f"Folder not found: {UTIL_DIR}. Do you have 'utiles/' (not 'utils/')?")
    st.stop()
if not os.path.exists(PKG_INIT):
    # create a blank __init__.py so Python recognizes the package
    try:
        open(PKG_INIT, "a").close()
        st.warning("Created 'utiles/__init__.py' automatically.")
    except Exception as _e:
        st.error(f"Couldn't create {PKG_INIT}: {_e}")
        st.stop()

# Show what's inside for quick sanity
try:
    st.caption("Contents of utiles/: " + ", ".join(sorted(os.listdir(UTIL_DIR))))
except Exception:
    pass

# Try the normal import first
try:
    from utiles.trending import scan_trending, explain_trending_row, TrendScanParams  # type: ignore
    _IMPORTED_VIA = "package"
except Exception as e:
    st.warning(f"Standard import failed: {e!r}. Falling back to path loader…")
    # Print full traceback so we see real cause (syntax error, missing dep, etc.)
    st.code(traceback.format_exc())
    if not os.path.exists(TREND_PATH):
        st.error(f"Cannot find trending.py at {TREND_PATH}")
        st.stop()
    # Load by file path
    spec = importlib.util.spec_from_file_location("utiles.trending", TREND_PATH)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # executes trending.py — any errors will display above
    except Exception:
        st.error("Exception while executing utiles/trending.py:")
        st.code(traceback.format_exc())
        st.stop()
    # Export names
    scan_trending = getattr(mod, "scan_trending", None)
    explain_trending_row = getattr(mod, "explain_trending_row", None)
    TrendScanParams = getattr(mod, "TrendScanParams", None)
    _IMPORTED_VIA = "file"

# Final sanity: all symbols present?
_missing = [n for n,v in {
    "scan_trending": scan_trending,
    "explain_trending_row": explain_trending_row,
    "TrendScanParams": TrendScanParams
}.items() if v is None]
if _missing:
    st.error("utiles.trending is missing: " + ", ".join(_missing))
    st.stop()

st.success(f"Loaded utiles.trending via `{_IMPORTED_VIA}` import.")
