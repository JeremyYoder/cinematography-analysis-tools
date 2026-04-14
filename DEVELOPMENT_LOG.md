# Development & Dependency Log

This log tracks the architectural evolution and dependency constraints of the Cinematography Analysis Tools.

## Project Timeline & High-Level Architecture

| Phase | Milestone | Key Achievements | Status |
| :--- | :--- | :--- | :--- |
| **Phase 1-4** | Legacy Pipeline | Basic classification using ResNet-50. Heavy disk usage (all frames extracted as JPEGs). CPU inference. | ✅ Done |
| **Phase 5** | M1 Hardware Unleashed | Migrated to **FFmpeg pipes** (In-Memory). Integrated **Apple Silicon MPS (Metal)**. Sub-30s speed. | ✅ Done |
| **Phase 6** | Modern Dashboards | Replaced static PNGs with **Plotly** interactive charts. First Gradio release. | ✅ Done |
| **Phase 7** | Systemic Stability | Normalized data schemas (`shot_type` vs `shot-type`). Fix KeyError crashes. | ✅ Done |
| **Phase 8** | Cinematic Intelligence | Added **Predictive Estimator** and **Analysis Presets** (Preview/Deep). | ✅ Done |
| **Phase 9** | Industrial Scale | Added **Visual Decimation** for 2-hour movies and a **Self-Calibrating Perf Log**. | ✅ Done |
| **Phase 11** | Elegant Scale | Upgraded Gradio to support chunked 2GB+ uploads without browser crashes. | ✅ Done |
| **Phase 12** | Silicon-Native | Migrated from Rosetta/Intel to **native ARM64 Python 3.11** via Homebrew. | ✅ Done |
| **Phase 14** | Stabilized Launch | Pinned Pydantic to 2.10.x to fix Gradio 5 schema bug. Cleaned up environment. | ✅ Done |

---

## Dependency Constraints & Issues Log

### 1. The Pydantic Version Constraint
- **Initial state**: Pinned to `pydantic<2.0.0` and `gradio==3.50.2` for legacy compatibility.
- **Phase 11**: Upgraded to Gradio 4.x + Pydantic 2.x for chunked upload support.
- **Phase 12**: Environment rebuilt on native Silicon Python 3.11. Pip resolved to `gradio==5.14.0` and `pydantic==2.12.5`.
- **Bug**: Pydantic 2.12 generates JSON schemas with bare `bool` values. Gradio's client-side parser (`gradio_client/utils.py:898`, `get_type()`) does `if "const" in schema:` which throws `TypeError: argument of type 'bool' is not iterable` when `schema` is `True`/`False`.
- **Resolution (Phase 14)**: Pinned `pydantic>=2.10.6,<2.11.0`. This produces dict-based schemas that Gradio can parse correctly.

### 2. The Mac Matplotlib Conflict
- **Error**: `NSException` and `Segmentation Fault: 11` when plotting on macOS in background threads.
- **Resolution**: Forced `os.environ["MPLBACKEND"] = "Agg"` and shifted to **Plotly** for frontend visualizations. Matplotlib is now strictly for internal math (CIELAB) and hidden buffers.

### 3. FFmpeg Bandwidth
- **Constraint**: Processing 1080p raw bytes was choking the CPU.
- **Resolution**: Implemented hardware-level scaling `scale=224:224` directly in the FFmpeg pipe. This reduced bandwidth by **40x** and enabled the "M1 Real-Time" performance.

---

## Current Build Configuration (v1.0.0-phase14)

| Package | Installed | Constraint | Reasoning |
| :--- | :--- | :--- | :--- |
| `python` | `3.11.15` | `>=3.11` | Native ARM64 via `/opt/homebrew` |
| `torch` | `2.11.0` | `>=1.7` | MPS (Metal) GPU acceleration |
| `gradio` | `5.14.0` | `>=5.14.0` | Chunked uploads for 2GB+ files |
| `pydantic` | `2.10.6` | `>=2.10.6,<2.11.0` | **Pinned** — 2.12+ crashes Gradio |
| `plotly` | `6.7.0` | `>=5.0` | Interactive dashboards |
| `fastapi` | `0.135.3` | `>=0.110.0` | Gradio 5's ASGI backend |
| `FFmpeg` | System | — | `/opt/homebrew/bin`. Source-decimation. |
