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
| **Phase 11** | Elegant Scale | **Gradio 4 Upgrade** to support chunked 2GB+ uploads without browser crashes. | ✅ Done |

---

## Dependency Constraints & Issues Log

### 1. The Pydantic V1 vs V2 Dilemma
- **Initial Fix**: We pinned to `pydantic<2.0.0` and `gradio==3.50.2` to ensure compatibility with legacy model metadata.
- **Resolution (Phase 11)**: Upgraded to **Gradio 4.30.0** (pinned for Mac stability) + **Pydantic 2.x** + **FastAPI 0.110.0**. 
- **Bug Resolution**: Replaced Gradio 4.44 (buggy schema scraper) with v4.30 to solve `TypeError: bool is not iterable`. This preserves chunked 2GB+ uploads.

### 2. The Mac Matplotlib Conflict
- **Error**: `NSException` and `Segmentation Fault: 11` when plotting on macOS in background threads.
- **Resolution**: Forced `os.environ["MPLBACKEND"] = "Agg"` and shifted to **Plotly** for frontend visualizations. Matplotlib is now strictly for internal math (CIELAB) and hidden buffers.

### 3. FFmpeg Bandwidth
- **Constraint**: Processing 1080p raw bytes was choking the CPU.
- **Resolution**: Implemented hardware-level scaling `scale=224:224` directly in the FFmpeg pipe. This reduced bandwidth by **40x** and enabled the "M1 Real-Time" performance.

---

## Current Build Configuration (v1.0.0-phase9)

| Package | Version | Reasoning |
| :--- | :--- | :--- |
| `torch` | `>=1.7` | Hardware-accelerated MPS support. |
| `gradio` | `3.50.2` | (Current) Old Svelte-based frontend (No chunking). |
| `pydantic` | `<2.0.0` | (Current) V1 compatibility. |
| `plotly` | `>=5.0` | Interactive dashboards. |
| `FFmpeg` | `System` | Path: `/usr/local/bin`. Handles source-decimation. |
