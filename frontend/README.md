# TruthLensAI — Frontend Application

Modern editorial intelligence workspace and multimodal verification dashboard for **TruthLensAI**, implemented in React, TypeScript, Vite, and Tailwind CSS. Built to strictly replicate the design system, typography, and component patterns defined in `DESIGN.md` and the Google Stitch project (`TruthLensAi`).

---

## 1. Architectural Highlights

- **Forest Intelligence Visual Identity**:
  - Deep Forest base: `#0A1612` / `#17231F`
  - Sage Green accents: `#B7D8B5` / `#D3F5D0` (for verified/clear states)
  - Copper Highlights: `#D98B52` / `#FFB785` (for tampered/alert states)
  - Warm Ivory: `#F2EDE2` / `#F0EBE0` (for editorial headlines)
- **Strict Two-Page Architecture**:
  1. **Home / Verify** (`/`):
     - Intake & Forensics for **Text**, **Image**, **Video**, and **Audio**.
     - Clear analysis stage progression (Ingestion → AI Signal Extraction → Evidence Fusion).
     - **Strict Signal Separation**: Decoupled cards for Factual Veracity (Real/Fake/Misleading) vs Media Provenance (AI-Generated/Authentic/Uncertain/Unavailable) vs Visual Misinformation Classification (VideoMAE).
      - 10-segment stepped confidence meters.
      - Live wire provenance ledger with clickable sources, domain badges, and reliability tiers.
      - **Point-Wise Research Report** (`ResearchReportPanel.tsx`): renders the backend-built `research_report` block — executive summary, numbered supporting / contradicting / contextual / unresolved findings with clickable sources, expandable per-claim verification cards, the real research methodology trace (queries issued, sources retrieved, failed retrievals, NLI counts) and the unresolved evidence gaps. Every value comes from the API response; the panel never recomputes a verdict or invents a source.
  2. **System Overview**:
     - Interactive 5-level expandable tree architecture.
     - Node detail inspector drawer revealing code entry points (`multimodal_router.py`, `pipeline.py`, `video_verifier.py`, `audio_module`).
     - Real model cards (Groq qwen3.8-27b, VideoMAE-base, EfficientNet-B0 + BiGRU, Wav2Vec2, ViT, YOLO11).
     - Measured evaluation benchmarks grounded in repo artifacts (`eval_metrics.json`, `ai_gen_evaluation_metrics.json`, `videomae_final_candidate_test_metrics.json`).
     - Operational boundary disclosures (lossy compression, dialect bias, ultra-brief audio failsafe).
     - Cryptographic verification seal and IFCN compliance disclosures.

---

## 2. Getting Started & Running

### Prerequisites
- Node.js >= 18 (Tested on v22.17.0)
- TruthLensAI FastAPI backend running on `http://127.0.0.1:8001`
  (`python -m uvicorn api.server:app --host 127.0.0.1 --port 8001`)

### Installation
```bash
cd frontend
npm install
```

### Development Server
```bash
npm run dev
```
The application will launch at `http://localhost:5173`. API requests to `/verify` and `/health` are automatically proxied to `http://127.0.0.1:8001`.

### Production Build
```bash
npm run build
npm run preview
```

---

## 3. Environment Configuration

Optional environment variables can be set in `frontend/.env`:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `VITE_API_URL` | Explicit backend API base URL | `""` (uses Vite proxy to port 8001) |

---

## 4. Testing & Verification

Type-checking and asset bundling:
```bash
npm run build
```
Build output produces an optimized, standalone single-page application under `frontend/dist/`.

Component tests (Vitest + Testing Library, jsdom environment):
```bash
npm run test
```

`src/components/ResearchReportPanel.test.tsx` renders the research report against
the **real** API response captured in `src/test-fixtures/realVerificationResponses.json`
(regenerate it with `python scratch/build_frontend_fixture.py` after a live backend
run). It asserts that the panel displays the backend's actual values — the real
search queries, the real verdict, the real source links — and that it produces no
findings and no links for a response whose research was skipped.
