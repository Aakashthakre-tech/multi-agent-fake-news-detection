# TruthLensAI — Design System & Screen Specifications

Retrieved and synthesized directly from Stitch project **TruthLensAi** (`projects/8272803068634163835`).

---

## 1. Project Metadata & Overview

- **Project ID**: `8272803068634163835`
- **Project Title**: `TruthLensAi`
- **Origin**: Stitch (`TEXT_TO_UI_PRO`)
- **Primary Form Factor**: Mobile (`390px` viewport baseline, responsive up to desktop multi-pane)
- **Visual Personality**: Editorial Intelligence & Scientific Cryptography. A marriage of investigative journalism newsroom broadsheets and high-security digital forensics laboratory tools. Avoids generic sci-fi neon glows in favor of restrained, parchment-and-slate dark tones, authoritative serif headers, and monospaced technical telemetry.

---

## 2. Color System & Design Tokens

The color architecture is rooted in Material Design 3 (M3) dark mode with tailored organic vegetal undertones.

### 2.1 Core Palette Tokens

| Token Name | Hex Code | Role & Description |
| :--- | :--- | :--- |
| `background` / `surface` | `#0a1612` | Deep forest dark canvas base, glare-free substrate |
| `surface-dim` | `#0a1612` | Dim ground level |
| `surface-container-lowest` | `#05100d` | Deep inset troughs, media wells, tab containers |
| `surface-container-low` | `#121e1a` | Low-elevation cards, secondary telemetry blocks |
| `surface-container` | `#16221e` | Primary card background, evidence modules |
| `surface-container-high` | `#202c28` | Elevated pills, active badge containers, button backdrops |
| `surface-container-highest`| `#2b3733` | Inspecting panels, thumbnail borders, inactive steps |
| `surface-bright` | `#2f3c37` | Highlighted buttons, active tab backdrops |
| `on-surface` | `#d8e6df` | Primary readable text (warm pale sage) |
| `on-surface-variant` | `#c3c8bf` | Subtitles, helper text, neutral descriptors |
| `outline` | `#8d928a` | Structural borders, divider lines |
| `outline-variant` | `#434841` | Low-contrast containment borders, arrows |
| `primary` | `#d3f5d0` | Verified clearance, authenticated nodes, high-contrast highlights |
| `primary-container` | `#b7d8b5` | Sage Green accent; primary action button background |
| `on-primary` | `#1b361e` | Dark text on primary buttons |
| `on-primary-container` | `#425f43` | Dark contrast text for primary tint badges |
| `secondary` | `#ffb785` | Copper / Amber alert; tampered signals, manipulation flags |
| `secondary-container` | `#713702` | Deep amber tint for alert badge backgrounds |
| `on-secondary-container` | `#f5a267` | Warm copper text for tamper callouts |
| `tertiary` | `#f0ebe0` | Warm Ivory; prominent headlines, dossier emphasis |
| `tertiary-container` | `#d3cfc4` | Ivory container backdrop |
| `on-tertiary-container` | `#5a5850` | Muted taupe body text |
| `error` | `#ffb4ab` | Critical failure, manifest verification fault |
| `error-container` | `#93000a` | High-priority failure containers |

### 2.2 Elevation & Border Principles
- **No Heavy Shadows**: Elevation is communicated through layered tonal contrast (`#05100d` → `#121e1a` → `#16221e` → `#202c28` → `#2b3733`) and delicate 1px containment borders (`#2D4039` or `#434841`).
- **Focus Highlighting**: Status transitions occur via border color changes (Sage `#b7d8b5` for authenticated, Copper `#ffb785` / `#d98b52` for tamper anomalies) without fluorescent glow.

---

## 3. Typography Scale & Fonts

The system pairs classical editorial broadsheets with cryptographic forensics.

### 3.1 Font Families
- **Headline Font**: `Newsreader` (Serif, variable optical size, weights 400 & 500) — Editorial authority, broadsheet integrity.
- **Body Font**: `Inter` (Sans-serif, weights 400, 500, 600) — High-density neutral UI clarity.
- **Label / Forensic Font**: `JetBrains Mono` (Monospace, weights 400, 500, 600) — Hashes, EXIF telemetry, timestamps, latency metrics, case IDs.

### 3.2 Typography Scale

| Class | Font Family | Size | Line Height | Tracking | Weight | Usage |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `headline-xl` | Newsreader | 48px | 56px | -0.02em | 400 | Desktop hero titles |
| `headline-xl-mobile` | Newsreader | 32px | 40px | -0.01em | 400 | Mobile hero banners |
| `headline-lg` | Newsreader | 36px | 44px | -0.015em| 400 | Section headers |
| `headline-lg-mobile` | Newsreader | 26px | 34px | -0.01em | 400 | Mobile page titles |
| `headline-md` | Newsreader | 24px | 32px | -0.01em | 500 | Card titles & stats |
| `headline-sm` | Newsreader | 20px | 28px | Normal | 500 | Subsection headings |
| `body-lg` | Inter | 16px | 26px | Normal | 400 | Long-form narratives |
| `body-md` | Inter | 14px | 22px | Normal | 400 | Standard body copy |
| `body-sm` | Inter | 12px | 18px | Normal | 400 | Compact card explanations |
| `label-lg` | Inter | 13px | 18px | +0.01em | 600 | Buttons & primary tabs |
| `label-md` | JetBrains Mono | 11px | 16px | +0.04em | 500 | Monospace badges, telemetry |
| `label-sm` | JetBrains Mono | 10px | 14px | +0.06em | 400 | Micro-stamps, hashes, dates |

---

## 4. Layout Rules, Shapes & Spacing

### 4.1 Corner Radii
- `rounded-sm`: `0.125rem` (2px) — Meter pips, micro chips.
- `DEFAULT` / `rounded`: `0.25rem` (4px) — Precision chips, input fields, badges.
- `rounded-md`: `0.375rem` (6px) — Media thumbnails, sub-containers.
- `rounded-lg`: `0.5rem` (8px) — Buttons, interactive cards.
- `rounded-xl`: `0.75rem` (12px) — Major sectional cards, modals, navigation containers.
- `rounded-full`: `9999px` — Avatars, status indicator dots.

### 4.2 Spacing Matrix
- `space-xs`: `0.25rem` (4px)
- `space-sm`: `0.5rem` (8px)
- `space-md`: `1.0rem` (16px)
- `space-lg`: `1.5rem` (24px)
- `space-xl`: `2.5rem` (40px)
- `margin-mobile`: `1.0rem` (16px)
- `gutter-mobile`: `0.75rem` (12px)

### 4.3 Viewport & Chrome Guidelines
- **Mobile Fixed Navigation**: Fixed top header (`h-16`, backdrop-blur, `pt-safe`) and fixed bottom navigation bar (`h-16`, 2-column tab layout, `pb-safe`).
- **Scroll Behavior**: Hidden scrollbars (`::-webkit-scrollbar { display: none; }`), smooth vertical flow with disciplined padding buffers (`pt-16 pb-20`).

---

## 5. Component Patterns & UI Elements

### 5.1 Sticky Editorial Top Bar
- **Branding**: Logo icon (`h-8`), Newsreader serif brand title `TruthLens`, mono badge `AI`, and edition tag `DESK // v3.4-LIVE`.
- **Telemetry / Actions**: Active investigations queue button with amber counter badge (`fact_check` icon + count `4`), user avatar bubble.
- **Backdrop**: Translucent dark slate `bg-surface/85 backdrop-blur-xl shadow-[0_1px_8px_rgba(0,0,0,0.35)]`.

### 5.2 Segmented Multimodal Selector
- **Container**: `bg-surface-container-lowest p-1 rounded-xl` with horizontal scroll capability.
- **Pills**: `px-3 py-1 rounded-lg font-mono text-[11px]`.
- **States**:
  - Inactive: `text-on-surface-variant hover:text-on-surface`.
  - Active: `bg-surface-container-high text-primary font-semibold shadow-sm`.
- **Modes**: Text Analysis, Image Forensics, Video / Audio, Batch URL.

### 5.3 Source Ingestion Well & Artifact Staging
- **Well Header**: Monospace uppercase header with pulsating `INTAKE READY` or `sensors` status icon.
- **Artifact Preview**: Staged media card with thumbnail (`w-16 h-16 rounded-md`), duration badge (`12.4s`), artifact token (`Artifact #894`), truncated SHA-256 string, and source provenance URI.
- **Drop / Paste Trigger**: Inset field with `add_link` icon, instruction label, and dedicated `PASTE` action pill.
- **Format Support Chips**: Monospace micro-chips (`JPEG/PNG/WebP`, `MP4 / WAV`, `ClaimReview Schema`).

### 5.4 Primary Adjudicated Verdict Dossier Card
- **Atmospheric Glow**: Ambient corner radial gradient (`bg-secondary-container/20 rounded-full blur-2xl`).
- **Classification Header**: Gavel icon + Adjudicated Verdict standard indicator (`ISO-17025 STANDARD`).
- **Verdict Statement**: High-contrast Serif verdict headline (`SYNTHETIC FABRICATION` in copper amber) + descriptive sub-profile.
- **Metric Readout**: Large serif percentage (`94.2%`) paired with monospace confidence tag.
- **Stepped Confidence Meter**: 10 discrete dash segments (`grid grid-cols-10 gap-1 h-2`) reflecting scientific thresholds rather than misleading smooth gradients.

### 5.5 Multi-Vector Forensic Evidence Breakdown
Cards displaying decomposed forensic pipelines:
1. **Visual PRNU & Noise Residue**: Side-by-side spectral thumbnails (PRNU spectra vs. frame seam magnification), anomaly frame ranges, diffusion blend lines.
2. **Formant & Vocoder Signature**: Dual-line waveform vector spectrogram (target vs. synthetic clone baseline) with clone fingerprint match note (`OpenVoice-v2 Clone Fingerprint Matched`).
3. **Wire Trace & Corpus Propagation**: Cross-referencing against Reuters FactNet, AP Wire, and state pool cameras with earliest attribution timestamps.

### 5.6 Provenance & Cross-Reference Ledger
- Items structured with leading icon containers:
  - **C2PA Manifest Check**: Cryptographic trust chain status (`FAILED` / `PASSED`).
  - **Debunk Wire Archive**: Corroborating historical fact-checks with matching stems (`Reuters Wire Debunk Archive Match #24-0812`).

### 5.7 Desk Actions Toolbar
- **Primary CTA**: Full-width button in `bg-primary-container text-on-primary` (`Export Evidence Dossier (PDF)`).
- **Secondary Actions**: Two-column utility buttons (`Flag Syndication`, `Add Annotation`).
- **Digital Seal Stamp**: Monospace verification hash stamp (`DIGITAL SEAL: ED-901-TL` and ISO UTC timestamp).

### 5.8 Multi-Stage Pipeline Flow (Overview Screen)
- Stepped stages linked by subtle downward directional indicators (`arrow_downward`).
- Each stage features stage tag (`STAGE 01`), latency/concurrency specs, descriptive bullets, and sub-metric split cards.
- Stages documented:
  1. Invariant Preprocessing & Parsing (Header parsing, 16kHz Mel-Log extraction).
  2. Parallel Forensic Ensembles (Visual ViT-H/14, Acoustic Conformer-2, Semantic Wire dense vector lookup).
  3. Bayesian Evidentiary Calibration (Subjective logic anti-hallucination normalization, 0.85 threshold meter).
  4. Analyst Workbench & Dossier Issuance (Human-in-the-loop gate).

### 5.9 Benchmark & Boundary Disclosure Modules
- **Benchmark Cards**: Tabular empirical evaluation rows with AUC, EER, and False Positive thresholds against attested corpora (`DFDC-v2`, `AV-Spoof`).
- **Boundary Disclosure Cards**: Warning-accented (`warning` icon + copper tags) cards detailing algorithmic failure modes (Extreme re-compression, satire/parody ambiguity, ultra-brief audio segments < 2.5s).

### 5.10 Bottom Navigation Bar
- Dual-tab bottom bar with `Verify` (magnifying checkmark / `verified` icon) and `Overview` (pipeline tree / `account_tree` icon).
- Active state indicated with Sage Green icon, label, and top border highlight (`border-t-2 border-primary`).

---

## 6. Retrieved Screen Catalog

| Screen Resource Name | Screen ID | Title | Dimensions | Downloadable HTML Mime |
| :--- | :--- | :--- | :--- | :--- |
| `projects/8272803068634163835/screens/77e904a16062487f8554fe26659efda6` | `77e904a16062487f8554fe26659efda6` | **Home / Verify** | 780 × 3512 | `text/html` |
| `projects/8272803068634163835/screens/f5cc6ea77cc74ea59b9f725210b2d0c9` | `f5cc6ea77cc74ea59b9f725210b2d0c9` | **System Overview** | 780 × 6768 | `text/html` |

---

## 7. Media & Asset References

- **TruthLensAI Desk Logo**:
  `https://lh3.googleusercontent.com/aida/AEtjO1VInzInkjrGZdkQjoz5P7pr02FDC5B4VhJMzp8_dI2m9vXozDFrTjSmklgP42Sf_xBr2Q6V0AGs6trroxSLLr3HQCUGj4nXGHZcHW2jCRtxZBKDdnYZTk6YPLjlN6q7N8dPXbZM0IZ0NSanR5IvRCU3Y0Zr-7Sgh1JYenr_4wZlJgSGq7wEyxQ-fevvflxcxrXNC_Uo0gWLIOVODgxUQEbbbxkd4KSuDUPEeO2ritH84mRezPSwwuVRiR8`
- **Political Rally Deepfake Sample Thumbnail**:
  `https://lh3.googleusercontent.com/aida-public/AB6AXuADaQV3yuERopUv_4AR2EQ9AE511lB7bVf4aR0Gzl-GkLvWzUcG4hSyfK_tHRepMZl6_LLUljECAhPC-TSadbHkgpQmQOFiGy3NIqzG9ZxJb_6y7atNkrmUpiLXA8ewaxAhgIyjnDqLu1qGWZ4mKwMOjFtzakjBp-wz3nudzxivY86TTvgk6XtJfzH0v2OBlAY93td_rXlmBaCF9WOMnMuypx9mI2gPV0lejz12iwS06Z9-GyQf4hmI`
- **PRNU Sensor Spectrogram Heatmap**:
  `https://lh3.googleusercontent.com/aida-public/AB6AXuC4LuBCFH86DmbwP0-8SCpaz1_51UL4748b0lUFuuR4MsXAgY1jAqePi3lRmoPOc9Tp1IshwpMXO11dw7fErMfbcJKH1JQRVYW_w6s1BqmxNVLXvd8pJ59p9NhYfkTEJQckNwB_7EpMs7huoaoSLej_uAqQNdNNtVVmmrLmw7m1F7a4cU9gwSyfrhBj9iP3D-L6tw3Baz3cGV_uQO6y69DqOX5J-ORWAGyHPI8ZX81WzpwFVSnbVRlP`
- **Microscopic Lapel Seam Detection Preview**:
  `https://lh3.googleusercontent.com/aida-public/AB6AXuBED_36gyre0fIEDXs5g9LsdO4u3HEBmTfudegXjT58ybhX_CAv90H0WMSbHkd7_Q2Mr4AIPj92-kk1vrJ9DQA2UzmGsvGyBzHND9EVbnDxcTWPJai0DXGnG1K5-x9P5_9NXWd0rxyag4mkNQcwG3iVe2vrRARTkaMzYv1io4Qah8BXsZsvEZHIfDJDO5AuGgcsA13t-sD6uQWFLh3B5Cgh9Et81Jcbkdl7yQaUyUseyFwGCV56MIes`
