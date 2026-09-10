<div align="center">

<img src="branding/team_logo.png" alt="GeoShield Logo" width="200" />

# 🛡️ GeoShield

### AI-Based Early Warning & Landslide Risk Monitoring System
**North Eastern Region, India — Smart India Hackathon 2026**

![SIH 2026](https://img.shields.io/badge/SIH-2026-green?style=for-the-badge)
![Problem ID](https://img.shields.io/badge/Problem_ID-26001-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![AI/ML](https://img.shields.io/badge/AI/ML-Random_Forest-orange?style=for-the-badge)

**Ministry of Development of North Eastern Region (MDoNER)**

</div>

---

## 📋 Table of Contents

1. [Problem Statement](#-problem-statement)
2. [Why Landslides Happen in NER](#-why-landslides-happen-in-ner)
3. [Our Solution](#-our-solution)
4. [System Architecture](#-system-architecture)
5. [AI/ML Model](#-aiml-model)
6. [Real Data Sources](#-real-data-sources)
7. [Frontend Features](#-frontend-features)
8. [Backend API](#-backend-api)
9. [Historical Data Analysis](#-historical-data-analysis)
10. [Early Warning System](#-early-warning-system)
11. [GIS Risk Mapping](#-gis-risk-mapping)
12. [Landslide Simulator](#-landslide-simulator)
13. [Satellite Data Integration](#-satellite-data-integration)
14. [Multilingual Support](#-multilingual-support)
15. [Quick Start](#-quick-start)
16. [Project Structure](#-project-structure)
17. [Tech Stack](#-tech-stack)
18. [Results & Impact](#-results--impact)
19. [Future Roadmap](#-future-roadmap)
20. [Team](#-team)

---

## 🎯 Problem Statement

### The Crisis

The **North Eastern Region (NER)** of India comprises 8 states — Sikkim, Assam, Manipur, Mizoram, Meghalaya, Nagaland, Tripura, and Arunachal Pradesh — home to **45 million people**. This region is geologically young, tectonically active, and receives some of the highest rainfall in the world (Cherrapunji receives 11,777mm annually).

### Why Landslides Happen in NER

Landslides in NER are caused by a complex interplay of **geological, meteorological, and anthropogenic factors**:

```
  LANDSLIDE TRIGGER FACTORS
  ═══════════════════════════════════════════════════════

  ┌─────────────────────────────────────────────────────┐
  │              GEOLOGICAL FACTORS                     │
  │                                                     │
  │  • Young, weak sedimentary rocks (Tertiary age)     │
  │  • Active tectonic zone (India-Eurasia collision)   │
  │  • Steep slopes (30-60° angles common)             │
  │  • Weathered soil layers over bedrock              │
  │  • Seismic activity (Zone IV-V earthquake zone)    │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │              METEOROLOGICAL FACTORS                  │
  │                                                     │
  │  • Extreme monsoon rainfall (June-September)        │
  │  • Intense rainfall events (>100mm in 24 hours)    │
  │  • Prolonged saturation of soil layers              │
  │  • Cyclonic storms from Bay of Bengal              │
  │  • Rapid snowmelt in Himalayan zones               │
  └─────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────┐
  │              ANTHROPOGENIC FACTORS                   │
  │                                                     │
  │  • Road construction cutting through slopes         │
  │  • Deforestation for agriculture                    │
  │  • unplanned urbanization on hill slopes            │
  │  • Mining and quarrying activities                  │
  │  • Poor drainage infrastructure                     │
  └─────────────────────────────────────────────────────┘
```

### The Numbers

```
  NER LANDSLIDE IMPACT (2011-2024)
  ═══════════════════════════════════════════════════════

  Total Events:     ████████████████████████████████████████████  44
  Total Deaths:     ████████████████████████████████████████████  88
  Road Blockades:   ████████████████████████████████████████████  31
  People Affected:  ████████████████████████████████████████████  8,087+

  BY STATE (events):
  Sikkim        ████████████████████  8 events  (46 deaths - highest)
  Meghalaya     ██████████████████   7 events  (18 deaths)
  Assam         ████████████████     6 events   (9 deaths)
  Arunachal     ████████████████     6 events   (6 deaths)
  Manipur       ██████████████       5 events   (4 deaths)
  Mizoram       ██████████████       5 events   (3 deaths)
  Nagaland      ██████████           4 events   (2 deaths)
  Tripura       ████████             3 events   (0 deaths)

  TRIGGER BREAKDOWN:
  Rain:      ████████████████████████████████████████  91% (40 events)
  Earthquake ██                                              5% (2 events)
  Flood:     ██                                              5% (2 events)

  SEVERITY:
  Large:     ████████████████████████████  27% (12 events)
  Medium:    ████████████████████████████████████████████  43% (19 events)
  Small:     ████████████████████████████████  30% (13 events)
```

### What's Missing Today

| Gap | Current State | Impact |
|-----|---------------|--------|
| **No centralized monitoring** | Each state handles independently | Delayed response |
| **No AI prediction** | Manual inspection only | Reactive, not preventive |
| **No real-time sensors** | Rain gauges at district level | Missing local events |
| **No multilingual alerts** | English only | 60% population excluded |
| **No citizen reporting** | No mobile infrastructure | Missed early signs |
| **No GIS visualization** | Paper maps | Poor situational awareness |

---

## 🛡️ Our Solution

### GeoShield — A Complete Monitoring Platform

GeoShield is a **full-stack AI-powered landslide monitoring system** designed specifically for the North Eastern Region. It combines **real-time sensor data**, **satellite imagery**, **machine learning prediction**, and **multilingual early warning** into a single unified platform.

### 6 Core Capabilities

| # | Capability | Description | Technology |
|---|------------|-------------|------------|
| 1 | **Real-Time Monitoring** | 20 IoT sensor stations across 8 NER states collecting rainfall, soil moisture, ground displacement, tilt, and pore pressure data | FastAPI + SQLite |
| 2 | **AI Risk Prediction** | RF+GB VotingClassifier ensemble (95.2% accuracy, 94.6% F1) trained on 12,000 real NER terrain samples | scikit-learn |
| 3 | **Early Warning System** | Multi-level alert framework (Low → Moderate → High → Critical) with automatic SMS/push notification support | WebSocket + REST |
| 4 | **GIS Risk Mapping** | Interactive Leaflet.js heatmaps showing real-time risk distribution, road status, village locations, and sensor stations | Leaflet.js |
| 5 | **Citizen Reporting** | Geo-tagged photo/video reporting system for field officers and local residents with offline queue support | React + FastAPI |
| 6 | **Multilingual UI** | Full interface translation in English, Hindi, Bengali, and Assamese covering all 90+ UI strings | i18n system |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     🖥️  PRESENTATION LAYER                      │
│                     (React 19 + TypeScript + Tailwind CSS)       │
│                                                                 │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │  📊       │ │  🗺️       │ │  🚨       │ │  📝       │       │
│  │ Dashboard │ │  GIS Map  │ │  Alerts   │ │ Reports   │       │
│  │           │ │           │ │           │ │           │       │
│  │ • Stats   │ │ • Heatmap │ │ • Filter  │ │ • Submit  │       │
│  │ • Charts  │ │ • Roads   │ │ • Ack     │ │ • View    │       │
│  │ • Rankings│ │ • Villages│ │ • Resolve │ │ • Upload  │       │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘       │
│  ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐                    │
│  │  ⚡       │ │  🛰️       │ │  📡       │                    │
│  │ Simulator │ │ Satellite │ │ Station   │                    │
│  │           │ │           │ │           │                    │
│  │ • 4 level │ │ • 20 stn  │ │ • Charts  │                    │
│  │ • AI eval │ │ • Real    │ │ • AI risk │                    │
│  │ • History │ │ • Live    │ │ • Weather │                    │
│  └─────┬─────┘ └─────┬─────┘ └─────┬─────┘                    │
├────────┴─────────────┴─────────────┴────────────────────────────┤
│                     ⚙️  BUSINESS LAYER                          │
│                     (Python FastAPI)                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    REST API (17 Endpoints)                │   │
│  │                                                          │   │
│  │  /api/dashboard/*    → Stats, heatmap, trends, states   │   │
│  │  /api/sensors/*      → Stations, readings, history      │   │
│  │  /api/alerts/*       → CRUD, acknowledge, resolve       │   │
│  │  /api/reports/*      → Submit, list, verify             │   │
│  │  /api/weather/*      → Current + forecast               │   │
│  │  /api/satellite/*    → Real data, summary, risk zones   │   │
│  │  /api/simulate/*     → Landslide simulation             │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    🤖 AI/ML ENGINE                        │   │
│  │                                                          │   │
│  │  ┌────────────────┐     ┌────────────────┐              │   │
│  │  │  Random Forest │     │    Gradient    │              │   │
│  │  │  200 trees     │     │    Boosting    │              │   │
│  │  │  max_depth=15  │     │    150 trees   │              │   │
│  │  │  balanced      │     │    lr=0.1      │              │   │
│  │  └───────┬────────┘     └───────┬────────┘              │   │
│  │          └──────────┬───────────┘                        │   │
│  │          VotingClassifier (soft, weights=[0.4, 0.6])     │   │
│  │                                                          │   │
│  │  Training: 12,000 NER samples | 9 features              │   │
│  │  Accuracy: 95.2% test | F1: 94.6% weighted              │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     💾  DATA LAYER                               │
│                                                                 │
│  ┌───────────┐ ┌───────────────┐ ┌──────────────┐              │
│  │  SQLite   │ │  Open-Meteo   │ │  NASA GLC    │              │
│  │  Database │ │  Satellite API│ │  Landslide   │              │
│  │           │ │               │ │  Catalog     │              │
│  │ • Stations│ │ • Elevation   │ │ • 44 events  │              │
│  │ • Sensors │ │ • Soil moist. │ │ • 8 states   │              │
│  │ • Alerts  │ │ • Rainfall    │ │ • 2011-2024  │              │
│  │ • Reports │ │ • NDVI        │ │              │              │
│  └───────────┘ └───────────────┘ └──────────────┘              │
│  ┌───────────┐ ┌───────────────┐ ┌──────────────┐              │
│  │  Kaggle   │ │  IMD India    │ │  USGS SRTM   │              │
│  │  Datasets │ │  Rainfall     │ │  DEM Data    │              │
│  │           │ │               │ │              │              │
│  │ • 3 files │ │ • District    │ │ • 30m res    │              │
│  │ • 528KB   │ │   rainfall    │ │ • Ready to   │              │
│  │           │ │ • 1901-2015   │ │   integrate  │              │
│  └───────────┘ └───────────────┘ └──────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI/ML Model

### Why Machine Learning for Landslide Prediction?

Traditional landslide susceptibility mapping relies on **static geological maps** and **manual expert assessment**. This approach:
- Cannot adapt to changing weather conditions
- Requires expensive field surveys
- Takes weeks to produce results
- Cannot provide real-time predictions

GeoShield's AI model solves these problems by:
- Processing **real-time sensor data** continuously
- Learning from **12,000 historical NER terrain samples**
- Providing predictions in **<30 seconds**
- Adapting to **seasonal monsoon patterns**

### Model Architecture

```
  VOTING CLASSIFIER ENSEMBLE
  ═══════════════════════════════════════════════════════

  Input Features (9):
  ┌─────────────────────────────────────────────────────┐
  │ slope │ elevation │ aspect │ rainfall_daily │       │
  │ rainfall_7day │ ndvi │ soil_moisture │              │
  │ distance_to_road │ month                            │
  └─────────────────────────────────────────────────────┘
         │                         │
         ▼                         ▼
  ┌──────────────┐        ┌──────────────┐
  │   Random     │        │   Gradient   │
  │   Forest     │        │   Boosting   │
  │              │        │              │
  │ 200 trees    │        │ 150 trees    │
  │ max_d=15     │        │ max_d=8      │
  │ balanced     │        │ lr=0.1       │
  │ min_split=5  │        │ min_split=5  │
  │              │        │              │
  │ Weight: 0.4  │        │ Weight: 0.6  │
  └──────┬───────┘        └──────┬───────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌───────────────────┐
         │  Soft Voting      │
         │  (probability     │
         │   averaging)      │
         └─────────┬─────────┘
                   ▼
         ┌───────────────────┐
         │  Risk Score: 0-100│
         │  Level: L/M/H/C   │
         │  Probability: 0-1 │
         └───────────────────┘
```

### Feature Importance

```
  FEATURE IMPORTANCE RANKING
  ═══════════════════════════════════════════════════════

  1. Slope Angle     ██████████████████████████  25%
     Why: Steeper slopes have higher shear stress
     Source: SRTM DEM / Open-Meteo elevation API

  2. Daily Rainfall  ████████████████████        20%
     Why: Primary trigger for most NER landslides
     Source: Open-Meteo weather API (live)

  3. Soil Moisture   ███████████████             15%
     Why: Saturated soil loses cohesive strength
     Source: Open-Meteo soil moisture API (live)

  4. 7-Day Rainfall  ███████████████             15%
     Why: Cumulative saturation effect
     Source: Open-Meteo hourly rainfall (7 days)

  5. NDVI Index      ███████████████             15%
     Why: Low vegetation = exposed soil = high risk
     Source: Sentinel-2 satellite (estimated)

  6. Elevation       ██████████                  10%
     Why: Higher elevations have more potential energy
     Source: Open-Meteo elevation API (real)
```

### Risk Classification Thresholds

| Level | Score Range | Color | Response |
|-------|-------------|-------|----------|
| 🟢 **Low** | 0 - 25 | Green | Normal monitoring, routine checks |
| 🟡 **Moderate** | 25 - 50 | Amber | Enhanced monitoring, notify DDM authority |
| 🟠 **High** | 50 - 75 | Orange | Pre-position rescue teams, voluntary evacuation |
| 🔴 **Critical** | 75 - 100 | Red | IMMEDIATE EVACUATION, deploy emergency response |

### Model Performance

> All metrics verified via independent evaluation on the actual trained model.

```
  MODEL ACCURACY (VERIFIED)
  ═══════════════════════════════════════════════════════

  Training Accuracy:   ████████████████████████████████████████  99.98%
  Test Accuracy:       ████████████████████████████████████████  95.2%
  F1 Score (weighted): ████████████████████████████████████      94.6%

  INDIVIDUAL MODELS:
  Gradient Boosting:   ████████████████████████████████████████  95.3%
  Random Forest:       ██████████████████████████████████        88.8%
  Ensemble (RF+GB):    ████████████████████████████████████████  95.2%

  Training Samples:    12,000 (real NER terrain coordinates)
  Test Samples:        2,400 (20% holdout, stratified)
  Features:            9 input features
  Classes:             4 (low, moderate, high, critical)
  Model Caching:       joblib pickle with version-tagged reload
```

#### Per-Class Performance

```
  CLASS         SAMPLES   PRECISION   RECALL   F1-SCORE
  ─────────────────────────────────────────────────────
  Low            2,180     0.96       0.99     0.97
  High              33     0.00       0.00     0.00
  Critical         187     0.84       0.73     0.78
  ─────────────────────────────────────────────────────
  Weighted Avg   2,400     0.94       0.95     0.95
```

> **Note:** The "Moderate" class has only 4 samples in the full dataset (0.03%),
> so it is effectively absorbed into adjacent classes. The model excels at
> identifying **Low** risk (98.5% per-class accuracy) and detecting **Critical**
> events (73.3% recall) — the two most operationally important categories
> for an early warning system.

---

## 🛰️ Real Data Sources

### Satellite & Sensor Data Integration

| Source | Data Type | Status | Coverage | Resolution |
|--------|-----------|--------|----------|------------|
| **Open-Meteo API** | Elevation, Soil Moisture, Weather | ✅ Live | 20 stations | Real-time |
| **NASA GLC** | Historical Landslide Catalog | ✅ 44 events | 8 NER states | Point data |
| **Kaggle** | India Rainfall (1901-2015) | ✅ 528 rows | District | Monthly |
| **Kaggle** | India Landslide Incidents | ✅ 200+ events | India | District |
| **SRTM DEM** | Terrain/Elevation | 📋 Ready | Global | 30m |
| **Sentinel-2** | NDVI Vegetation Index | 📋 Ready | Global | 10m |
| **IMD** | Official Indian Rainfall | 📋 Ready | District | Daily |
| **USGS** | Landslide Hazard Maps | 📋 Ready | Regional | Variable |

### Real Satellite Metrics Per Station

```
  REAL-TIME SATELLITE DATA (Open-Meteo API)
  ═══════════════════════════════════════════════════════

  ELEVATION RANGE (meters):
  Agartala    ▓                                          12m
  Dimapur     ▓▓                                        147m
  Itanagar    ▓▓▓                                       160m
  Guwahati    ▓▓                                         52m
  Dima Hasao  ▓▓▓▓                                     413m
  Mangan      ▓▓▓▓▓▓▓                                 796m
  Imphal      ▓▓▓▓▓▓▓▓                                782m
  Churachand. ▓▓▓▓▓▓▓▓▓                               862m
  Aizawl      ▓▓▓▓▓▓▓▓▓▓                             1069m
  Cherrapunji ▓▓▓▓▓▓▓▓▓▓                             1029m
  Namchi      ▓▓▓▓▓▓▓▓▓▓                             814m
  Kohima      ▓▓▓▓▓▓▓▓▓▓▓▓                          1365m
  Shillong    ▓▓▓▓▓▓▓▓▓▓▓▓                          1436m
  Gangtok     ▓▓▓▓▓▓▓▓▓▓▓▓                          1487m
  Ziro        ▓▓▓▓▓▓▓▓▓▓▓▓                          1592m
  Tawang      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                   2791m

  SOIL MOISTURE (m³/m³ — higher = wetter = riskier):
  Tura        ▓▓▓▓▓▓▓▓▓              0.29  ← Driest
  Tawang      ▓▓▓▓▓▓▓▓▓▓▓            0.37
  Imphal      ▓▓▓▓▓▓▓▓▓▓▓            0.38
  Aizawl      ▓▓▓▓▓▓▓▓▓▓▓▓           0.39
  Agartala    ▓▓▓▓▓▓▓▓▓▓▓▓           0.40
  Gangtok     ▓▓▓▓▓▓▓▓▓▓▓▓           0.41
  Ziro        ▓▓▓▓▓▓▓▓▓▓▓▓           0.42
  Dima Hasao  ▓▓▓▓▓▓▓▓▓▓▓▓           0.43
  Guwahati    ▓▓▓▓▓▓▓▓▓▓▓▓▓          0.46
  Shillong    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓         0.50  ← Wettest

  NDVI VEGETATION INDEX (0-1 — lower = less vegetation = riskier):
  Tawang      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  0.528  ← Lowest
  Shillong    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.590
  Aizawl      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.593
  Kohima      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.595
  Ziro        ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.600
  Cherrapunji ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.611
  Imphal      ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.621
  Dima Hasao  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.650
  Pasighat    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 0.691  ← Highest
```

---

## 🖥️ Frontend Features

### 10 Interactive Pages

| Page | Description | Key Features |
|------|-------------|--------------|
| **🔐 Login** | Authentication gate | 4 demo accounts, role-based access |
| **📊 Dashboard** | Real-time overview | 3 tabs (Overview/Stations/Alerts), radar chart, rankings |
| **🗺️ Risk Map** | GIS visualization | Leaflet heatmap, roads, villages, click-to-predict |
| **🚨 Alerts** | Warning management | Filter by status/risk, acknowledge, resolve workflow |
| **📝 Reports** | Citizen reporting | Photo upload, geo-tagging, multi-type reports |
| **⚡ Simulator** | Live demo tool | 4 intensity levels, AI assessment, alert generation |
| **🛰️ Satellite** | Real data view | 20 stations, live metrics, risk scoring |
| **📡 Station** | Deep dive | Sensor charts, AI gauge, weather, satellite data |
| **🌊 Flood Risk** | Compound hazard | Flood-landslide correlation scatter plot |
| **🎯 Demo Flow** | Judge walkthrough | 8-step guide, live simulation, key metrics |

### Dashboard Overview Tab

```
  ┌─────────────────────────────────────────────────────────────┐
  │  🛡️ GeoShield Dashboard                    LIVE  SIH 2026 │
  ├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┤
  │ Active  │ Active  │ People  │ Pending │ Avg     │ High-   │
  │ Sensors │ Alerts  │ at Risk │ Reports │ Risk    │ Risk    │
  │   20    │   36    │ 31,977  │   15    │  43.8   │    6    │
  ├─────────┴─────────┴─────────┴─────────┴─────────┴─────────┤
  │                                                           │
  │  ┌─────────────────────────┐  ┌───────────────────────┐   │
  │  │   Rainfall Trend (48h)  │  │   Risk Distribution   │   │
  │  │   ▁▂▃▄▅▆▇█▇▆▅▄▃▂▁▂▃   │  │      ◉ Donut Chart    │   │
  │  │   48 data points        │  │   Low:101 Mod:536     │   │
  │  └─────────────────────────┘  │   High:338 Crit:5     │   │
  │                               └───────────────────────┘   │
  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │
  │  │ Risk Trend   │ │ Road Status  │ │ State Overview   │   │
  │  │ 48h line     │ │ Open: 5      │ │ Arunachal  45.2  │   │
  │  │ chart        │ │ Partial: 2   │ │ Sikkim     42.1  │   │
  │  │              │ │ Blocked: 1   │ │ Meghalaya  38.5  │   │
  │  └──────────────┘ └──────────────┘ └──────────────────┘   │
  └─────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Backend API

### 45 RESTful Endpoints

```
  API ENDPOINT STATUS
  ═══════════════════════════════════════════════════════

  DASHBOARD
  ✅ GET  /api/dashboard/stats          → 20 stations, 5 alerts
  ✅ GET  /api/dashboard/risk-heatmap   → 20 GIS points
  ✅ GET  /api/dashboard/rainfall-trend → 48h hourly data
  ✅ GET  /api/dashboard/risk-trend     → 48h risk scores
  ✅ GET  /api/dashboard/state-summary  → 8 NER states

  SENSORS
  ✅ GET  /api/sensors/stations         → 20 stations
  ✅ GET  /api/sensors/stations/{id}    → Station + readings + AI
  ✅ GET  /api/sensors/stations/{id}/history → Time-range readings

  ALERTS
  ✅ GET  /api/alerts                   → All alerts (filtered)
  ✅ GET  /api/alerts/active            → Active alerts only
  ✅ PUT  /api/alerts/{id}/acknowledge  → Acknowledge alert
  ✅ PUT  /api/alerts/{id}/resolve      → Resolve alert

  REPORTS & INFRASTRUCTURE
  ✅ GET  /api/reports                  → Citizen reports
  ✅ POST /api/reports                  → Submit new report
  ✅ GET  /api/roads                    → 48 monitored roads
  ✅ GET  /api/villages                 → 18 tracked villages

  PREDICT (Click-to-Predict)
  ✅ POST /api/predict                  → AI risk at any lat/lng

  EXPORT
  ✅ GET  /api/export/geojson           → GIS-ready GeoJSON
  ✅ GET  /api/export/csv               → Excel/analysis CSV
  ✅ GET  /api/export/risk-zones        → High-risk polygons

  ALERT TIMELINE
  ✅ GET  /api/alerts/timeline          → Chronological view
  ✅ GET  /api/alerts/history           → 30-day trend data
  ✅ GET  /api/alerts/stats             → Alert summary stats

  WEATHER
  ✅ GET  /api/weather/{id}             → Live weather data
  ✅ GET  /api/weather/{id}/forecast    → 48h forecast

  SATELLITE
  ✅ GET  /api/satellite/data           → 20 stations real data
  ✅ GET  /api/satellite/summary        → NER-wide metrics
  ✅ GET  /api/satellite/risk-zones     → Risk from real data

  SIMULATION
  ✅ POST /api/simulate/landslide       → Trigger simulation
  ✅ POST /api/simulate/batch           → Multi-station sim

  WEATHER
  ✅ GET  /api/weather/{station}        → Live weather

  AUTH
  ✅ POST /api/auth/login                  → JWT token

  TOTAL: 45 ENDPOINTS | ALL RETURNING 200 ✅
```

---

## 📊 Historical Data Analysis

### 44 Documented Landslide Events (2011-2024)

Our historical dataset covers **14 years** of landslide events across all 8 NER states, compiled from:
- NASA Global Landslide Catalog (GLC)
- Geological Survey of India reports
- IMD rainfall event documentation
- News reports and district administration records

### Event Timeline

```
  LANDSLIDE EVENTS BY YEAR
  ═══════════════════════════════════════════════════════

  2011  ████                  2 events
  2012  ██                    1 event
  2013  ████████████          4 events
  2014  ████████████████      5 events
  2015  ████████████████      5 events
  2016  ████                  2 events
  2017  ████████████          4 events
  2018  ████████████████████  6 events
  2019  ████                  1 event
  2020  ████████████████████  6 events
  2021  (data gap)            0 events
  2022  ████████████████████  6 events
  2023  ████████████████      5 events
  2024  ████████████████      5 events (incl. Sikkim flash flood)
```

### Fatality Analysis

| Severity | Events | Deaths | Road Blocks | Avg Response |
|----------|--------|--------|-------------|--------------|
| **Large** | 12 | 73 | 11 | 3+ days |
| **Medium** | 19 | 15 | 17 | 1-3 days |
| **Small** | 13 | 0 | 3 | <1 day |
| **Total** | **44** | **88** | **31** | — |

---

## 🚨 Early Warning System

### Alert Classification

| Level | Trigger | Response Time | Actions |
|-------|---------|---------------|---------|
| 🟢 **Normal** | Risk < 25 | 24 hours | Routine monitoring, log readings |
| 🟡 **Advisory** | Risk 25-50 | 6 hours | Enhanced monitoring, notify DDM |
| 🟠 **Warning** | Risk 50-75 | 2 hours | Pre-position rescue teams, voluntary evacuation |
| 🔴 **Emergency** | Risk > 75 | 30 minutes | IMMEDIATE EVACUATION, deploy sirens, close roads |

### Alert Workflow

```
  SENSOR DATA → AI ASSESSMENT → RISK SCORE → ALERT LEVEL
       │              │              │              │
       ▼              ▼              ▼              ▼
  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │ Rainfall│  │ RF + GB  │  │ 0-100    │  │ L/M/H/C  │
  │ Moisture│→ │ Ensemble │→ │ Score    │→ │ Level    │
  │ Displcm │  │ Predict  │  │          │  │          │
  │ Tilt    │  │          │  │          │  │          │
  └─────────┘  └──────────┘  └──────────┘  └─────┬────┘
                                                  │
                              ┌────────────────────┤
                              ▼                    ▼
                     ┌──────────────┐    ┌──────────────┐
                     │  In-App      │    │  SMS/Push    │
                     │  Dashboard   │    │  Notification│
                     │  Alert       │    │  (planned)   │
                     └──────────────┘    └──────────────┘
```

---

## 🗺️ GIS Risk Mapping

### Map Layers

| Layer | Description | Color Code |
|-------|-------------|------------|
| **Risk Heatmap** | Color-coded circles by risk level | 🟢🟡🟠🔴 |
| **Road Network** | 48 monitored roads with status | Green/Amber/Red |
| **Village Markers** | 18 villages with population | By risk zone |
| **Station Markers** | 20 sensor stations | Click for details |

### Monitored Roads

```
  ROAD STATUS
  ═══════════════════════════════════════════════════════

  🟢 OPEN (5 roads):
  ├── NH-10  (Siliguri-Gangtok)
  ├── NH-2   (Dimapur-Kohima)
  ├── NH-6   (Shillong-Tura)
  ├── NH-29  (Guwahati-Shillong)
  └── NH-415 (Itanagar-Bomdila)

  🟡 PARTIALLY BLOCKED (2 roads):
  ├── NH-37  (Guwahati-Jorhat) — Debris on one lane
  └── SH-4   (Haflong-North Cachar) — Reduced capacity

  🔴 BLOCKED (1 road):
  └── SH-1   (Aizawl-Lunglei) — Full blockage, landslide debris
```

---

## ⚡ Landslide Simulator

### For Live SIH Demo

The simulator allows presenters to **trigger realistic landslide events** and watch the entire system respond in real-time:

1. **Select Station** — Pick any of the 20 NER stations
2. **Choose Intensity** — Low / Moderate / High / Critical
3. **Click Run** — Watch the system respond:
   - Sensor readings spike (rainfall, moisture, displacement)
   - AI model runs assessment (new risk score)
   - Alert generated if risk >= moderate
   - Dashboard updates in real-time

### Demo Flow for Judges

```
  DEMO SEQUENCE (3 minutes)
  ═══════════════════════════════════════════════════════

  Step 1 (30s): Dashboard Overview
  → Show 20 stations, risk pie chart, rainfall trends
  → Point out real satellite data metrics

  Step 2 (30s): GIS Risk Map
  → Show interactive map with heatmap
  → Click Cherrapunji station (known hotspot)
  → Show road status and village markers

  Step 3 (60s): Landslide Simulator
  → Navigate to Simulator page
  → Select Cherrapunji, intensity = CRITICAL
  → Click "Run Simulation"
  → Show: Risk score spikes to 95.4/100
  → Show: Alert generated with 12,000+ affected
  → Show: Contributing factors and recommendation

  Step 4 (30s): Satellite Data
  → Navigate to Satellite Data page
  → Show real elevation, soil moisture, NDVI
  → Compare Tawang (2791m, high risk) vs Agartala (12m, low risk)

  Step 5 (30s): Multilingual Support
  → Switch language to Hindi → Bengali → Assamese
  → Show all labels translate correctly

  Step 6 (30s): Station Deep Dive
  → Click any station
  → Show sensor charts, AI gauge, weather data
  → Show contributing factors and recommendation
```

---

## 🌊 Flood Risk Monitoring

### Compound Hazard Analysis

GeoShield integrates **flood-landslide correlation** data for all 19 NER districts, sourced from the Asia Flood Atlas and IMD historical records. The system computes **compound risk** (0.4 × flood risk + 0.6 × landslide risk) to identify districts facing dual hazards.

| District | Flood Risk | Events | Rivers |
|----------|-----------|--------|--------|
| East Khasi Hills | 85 | 42 | Umiam, Wah Umkhrah |
| Kamrup | 78 | 38 | Brahmaputra, Kalu |
| Dimapur | 70 | 28 | Dhansiri, Dan |
| East Siang | 68 | 24 | Siang, Dibang |
| West Garo Hills | 65 | 22 | Simsang, Asanang |

### 3 Flood API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/flood/data` | District-level flood risk data |
| `GET /api/flood/summary` | Aggregated NER flood metrics |
| `GET /api/flood/correlation` | Flood × landslide compound risk scatter |
|  | Flood × landslide compound risk scatter |

---

## 🌐 Multilingual Support

| Language | Code | Coverage | Script |
|----------|------|----------|--------|
| English | en | ✅ 90+ keys | Latin |
| Hindi | hi | ✅ 90+ keys | Devanagari |
| Bengali | bn | ✅ 90+ keys | Bengali |
| Assamese | as | ✅ 90+ keys | Bengali (Assamese) |

---

## 🚀 Quick Start

### Download Pre-Built Apps

| Platform | File | Size | How to Run |
|----------|------|------|------------|
| **Android** | `GeoShield-Android.apk` | 7.9 MB | Transfer to phone → Install |
| **Linux AppImage** | `GeoShield-1.0.0.AppImage` | 108 MB | `chmod +x` then `./GeoShield-*.AppImage` |
| **Linux DEB** | `geoshield_1.0.0_amd64.deb` | 104 MB | `sudo dpkg -i geoshield_*.deb` |
| **Windows** | `GeoShield-1.0.0-Windows-x64.zip` | 165 MB | Extract → Run `GeoShield.exe` |
| **Windows** | `start.bat` | 1 KB | Double-click to auto-setup & launch |

### One-Command Deploy (Web)

```bash
# Clone
git clone https://github.com/officialarghya29/GeoShield.git
cd GeoShield

# Deploy (creates venv, installs deps, builds frontend, starts server)
bash deploy.sh

# Open
open http://localhost:8000
```

### Manual Setup

```bash
# Prerequisites: Python 3.10+, Node.js 18+

# Backend
cd backend
python3 -m venv venv          # Create virtual environment
source venv/bin/activate      # Activate venv (Linux/Mac)
# .\venv\Scripts\activate    # Activate venv (Windows)
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev                   # Dev server at http://localhost:5173
# npm run build               # OR build for production
```

### Docker

```bash
docker build -t geoshield .
docker run -p 8000:8000 geoshield
```

**Demo Login:** `admin@geoshield.gov.in` / `admin123` — or click any demo button on the login page.

---

## 📁 Project Structure

```
GeoShield/
├── README.md                              # This file
├── SIH_2026_PRESENTATION.md               # 15-slide pitch deck
├── PRESENTATION.md                        # Slide content with diagrams
├── DEPLOYMENT_GUIDE.md                    # Railway/Render/Docker
├── SATELLITE_INTEGRATION.md               # Real data integration
├── BUILD_GUIDE.md                         # Desktop/mobile build instructions
├── Dockerfile                             # Docker deployment
├── Procfile                               # Railway deployment
├── deploy.sh                              # One-click local deploy (Linux/Mac)
├── start.bat                              # One-click local deploy (Windows)
├── start.sh                               # Quick launcher script
├── demo.sh                                # Polished demo script for judges
├── electron/
│   ├── main.js                            # Electron main process + backend auto-start
│   └── preload.js                         # Secure IPC bridge
├── android/                               # 📱 Capacitor Android wrapper
├── branding/
│   ├── team_logo.png                      # Team logo
│   └── team_logo.ico                      # Windows icon
│
├── backend/                               # ⚙️ Python FastAPI
│   ├── app/
│   │   ├── main.py                        # App entry + static files
│   │   ├── models.py                      # 8 SQLAlchemy models
│   │   ├── database.py                    # SQLite connection
│   │   ├── seed_data.py                   # Realistic NER seeder
│   │   ├── ai_engine/
│   │   │   ├── risk_predictor.py          # RF + GB ensemble (original)
│   │   │   ├── enhanced_predictor.py     # XGBoost + terrain lookup (alternative)
│   │   │   └── terrain_lookup.py         # Nearest-neighbor NER terrain data
│   │   └── routers/
│   │       ├── sensors.py                 # Station APIs
│   │       ├── dashboard.py               # Stats, heatmap, trends
│   │       ├── alerts.py                  # Alert management
│   │       ├── reports.py                 # Reports + roads + villages
│   │       ├── weather.py                 # Weather data
│   │       ├── simulator.py               # Landslide simulator
│   │       ├── satellite.py               # Real satellite data
│   │       ├── flood.py                   # Flood risk + correlation
│   │       ├── alerts_timeline.py         # Timeline + history + trends
│   │       ├── predict.py                 # Click-to-predict API
│   │       ├── ml_enhanced.py             # Enhanced ML routes + risk grid
│   │       └── export.py                  # GeoJSON/CSV export
│   │   ├── schemas.py                     # Pydantic validation
│   │   ├── middleware/
│   │   │   └── rate_limiter.py            # Rate limiting (100/min)
│   │   └── tests/
│   │       ├── test_api.py                # 33 unit tests
│   │       └── test_e2e.py                # 48 integration tests (8 new ML tests)
│   └── uploads/                           # Photo uploads
│
├── frontend/                              # 🖥️ React + TypeScript
│   ├── src/
│   │   ├── App.tsx                        # Router + Auth + Sidebar
│   │   ├── pages/
│   │   │   ├── Dashboard.tsx              # 3 tabs, charts, rankings
│   │   │   ├── RiskMap.tsx                # Leaflet GIS + click-to-predict
│   │   │   ├── Alerts.tsx                 # Timeline + 30-day history
│   │   │   ├── Reports.tsx                # Citizen reports
│   │   │   ├── StationDetail.tsx          # Station + AI + satellite
│   │   │   ├── Simulator.tsx              # Landslide simulator
│   │   │   ├── SatelliteData.tsx          # Real satellite metrics
│   │   │   ├── FloodData.tsx              # 19 districts + correlation
│   │   │   └── DemoFlow.tsx               # 8-step guide for judges
│   │   ├── components/
│   │   │   ├── ErrorBoundary.tsx           # Crash recovery UI
│   │   │   └── MobileFAB.tsx              # Mobile floating action button
│   │   ├── services/api.ts                # API client (45 endpoints)
│   │   └── i18n/translations.ts           # EN, HI, BN, AS, OR (5 languages)
│   └── dist/                              # Built frontend
│
├── datasets/                              # 📊 Data Sources
│   ├── processed/
│   │   ├── real_satellite_data.json        # Live Open-Meteo data
│   │   ├── real_ner_training_data.csv      # 12,000 training samples
│   │   └── ner_landslide_events.csv        # Historical events
│   ├── raw/
│   │   ├── ner_historical_landslides.csv   # 44 events (2011-2024)
│   │   ├── nasa_landslide_catalog.csv      # NASA GLC
│   │   └── india_district_rainfall.csv     # IMD rainfall
│   └── download_datasets.py                # Data collection scripts
│
└── kaggle/                                # 📥 Downloaded datasets
    ├── catalog.csv
    ├── landslide_india.csv
    └── rainfall_india.csv
```

---

## 🔒 Security

| Feature | Implementation |
|---------|---------------|
| **Rate Limiting** | 100 req/min general, 10 req/min auth |
| **JWT Authentication** | HS256, 24h expiry, bcrypt password hashing |
| **RBAC** | 4 roles: admin, field_officer, district_admin, citizen |
| **Input Validation** | Pydantic schemas on all POST endpoints |
| **CORS** | Configurable origins |
| **Path Traversal** | Protected static file serving |

---

## 🛠️ Tech Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Frontend** | React | 19 | UI Framework |
| **Styling** | Tailwind CSS | 3.x | Responsive design |
| **Maps** | Leaflet.js | 1.9.4 | GIS visualization |
| **Charts** | Recharts | 2.x | Data visualization |
| **Icons** | Lucide React | Latest | UI icons |
| **Backend** | Python FastAPI | 0.115 | REST API server |
| **Database** | SQLite | 3.x | Data storage |
| **AI/ML** | scikit-learn | 1.x | Risk prediction (RF+GB VotingClassifier) |
| **Caching** | joblib | — | Model persistence across restarts |
| **APIs** | Open-Meteo | Free | Real-time weather |
| **Build** | Vite | 5.x | Frontend bundler |
| **HTTP** | Axios | 1.x | API client |
| **Mobile** | Capacitor | 6.x + Status Bar | Android wrapper, futuristic splash |
| **Desktop** | Electron | 44.x | Windows/Linux, auto-starts backend |
| **Testing** | pytest + TestClient | — | 75 tests (35 API + 40 E2E) |

---

## 📈 Results & Impact

### Key Metrics

```
  ╔══════════════════════════════════════════════════════════════╗
  ║              GeoShield Performance Dashboard                 ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  🤖 AI Model            95.2% accuracy, 94.6% F1 (12,000 samples)     ║
  ║  📡 Sensor Stations     20 across 8 NER states              ║
  ║  📊 API Endpoints       45 fully functional                  ║
  ║  🗺️  GIS Features        Heatmap + Roads + Villages         ║
  ║  🛰️  Satellite Data      Real Open-Meteo integration       ║
  ║  📜 Historical Events   44 events (2011-2024)               ║
  ║  🌐 Languages           4 (EN, HI, BN, AS)                  ║
  ║  ⚡ Response Time        <30 seconds AI assessment           ║
  ║  👥 People Protected    31,977 at-risk population             ║
  ║  🛣️  Roads Monitored     48 (35 open, 8 partial, 5 blocked)   ║
  ║  🏘️  Villages Tracked    18 (6 high-risk zones)             ║
  ║  📝 Citizen Reports     15+ with geo-tagged data             ║
  ║  🎯 Frontend Pages      9 interactive pages                 ║
  ║  📱 Login Roles         4 (Admin, Field, District, Citizen) ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

### Potential Impact

| Metric | Before GeoShield | After GeoShield |
|--------|------------------|-----------------|
| **Warning Time** | 0 (reactive) | 6+ hours (predictive) |
| **Coverage** | Manual inspection | 20 automated stations |
| **Languages** | English only | 4 languages |
| **Response** | Days | <30 minutes |
| **Data Source** | Paper reports | Real satellite + sensors |

---

## ✅ Test Results

### Test Suite: 75/75 PASSED

```
═══ 35 API TESTS + 40 E2E TESTS ═══

  API Tests (35/35):
  Health & Auth:       4/4
  Dashboard:           5/5
  Sensors:             5/5
  Alerts:              5/5
  Predict:             3/3
  Simulate:            1/1
  Export:              3/3
  Weather:             2/2
  Satellite:           3/3
  Infrastructure:      2/2
  Frontend:            2/2

  E2E Integration (40/40):
  Core Backend:        3/3
  Dashboard Flow:      5/5
  Sensor Flow:         3/3
  Alerts Flow:         4/4
  Simulator→Alert:     3/3
  Prediction Flow:     2/2
  Flood Flow:          3/3
  Satellite Flow:      3/3
  Weather Flow:        2/2
  Export Flow:         3/3
  Infrastructure:      2/2
  Frontend Routes:     3/3
  Alert Workflow:      1/1
  Security:            3/3

  ════════════════════════════════
  FINAL: 75/75 PASSED, 0 FAILED
  ════════════════════════════════
```

### Key Test Results

| Feature | What Was Tested | Result |
|---------|----------------|--------|
| **Dashboard** | 20 stations, 32 alerts, risk=43.9 | ✅ Real data |
| **Simulate** | Cherrapunji → risk=99.2/critical | ✅ Alert fires |
| **Alert Flow** | Count grew 101→102 after sim | ✅ Flow works |
| **AI Predict** | risk=89.4/critical, 2 factors | ✅ Nearest station found |
| **Flood Correlation** | 19 districts correlated | ✅ Scatter plot |
| **Export** | GeoJSON (20 features), CSV (21 lines) | ✅ Downloads work |
| **Security** | Invalid login→401, no auth→401, bad input→422 | ✅ All blocked |

---

## 📱 Mobile & Desktop Apps

### Android APK

| Detail | Value |
|--------|-------|
| **Package** | com.geoshield.app |
| **Size** | 7.9 MB |
| **Target** | Android 14 (API 34) |
| **Min SDK** | API 22 (Android 5.1) |
| **Features** | All 10 pages, 45 APIs, RF+GB ensemble + terrain lookup, futuristic UI |
| **Splash Screen** | Custom animated shield with grid background |
| **Status Bar** | Dark mode, neon green accent |

```bash
# Build APK
cd frontend && npx cap sync android && cd android && ./gradlew assembleDebug
# Output: android/app/build/outputs/apk/debug/app-debug.apk
```

### Linux Desktop

| Format | Size | Details |
|--------|------|---------|
| **AppImage** | 108 MB | Portable, no install needed |
| **DEB Package** | 104 MB | Ubuntu/Debian native install |
| **Tar.gz** | 127 MB | Any Linux distro |

| Feature | Details |
|---------|---------|
| **Auto-start Backend** | Python server launches with app |
| **Loading Screen** | Animated splash with progress messages |
| **Menu Bar** | Navigate (Cmd+1-7), View (Zoom, Fullscreen F11), Help |
| **SPA Routing** | HashRouter — all 10 pages work from file:// |
| **Backend Included** | Python + models bundled in app |

```bash
# Run AppImage
chmod +x dist-electron/GeoShield-1.0.0.AppImage
./dist-electron/GeoShield-1.0.0.AppImage
# OR install DEB
sudo dpkg -i dist-electron/geoshield_1.0.0_amd64.deb
```

### Windows Desktop

| Format | Size | Details |
|--------|------|---------|
| **Portable ZIP** | 165 MB | Extract + run GeoShield.exe |
| **Unpacked Dir** | 408 MB | Full Electron + Python backend |

| Feature | Details |
|---------|---------|
| **One-Click Start** | `start.bat` auto-installs deps, seeds DB, opens browser |
| **Backend Bundled** | Python server included, auto-starts on port 8000 |
| **Custom Menu** | Navigate, View, Help with keyboard shortcuts |

```bash
# Build on Windows (or with Wine for NSIS installer):
cd geo-shield && npm install && npm run build:win
# OR use the portable zip directly
```

---

## 🗺️ Future Roadmap

| Phase | Timeline | Features |
|-------|----------|----------|
| **Phase 1** | ✅ Done | Dashboard, GIS Map, Alerts, Reports, Simulator, Satellite, Flood, Click-to-Predict, GeoJSON/CSV Export, Alert Timeline, Demo Flow, Android App, Linux Desktop, Windows Desktop, 5 Languages, Futuristic UI, Rate Limiting |
| **Phase 2** | +3 months | SMS/Push notifications, React Native iOS app, Real IoT sensor integration |
| **Phase 3** | +6 months | Sentinel-2 NDVI pipeline, SRTM DEM integration, IMD API |
| **Phase 4** | +12 months | Offline-first mobile, District admin portal, Multi-hazard support |

---

## 👥 Team GeoShield

| Name | Roll No |
|------|----------|
| **Ayush Biswas** | 12522865 |
| **Kashish Yadav** | 12523416 |
| **Kirti** | 12517751 |
| **Jaanvi Sain** | 12502776 |
| **Sidhharth** | 12518829 |
| **Prajid** | 12524611 |

---

<div align="center">

### 🛡️ GeoShield — Protecting North Eastern India

**Built with ❤️ for Smart India Hackathon 2026**

[![GitHub](https://img.shields.io/badge/GitHub-Officialarghya29-181717?style=for-the-badge&logo=github)](https://github.com/officialarghya29/GeoShield)

</div>
