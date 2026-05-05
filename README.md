# Maritime Conflict Intelligence System (MCIS)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Alpha](https://img.shields.io/badge/Status-Alpha-orange)](pyproject.toml)

> **AIS(선박자동식별장치) 데이터 기반 해상 이상행동 탐지 및 분쟁 조기경보 연구 프레임워크**

MCIS는 대규모 AIS 시계열 데이터에서 선박 행동 변화를 추출하고, 무력 분쟁 발생 시점과의 상관·선행 신호를 분석해 **“해상 교통 변화가 분쟁의 전조가 될 수 있는가?”**를 검증하는 연구/엔지니어링 프로젝트입니다.

---

## ✨ Why MCIS?

- **End-to-end 파이프라인**: 적재 → 정제 → 피처 생성 → 집계 → 분석 → 시각화 → 모델링.
- **분쟁 맥락 반영 피처**: AIS silence, ROT 급변, 항로 엔트로피, 군용/민간 비율 등 행동 기반 신호.
- **재현 가능한 연구 구조**: 설정 파일 기반(`config/settings.yaml`) + 모듈형 CLI + 테스트 코드.
- **논문 지향 설계**: Event Study, ITS, Granger, DiD 등 통계 분석과 예측 모델링을 함께 구성.

---

## 🎯 Core Research Question

> “선박 트래픽 밀도, 선종 구성비, 속도/침로 변동성, AIS silence, ROT 이상치, 항로 엔트로피 같은 해상 행동 신호가 분쟁 발생 전후 유의미하게 변화하는가? 또한 이 신호를 이용해 분쟁 이벤트를 사전에 예측할 수 있는가?”

---

## 🌍 Target Conflict Zones

| Conflict | Onset | Key Zone |
|---|---:|---|
| Russia–Ukraine War | 2022-02-24 | Black Sea, Sea of Azov, Kerch Strait |
| Red Sea / Houthi Crisis | 2023-11-19 | Red Sea, Gulf of Aden, Bab-el-Mandeb |
| Taiwan Strait Tensions | 2022-08-04 | Taiwan Strait |
| South China Sea Disputes | Ongoing | SCS, Paracel & Spratly Islands |
| Iran–Gulf Tensions | Ongoing | Strait of Hormuz |

---

## 🧱 System Architecture

```text
Raw AIS CSV (daily)
    ↓
[Loader] 날짜 범위 로딩
    ↓
[Cleaner] 품질/물리 제약 기반 정제
    ↓
[Feature Engineer] 행동/지리/목적지 피처 생성
    ↓
[Aggregator] Grid × Time bucket 집계
    ↓
┌───────────────┬───────────────────┬────────────────────┐
│ Visualization │ Statistical Analysis│ Predictive Modeling │
└───────────────┴───────────────────┴────────────────────┘
```

---

## 📦 Project Layout

```bash
ais-conflict-intelligence/
├── config/settings.yaml
├── data/                     # AIS 원천 데이터(일자 폴더)
├── data_external/            # ACLED, world_ports 등 외부 데이터
├── docs/report.md
├── notebooks/                # EDA/검증/시각화/상관/모델링
├── scripts/
│   ├── run_pipeline.py
│   ├── run_pipeline.sh
│   └── generate_report.py
├── src/
│   ├── preprocessing/        # loader, cleaner, feature_engineer, aggregator
│   ├── visualization/        # spatial/temporal/statistical/conflict overlay
│   ├── analysis/             # traffic/behavioral/correlation/ROT/destination
│   └── models/               # anomaly, predictor, baseline, evaluator
└── tests/                    # unit tests
```

---

## 🚀 Quick Start

### 1) Installation

```bash
git clone https://github.com/Navy10021/ais-conflict-intelligence.git
cd ais-conflict-intelligence

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e .
```

> 선택 설치 예시: `pip install -e .[dev,viz,ml,forecasting,notebooks]`

### 2) Data Setup

1. Danish Maritime Authority AIS 데이터를 확보합니다.
2. 아래 형태로 저장합니다.
   - `data/YYYY-MM-DD/aisdk-YYYY-MM-DD.csv`
3. 외부 데이터(ACLED, ports)를 `data_external/`에 배치합니다.
   - `data_external/acled_events.csv`
   - `data_external/world_ports.csv`

### 3) Run Full Pipeline

```bash
# Bash wrapper
bash scripts/run_pipeline.sh

# Cross-platform Python CLI
python scripts/run_pipeline.py --date-start 2024-03-01 --date-end 2024-08-31 --verbose
```

### 4) Run by Stage (CLI Modules)

```bash
python -m src.preprocessing.loader --data-root ./data --output outputs/processed/ais_raw.parquet --verbose
python -m src.preprocessing.cleaner --input outputs/processed/ais_raw.parquet --output outputs/processed/ais_clean.parquet --verbose
python -m src.preprocessing.feature_engineer --input outputs/processed/ais_clean.parquet --output outputs/processed/ais_features.parquet --conflict-events data_external/acled_events.csv --ports data_external/world_ports.csv --verbose
python -m src.preprocessing.aggregator --input outputs/processed/ais_features.parquet --output outputs/processed/ais_aggregated.parquet --verbose
```

---

## 🧠 Feature Space (Examples)

| Category | Representative Features |
|---|---|
| Kinematic | `delta_sog`, `delta_cog`, `speed_category`, `sog_z_score` |
| ROT / Maneuver | `rot_abs`, `rot_spike`, `evasive_maneuver` |
| Geospatial | `grid_cell`, `in_conflict_zone`, `dist_chokepoint_km` |
| Behavioral | `route_entropy`, `loitering_flag`, `zig_zag_index` |
| Destination/ETA | `dest_changed`, `dest_is_conflict_port`, `eta_hours_remaining` |
| Temporal Aggregation | `traffic_count`, `military_ratio`, `dark_ship_ratio` |

---

## 📊 Analysis & Modeling Modules

### Analysis
- `traffic_analyzer`: 교통량/밀도/구성 변화 분석
- `behavioral_analyzer`: 행동 패턴·이상행동 시그널 분석
- `correlation_analyzer`: Granger, DiD, ITS, Event Study 기반 상관/인과 탐색
- `rot_analyzer`: 급선회/회피기동 관련 이상 시그널
- `destination_analyzer`: 목적지·입출항·ETA 기반 경로 변화

### Models
- `anomaly_model`: Isolation Forest, DBSCAN, LOF 등 이상탐지 접근
- `conflict_predictor`: 시계열 기반 분쟁 예측 모델 학습 파이프라인
- `baseline`: 비교 기준선 모델
- `evaluator`: 성능 평가 지표/리포팅

---

## 🧪 Testing

```bash
pytest tests/
pytest tests/test_loader.py -v
```

---

## 📚 Documentation

- `notebooks/01_EDA.ipynb` — 데이터 탐색
- `notebooks/02_preprocessing_validation.ipynb` — 전처리 검증
- `notebooks/03_visualization.ipynb` — 시각화
- `notebooks/04_conflict_correlation.ipynb` — 상관/이벤트 분석
- `notebooks/05_model_development.ipynb` — 모델링/평가
- `docs/report.md` — 산출물 리포트 초안

---

## 🛠️ Configuration Highlights

주요 실행 파라미터는 `config/settings.yaml`에서 관리합니다.

- 데이터 경로, 출력 경로
- 분석 기간 (`date_range`)
- 분쟁 구역 bbox 및 chokepoint 좌표
- 전처리 단위(`grid_size`, `time_bucket`)
- 학습 하이퍼파라미터
- 정제/검증 임계값

---

## 🗺️ Output Convention

기본 출력 루트는 `./outputs` 입니다.

- `outputs/processed/`: parquet 중간 산출물
- `outputs/figures/`: 시각화 결과
- `outputs/tables/`: 통계 테이블
- `outputs/models/`: 모델 아티팩트
- `outputs/reports/`: 리포트 파일

---

## 🤝 Contributing

PR을 환영합니다.

1. Fork
2. Feature branch 생성
3. 변경/테스트
4. Commit & Push
5. Pull Request 생성

---

## 📜 License

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 참고하세요.

## 📧 Contact

- Author: Yoonseop Lee
- Email: iyunseob4@gmail.com
- Project: https://github.com/Navy10021/ais-conflict-intelligence
