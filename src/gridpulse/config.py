"""Central configuration for GridPulse."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARQUET_DIR = DATA_DIR / "parquet"
DB_PATH = DATA_DIR / "gridpulse.duckdb"
REPORTS_DIR = PROJECT_ROOT / "reports"
BRIEFS_DIR = REPORTS_DIR / "briefs"
MODELS_DIR = PROJECT_ROOT / "models_artifacts"
EVALS_DIR = PROJECT_ROOT / "evals"

# NEM regions and the capital city whose weather drives most of the region's demand.
REGIONS = {
    "NSW1": {"city": "Sydney", "lat": -33.87, "lon": 151.21},
    "QLD1": {"city": "Brisbane", "lat": -27.47, "lon": 153.03},
    "SA1": {"city": "Adelaide", "lat": -34.93, "lon": 138.60},
    "TAS1": {"city": "Hobart", "lat": -42.88, "lon": 147.33},
    "VIC1": {"city": "Melbourne", "lat": -37.81, "lon": 144.96},
}

# NEM operates on Australian Eastern Standard Time (UTC+10), no daylight saving.
NEM_TZ = "Etc/GMT-10"

# A "spike" is any 5-min dispatch price at or above this level ($/MWh).
# $300 ~ the level where spot exposure starts to hurt retailers and where
# demand-response programs typically trigger.
SPIKE_THRESHOLD = 300.0

AEMO_PD_URL = (
    "https://www.aemo.com.au/aemo/data/nem/priceanddemand/"
    "PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
)
