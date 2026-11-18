import os

# Base directory for random_forest folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Locate Student folder dynamically
candidate_dirs = [
    os.path.join(REPO_ROOT, "Student folder", "Student folder"),
    os.path.join(REPO_ROOT, "Student folder"),
    os.path.join(REPO_ROOT, "data", "Student folder"),
    os.path.join(BASE_DIR, "Student folder", "Student folder"),
    os.path.join(BASE_DIR, "Student folder"),
    r"C:\Users\sanju\ml-hackathon\Student folder\Student folder"
]

STUDENT_DATA_DIR = next((d for d in candidate_dirs if os.path.exists(d)), candidate_dirs[0])
TRAIN_DIR = os.path.join(STUDENT_DATA_DIR, "train")
TEST_DIR = os.path.join(STUDENT_DATA_DIR, "student_test")

DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Coordinate projection constants centered at Darmstadt/Frankfurt, Germany
REF_LAT = 49.85
REF_LON = 8.59
EARTH_RADIUS = 6371000.0  # meters

# Spatial candidate generation parameters
SEGMENT_SAMPLE_INTERVAL = 25.0  # meters
CANDIDATE_SEARCH_RADIUS = 50.0  # meters
MAX_CANDIDATES_PER_OBS = 6

FEATURE_COLUMNS = [
    "perp_dist",
    "heading_diff",
    "seg_len",
    "proj_t",
    "dist_rank",
    "veh_spd",
    "spd_change",
    "n_candidates",
    "perp_dist_norm"
]

RANDOM_STATE = 42
