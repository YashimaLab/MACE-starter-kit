#====個人設定==========================
TRAJ_FILE = "history.traj"  # 前回の計算trajファイル(この続きから計算され、計算結果はこのファイルに追記される)
FILENAME = "Unknown"  # 結果やログファイルに使用される
TIME_STEP = 1  # 計算間隔 (fs)
MAIN_STEPS = 100 * 1000  # 計算時間 (steps)
HEATING_TEMPERATURE = 1373  # 目標温度(K)

# MACEモデル設定
MODEL_SIZE = "medium"  # "small", "medium", "large"
ENABLE_CUEQ = True  # cuEquivarianceによる高速化を有効化
DEFAULT_DTYPE = "float32"  # MD計算は高速なfloat32推奨
#=====================================

import warnings

def _suppress_known_warnings():
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.(script|load).*")
    warnings.filterwarnings("ignore", message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=r".*To copy construct from a tensor.*")
    warnings.filterwarnings("ignore", message=r".*Issues encountered while parsing CIF.*")
    warnings.filterwarnings("ignore", message=r".*Setting the shape on a NumPy array.*")

_suppress_known_warnings()

import logging
import torch
_suppress_known_warnings()
from ase import units, io
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from mace.calculators import mace_mp

_suppress_known_warnings()

# 計算詳細設定
loginterval = 10  # 計算結果を保存する計算間隔
tdamp = 50.0  # 温度調整をする間隔
tchain = 3  # 計算におけるchain数

print("MACE-MP-0 (medium) + cuEquivariance NVT-MD Successer")

# 計算処理時刻ログ作成設定
logger = logging.getLogger("PROCESS_TIME")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(f'{FILENAME}_process.txt', encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# モデル準備
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Preparing MACE-MP-0 ({MODEL_SIZE}) on {device} (cuEq={ENABLE_CUEQ})...")
atoms = io.read(TRAJ_FILE, index=-1)
traj = io.Trajectory(TRAJ_FILE, 'a', atoms)
atoms.calc = mace_mp(
    model=MODEL_SIZE,
    device=device,
    default_dtype=DEFAULT_DTYPE,
    enable_cueq=ENABLE_CUEQ,
)

# 目標温度でMD計算
temp_K = HEATING_TEMPERATURE
time_ps = TIME_STEP * MAIN_STEPS / 1000
logger.info(f"Warm-up complete. Main simulation begins: {temp_K}K-{time_ps}ps")
dyn = NoseHooverChainNVT(
    atoms=atoms,
    timestep=TIME_STEP * units.fs,
    temperature_K=temp_K,
    tdamp=tdamp * units.fs,
    tchain=tchain,
    trajectory=traj,
    logfile=f"{FILENAME}_MD.log",
    loginterval=loginterval,
)
dyn.run(steps=MAIN_STEPS)
traj.close()

logger.info("All calculation has been completed!")
