#====個人設定==========================
CIF_FILE = "target.cif"  # もととなるcifファイル
FILENAME = "Unknown"  # 結果やログファイルに使用されるファイル名の先頭
TIME_STEP = 1  # 計算間隔 (fs)
MAIN_STEPS = 100 * 1000  # 昇温後の計算時間 (steps)

# 段階的に昇温する際の目標温度(K).最終温度は計算したい温度
HEATING_SCHEDULE = [100, 300, 500, 700, 900, 1100, 1373]
STEPS_PER_TEMP = 2000  # 各温度での待機ステップ数

# MACEモデル設定(※import torchは衝突防止のため使用しない)
# MD計算は高速な float32 で変換したモデル推奨
MODEL_PT = "mace_mp_medium_fp32.model-lammps.pt"
#=====================================

import os
import logging
import warnings

def _suppress_known_warnings():
    warnings.filterwarnings("ignore", message=r".*Issues encountered while parsing CIF.*")
    warnings.filterwarnings("ignore", message=r".*Setting the shape on a NumPy array.*")

_suppress_known_warnings()

from ase import units, io
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.calculators.lammpslib import LAMMPSlib

_suppress_known_warnings()

# 計算詳細設定
loginterval = 10  # 計算結果を保存する計算間隔
tdamp = 50.0  # 温度調整をする間隔
tchain = 3  # 計算におけるchain数

print(f"LAMMPS (ML-MACE) NVT-MD Calculator (Model: {MODEL_PT})")

if not os.path.exists(MODEL_PT):
    raise FileNotFoundError(f"LAMMPS用モデルファイルが見つかりません: {MODEL_PT}")

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

# モデル・構造準備
logger.info(f"Preparing LAMMPS (ML-MACE) with {MODEL_PT}...")
atoms = io.read(CIF_FILE)
elements = sorted(set(atoms.get_chemical_symbols()))
atom_types = {el: i + 1 for i, el in enumerate(elements)}

calc = LAMMPSlib(
    lammps_header=[
        "units metal",
        "atom_style atomic",
        "atom_modify map yes",
        "newton on",
        "box tilt large",
    ],
    lmpcmds=[
        "pair_style mace no_domain_decomposition",
        f"pair_coeff * * {MODEL_PT} {' '.join(elements)}",
    ],
    atom_types=atom_types,
    log_file="none",
)
atoms.calc = calc

try:
    # ループ処理で昇温
    logger.info("Heating process start")
    for temp_K in HEATING_SCHEDULE:
        time_ps = TIME_STEP * STEPS_PER_TEMP / 1000
        logger.info(f"Current target temperature: {temp_K}K-{time_ps}ps")

        dyn = NoseHooverChainNVT(
            atoms=atoms,
            timestep=TIME_STEP * units.fs,
            temperature_K=temp_K,
            tdamp=tdamp * units.fs,
            tchain=tchain,
            trajectory=f"{FILENAME}_heating_{int(temp_K):d}K_{int(time_ps):d}ps.traj",
            logfile=f"{FILENAME}_MD.log",
            loginterval=loginterval,
        )

        # 昇温用のMD計算
        dyn.run(steps=STEPS_PER_TEMP)

    # 目標温度でMD計算
    temp_K = HEATING_SCHEDULE[-1]
    time_ps = TIME_STEP * MAIN_STEPS / 1000
    logger.info(f"Warm-up complete. Main simulation begins: {temp_K}K-{time_ps}ps")
    dyn = NoseHooverChainNVT(
        atoms=atoms,
        timestep=TIME_STEP * units.fs,
        temperature_K=temp_K,
        tdamp=tdamp * units.fs,
        tchain=tchain,
        trajectory=f"{FILENAME}_main_{int(temp_K):d}K_{int(time_ps):d}ps.traj",
        logfile=f"{FILENAME}_MD.log",
        loginterval=loginterval,
    )
    dyn.run(steps=MAIN_STEPS)
finally:
    if hasattr(calc, "lmp") and calc.lmp is not None:
        calc.lmp.close()

logger.info("All calculation has been completed!")
