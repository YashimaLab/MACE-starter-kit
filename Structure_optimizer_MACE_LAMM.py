#====個人設定==========================
INPUT_CIF_PATTERN = 'target/*.cif'  # 計算したいcifファイルの場所
OUTPUT_CIF = './output'  # 最適化後のcifファイルの格納場所(フォルダは自動で作られる)
OUTPUT_CSV = "result.csv"  # 処理したcifファイル名、エネルギー、ステップ数が記録されるcsvファイル名
ALLOW_CELL_DEFORMATION = False  # 格子定数を変化させるか(Trueならば格子緩和あり)

OPTIMIZER_CLASS = "BFGS"  # 構造を収束させる際の解法 ("BFGS" または "FIRE")
# 格子間酸素があるなど不安定な原子位置がある場合には"FIRE"を選択するべき
# いずれでも最終的な結果は通常変わらないが、"BFGS"の方が原子が飛んで想定外の収束をしやすい代わりに速い。

# MACEモデル設定
MODEL_PT = "mace_mp_medium_fp64.model-lammps.pt"
#=====================================

import glob
import os
import datetime
import warnings

def _suppress_known_warnings():
    warnings.filterwarnings("ignore", message=r".*Issues encountered while parsing CIF.*")
    warnings.filterwarnings("ignore", message=r".*Setting the shape on a NumPy array.*")

_suppress_known_warnings()

import pandas as pd
from ase.optimize import BFGS, FIRE
from ase.filters import FrechetCellFilter
from ase.calculators.lammpslib import LAMMPSlib
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

_suppress_known_warnings()

OPTIMIZERS = {
    "BFGS": BFGS,
    "FIRE": FIRE,
}

# 元素リストごとにLAMMPSインスタンスをキャッシュして使い回す辞書
_CALC_CACHE = {}


def get_lammps_calculator(elements_tuple, model_pt):
    if elements_tuple not in _CALC_CACHE:
        atom_types = {el: i + 1 for i, el in enumerate(elements_tuple)}
        _CALC_CACHE[elements_tuple] = LAMMPSlib(
            lammps_header=[
                "units metal",
                "atom_style atomic",
                "atom_modify map yes",
                "newton on",
                "box tilt large",
            ],
            lmpcmds=[
                "pair_style mace no_domain_decomposition",
                f"pair_coeff * * {model_pt} {' '.join(elements_tuple)}",
            ],
            atom_types=atom_types,
            log_file="none",
        )
    return _CALC_CACHE[elements_tuple]


def RAP(file_path, model_pt, results):
    print(f"{file_path} 構造最適化開始({datetime.datetime.now()})")

    # 構造の読み込み
    structure = Structure.from_file(file_path)
    structure.set_charge(0.0)
    atoms = AseAtomsAdaptor.get_atoms(structure)

    # 元素の組み合わせが同じなら既存のLAMMPSインスタンスを再利用
    elements_tuple = tuple(sorted(set(atoms.get_chemical_symbols())))
    calc = get_lammps_calculator(elements_tuple, model_pt)
    atoms.calc = calc

    # 格子緩和の有無に応じてターゲットを設定
    opt_target = FrechetCellFilter(atoms) if ALLOW_CELL_DEFORMATION else atoms

    if OPTIMIZER_CLASS not in OPTIMIZERS:
        raise ValueError(f"未対応のOPTIMIZER_CLASSです: {OPTIMIZER_CLASS} ('BFGS' または 'FIRE' を指定)")
    opt = OPTIMIZERS[OPTIMIZER_CLASS](opt_target, logfile=None)

    # 最適化実行
    opt.run(
        fmax=0.01,
        steps=500,
    )

    # 結果の保存
    final_energy = atoms.get_potential_energy()
    final_structure = AseAtomsAdaptor.get_structure(atoms)

    results.append({
        'filename': os.path.basename(file_path),
        'Energy': f"{final_energy:.4f}",
        'step': opt.nsteps,
    })
    output_file_path = os.path.join(OUTPUT_CIF, f"{os.path.splitext(os.path.basename(file_path))[0]}_relaxed.cif")
    final_structure.to(filename=output_file_path)


print(f"=== LAMMPS (ML-MACE) 構造最適化 (Model: {MODEL_PT}) ===")

if not os.path.exists(MODEL_PT):
    raise FileNotFoundError(f"LAMMPS用モデルファイルが見つかりません: {MODEL_PT}")

cif_files = sorted(glob.glob(INPUT_CIF_PATTERN))
results = []

try:
    if not cif_files:
        print("CIFファイルが見つかりませんでした。")
    else:
        print(f"{len(cif_files)} 個のCIFファイルが見つかりました。処理を開始します...")
        os.makedirs(OUTPUT_CIF, exist_ok=True)
        for cif_file in cif_files:
            RAP(cif_file, MODEL_PT, results)
finally:
    for calc in _CALC_CACHE.values():
        if hasattr(calc, "lmp") and calc.lmp is not None:
            calc.lmp.close()

df = pd.DataFrame(results, columns=['filename', 'Energy', 'step'])
df.to_csv(OUTPUT_CSV, index=False)
print(f"すべての処理が完了しました({datetime.datetime.now()})")