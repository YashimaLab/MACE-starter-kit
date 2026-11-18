#====個人設定==========================
INPUT_CIF_PATTERN = 'target/*.cif'  # 計算したいcifファイルの場所
OUTPUT_CIF = './output'  # 最適化後のcifファイルの格納場所(フォルダは自動で作られる)
OUTPUT_CSV = "result.csv"  # 処理したcifファイル名、エネルギー、ステップ数が記録されるcsvファイル名
ALLOW_CELL_DEFORMATION = False  # 格子定数を変化させるか(Trueならば格子緩和あり)

OPTIMIZER_CLASS = "BFGS"  # 構造を収束させる際の解法 ("BFGS" または "FIRE")
# 格子間酸素があるなど不安定な原子位置がある場合には"FIRE"を選択するべき
# いずれでも最終的な結果は通常変わらないが、"BFGS"の方が原子が飛んで想定外の収束をしやすい代わりに速い。

# MACEモデル設定
MODEL_SIZE = "medium"  # "small", "medium", "large"
ENABLE_CUEQ = True  # cuEquivarianceによる高速化を有効化
DEFAULT_DTYPE = "float64"  # 構造最適化は力の収束精度を高めるためfloat64推奨
#=====================================

import warnings

def _suppress_known_warnings():
    warnings.filterwarnings("ignore", message=r".*torch\.jit\.(script|load).*")
    warnings.filterwarnings("ignore", message=r".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", message=r".*To copy construct from a tensor.*")
    warnings.filterwarnings("ignore", message=r".*Issues encountered while parsing CIF.*")
    warnings.filterwarnings("ignore", message=r".*Setting the shape on a NumPy array.*")

_suppress_known_warnings()

import glob
import os
import datetime
import torch
_suppress_known_warnings()
import pandas as pd
from ase.optimize import BFGS, FIRE
from ase.filters import FrechetCellFilter
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp

_suppress_known_warnings()

OPTIMIZERS = {
    "BFGS": BFGS,
    "FIRE": FIRE,
}


def RAP(file_path, calc, results):
    print(f"{file_path} 構造最適化開始({datetime.datetime.now()})")

    # 構造の読み込み (元のCHGNetスクリプトと同じくpymatgen経由で形式統一)
    structure = Structure.from_file(file_path)
    structure.set_charge(0.0)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calc

    # 格子緩和の有無に応じてターゲットを設定
    opt_target = FrechetCellFilter(atoms) if ALLOW_CELL_DEFORMATION else atoms

    if OPTIMIZER_CLASS not in OPTIMIZERS:
        raise ValueError(f"未対応のOPTIMIZER_CLASSです: {OPTIMIZER_CLASS} ('BFGS' または 'FIRE' を指定)")
    opt = OPTIMIZERS[OPTIMIZER_CLASS](opt_target, logfile=None)

    # 最適化実行
    opt.run(
        fmax=0.01,  # 収束したとみなす力の最大値 (eV/Å)
        steps=500,  # 最大ステップ数
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


print(f"=== MACE-MP-0 ({MODEL_SIZE}) 構造最適化 (cuEq={ENABLE_CUEQ}) ===")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用ハードウェア: {device}")

cif_files = sorted(glob.glob(INPUT_CIF_PATTERN))
results = []

if not cif_files:
    print("CIFファイルが見つかりませんでした。")
else:
    print(f"{len(cif_files)} 個のCIFファイルが見つかりました。処理を開始します...")
    calc = mace_mp(
        model=MODEL_SIZE,
        device=device,
        default_dtype=DEFAULT_DTYPE,
        enable_cueq=ENABLE_CUEQ,
    )

    os.makedirs(OUTPUT_CIF, exist_ok=True)
    for cif_file in cif_files:
        RAP(cif_file, calc, results)

df = pd.DataFrame(results, columns=['filename', 'Energy', 'step'])
df.to_csv(OUTPUT_CSV, index=False)
print(f"すべての処理が完了しました({datetime.datetime.now()})")
