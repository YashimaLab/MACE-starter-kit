import os
import time
import warnings
from platform import python_version
import ase
from ase.build import bulk
from ase.calculators.lammpslib import LAMMPSlib

warnings.filterwarnings("ignore")

# ※LAMMPS (LibTorch) とのメモリ衝突(Segfault)を防ぐため、import torch / import mace を行っていません。

MODEL_FP32 = "mace_mp_medium_fp32.model-lammps.pt"
MODEL_FP64 = "mace_mp_medium_fp64.model-lammps.pt"


def benchmark_lammps_mace(model_path, label, steps=20):
    if not os.path.exists(model_path):
        print(f"⚠️ スキップ ({label}): {model_path} が見つかりません")
        return None

    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 2, 2))
    elements = sorted(set(atoms.get_chemical_symbols()))
    atom_types = {el: i + 1 for i, el in enumerate(elements)}

    calc = LAMMPSlib(
        lammps_header=[
            "units metal",
            "atom_style atomic",
            "atom_modify map yes",
            "newton on",
        ],
        lmpcmds=[
            "pair_style mace no_domain_decomposition",
            f"pair_coeff * * {model_path} {' '.join(elements)}",
        ],
        atom_types=atom_types,
        log_file="none",
    )
    atoms.calc = calc

    try:
        print(f"-> [{label}] モデル読み込み & ウォームアップ実行中...")
        _ = atoms.get_potential_energy()

        t0 = time.perf_counter()
        for _ in range(steps):
            atoms.positions[0, 0] += 1e-5  # ASEのキャッシュを無効化して毎回計算させる
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
        avg_ms = (time.perf_counter() - t0) / steps * 1000

        print(f"✅ [{label}] 完了! エネルギー: {energy / len(atoms):.6f} eV/atom")
        print(f"   平均推論時間: {avg_ms:.2f} ms / step ({len(atoms)}原子)")
        return avg_ms
    finally:
        if hasattr(calc, "lmp") and calc.lmp is not None:
            calc.lmp.close()


if __name__ == "__main__":
    print("=== LAMMPS (ML-MACE) 動作環境確認 ===")
    print(f"Python version : {python_version()}")
    print(f"ASE version    : {ase.__version__}")
    print("※ GPU認識の有無は下記の LAMMPS 出力 (CUDA found...) で確認できます")

    print("=== MACE-MP-0 (medium) LAMMPS ベンチマーク ===")
    t_fp32 = benchmark_lammps_mace(MODEL_FP32, "単精度 float32")
    print()
    t_fp64 = benchmark_lammps_mace(MODEL_FP64, "倍精度 float64")

    if t_fp32 is not None and t_fp64 is not None:
        print(f"📊 速度比較: 単精度 (fp32) は 倍精度 (fp64) より {t_fp64 / t_fp32:.2f}x 高速です")
