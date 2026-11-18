import time
import warnings
from platform import python_version
import torch
import ase
from ase.build import bulk
import mace
import cuequivariance
from mace.calculators import mace_mp

warnings.filterwarnings("ignore", message=".*torch.jit.script.*")

print("=== 動作環境確認 ===")
print(f"Python version : {python_version()}")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f" プログラムはGPUで実行可能 ({torch.cuda.get_device_name(0)})")
else:
    print(" プログラムはCPUで実行されます")
print(f"ASE version    : {ase.__version__}")
print(f"MACE version   : {mace.__version__}")
print(f"cuEq version   : {cuequivariance.__version__}")

print("\n=== MACE-MP-0 (medium) + cuEquivariance 動作テスト ===")
print("-> モデルを読み込んでいます (初回はモデルのダウンロードが行われます)...")
calc_cueq = mace_mp(
    model="medium",
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_dtype="float32",
    enable_cueq=True,
)

# テスト用Siスーパーセル (64原子)
atoms = bulk("Si", "diamond", a=5.43).repeat((2, 2, 2))
atoms.calc = calc_cueq

print("-> cuEquivariance 有効状態で計算実行中...")
_ = atoms.get_potential_energy()  # ウォームアップ
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
torch.cuda.synchronize()
t_cueq = (time.perf_counter() - t0) / 20 * 1000

print(f"✅ 計算完了! エネルギー: {energy / len(atoms):.4f} eV/atom")
print(f"   平均推論時間 (cuEq ON) : {t_cueq:.2f} ms / step (64原子 Si)")

# 比較用: cuEquivariance OFF
calc_no_cueq = mace_mp(
    model="medium",
    device="cuda" if torch.cuda.is_available() else "cpu",
    default_dtype="float32",
    enable_cueq=False,
)
atoms.calc = calc_no_cueq
_ = atoms.get_potential_energy()
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
torch.cuda.synchronize()
t_no_cueq = (time.perf_counter() - t0) / 20 * 1000

print(f"   平均推論時間 (cuEq OFF): {t_no_cueq:.2f} ms / step (速度倍率: {t_no_cueq / t_cueq:.2f}x)")
