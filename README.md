# MACE-starter-kit
汎用機械学習ポテンシャル **MACE (MACE-MP-0)** を用いて、結晶構造最適化および分子動力学（MD）計算を高速に実行するための環境構築マニュアル・動作テスト・計算スクリプト一式をまとめたリポジトリです。

用途や実行環境に応じて使い分けられるよう、**① NVIDIA `cuEquivariance` による GPU 最適化環境（`*_cuEq.py`）** と、**② 研究室 PC 環境に準拠した `LAMMPS (ML-MACE)` 連携環境（`*_LAMM.py`）** の 2 系統を収録しています。

---

## 1. リポジトリ内のファイル一覧

| ファイル名 | 種別 | 概要・役割 |
| :--- | :--- | :--- |
| **`README.md`** | ドキュメント | 本ファイル。リポジトリ全体の概要、環境比較、使い分けの指針を記載 |
| **`env_construction_MACE_cuEq.md`** | マニュアル | `cuEquivariance` 有効化環境（`mace_env`）の構築手順・トラブルシューティング |
| **`env_construction_MACE_LAMM.md`** | マニュアル | GPU 対応 `LAMMPS (ML-MACE)` 環境（`lammps_mace_env`）のビルド・運用手順 |
| **`install_tester_MACE_cuEq.py`** | テスト | `cuEq` 環境の GPU 認識・推論速度検証スクリプト |
| **`install_tester_MACE_LAMM.py`** | テスト | `LAMMPS` 環境の GPU 認識・単精度/倍精度（`fp32`/`fp64`）速度比較スクリプト |
| **`Structure_optimizer_MACE_cuEq.py`** | 計算実行 | `cuEq` 環境用：複数 `.cif` ファイルの一括構造最適化（BFGS / FIRE, 格子緩和対応） |
| **`Structure_optimizer_MACE_LAMM.py`** | 計算実行 | `LAMMPS` 環境用：複数 `.cif` ファイルの一括構造最適化（組成ごとのモデルキャッシュ機能付） |
| **`MD_calculator_MACE_cuEq.py`** | 計算実行 | `cuEq` 環境用：段階昇温 ＋ 本計算 NVT-MD シミュレーション（Nose-Hoover Chain） |
| **`MD_calculator_MACE_LAMM.py`** | 計算実行 | `LAMMPS` 環境用：段階昇温 ＋ 本計算 NVT-MD シミュレーション（Nose-Hoover Chain） |
| **`MD_successer_MACE_cuEq.py`** | 計算実行 | `cuEq` 環境用：既存の `.traj` ファイルの最終フレームからの継続 MD（リスタート） |
| **`MD_successer_MACE_LAMM.py`** | 計算実行 | `LAMMPS` 環境用：既存の `.traj` ファイルの最終フレームからの継続 MD（リスタート） |
| **`mace_mp_medium_fp64.model-lammps.pt`** | モデル | LAMMPS (`pair_style mace`) 読み込み用に変換済みの MACE-MP-0 (medium, 倍精度) モデル |

---

## 2. MACE とは何か（CHGNet・M3GNet との違い）

**MACE (Multi-Atomic Cluster Expansion)** は、Atomic Cluster Expansion (ACE) の多体相関表現と、$E(3)$ 等変メッセージパッシングニューラルネットワーク（Equivariant MPNN）を組み合わせた機械学習原子間ポテンシャルです。

* **CHGNet / M3GNet との構造的な違い**:
  CHGNet や M3GNet が主に原子間距離や結合角といった「スカラー量（不変量）」をベースに局所環境を記述するのに対し、MACE は原子の周囲の幾何学的な方向や対称性を保ったまま「高次テンソル（等変量）」として情報を伝播させます。
* **ポテンシャルエネルギー面（PES）の滑らかさ**:
  多体相互作用と対称性を厳密に組み込んでいるため、原子変位に対するエネルギーと力の勾配（微分）が非常に滑らかになります。これにより、構造最適化（BFGS 等）において最短ルートで極小点へ収束しやすく、未知の配位環境や複雑な多元素系でも高い予測精度を発揮します。

---

## 3. CHGNet との性能・メモリ使用量比較

$3 \times 3 \times 1$ Sillen構造（72 原子系 、BFGS 構造最適化）および RTX 3060 Ti (VRAM 8 GB) 環境での実測・リソース比較は以下の通りです。

### 計算速度と収束性の実測比較（72 原子系）

| 手法・環境 | 計算精度 | 1 モデルあたりの処理時間 | 収束までのステップ数 | 特徴 |
| :--- | :---: | :---: | :---: | :--- |
| **CHGNet** | `float32` | 約 10 秒 / model | 約 150 step | 軽量だが PES の微細な凹凸によりステップ数が嵩む |
| **MACE + cuEquivariance** | **`float64`** | **約 4 秒 / model** | **約 50 step** | **倍精度でありながら CHGNet の 2.5 倍高速** |
| **MACE + LAMMPS (TorchScript)** | `float64` | 約 16 秒 / model | 約 50 step | GPU計算部分は最適化なし |

* **MACE が CHGNet より速い理由**:
  ポテンシャル曲面が滑らかなため **収束ステップ数が約 1/3（150 step $\rightarrow$ 50 step）に減少** し、さらに `cuEquivariance` による GPU カーネル最適化が効くことで、倍精度（`fp64`）計算であっても CHGNet を大きく上回るスループットを実現しています。

### GPU メモリ（VRAM）消費量の違い

MACE は CHGNet に比べて **約 2〜4 倍の VRAM** を消費します。

| モデル・精度 | カットオフ半径 | 72 原子計算時 | 400 原子計算時 | 備考 |
| :--- | :---: | :---: | :---: | :--- |
| **CHGNet** (`float32`) | $5.0\text{ \AA}$ | 約 0.6〜1.0 GB | 約 1.5〜2.5 GB | パラメータ数が少なく省メモリ |
| **MACE (medium, `fp32`)** | $6.0\text{ \AA}$ | 約 1.5〜2.5 GB | 約 4.5〜6.5 GB | VRAM 8 GB 環境では 400 原子前後まで実用的 |
| **MACE (medium, `fp64`)** | $6.0\text{ \AA}$ | 約 2.5〜4.0 GB | **8 GB 超過 (OOM 危険)** | テンソルが倍精度になり消費量が倍増 |

* カットオフ半径が $6.0\text{ \AA}$ と広いこと（体積比で隣接原子数が CHGNet の約 1.7 倍）と、高次等変テンソルの勾配を保持する構造上、原子数が増えると急激に VRAM を消費します。VRAM 8 GB の GPU で 300 原子以上の系を扱う場合は `float32` の使用を推奨します。

---

## 4. システム環境（研究室環境 vs ローカル構築環境）

### 研究室 PC 環境（標準運用構成）
* **構成**: `ACEsuit/lammps`（`ML-MACE` パッケージ有効） + CUDA 版 C++ `LibTorch` + Python (`ASE` の `LAMMPSlib`)
* **特徴**: PyTorch の Python 側パッケージではなく、C++ でビルドされた LAMMPS バイナリおよび共有ライブラリ（`liblammps.so`）から、TorchScript 形式に変換済みの `.model-lammps.pt` を直接 GPU にロードして計算を実行します（`cuEquivariance` は未導入）。

### ローカル構築環境（益山自宅 PC: Ubuntu 26.04 on WSL2）
* **ハードウェア**: NVIDIA GeForce RTX 3060 Ti (VRAM 8 GB)
* **環境分離設計**: Python 版 PyTorch（CUDA 13.x）と LAMMPS 組み込みの C++ 版 LibTorch（CUDA 12.6）のライブラリ衝突（Segmentation fault）を防ぐため、2 つの独立した仮想環境を構築しています。
  1. **`~/mace_env`（`cuEquivariance` 環境）**
     * Python 3.14 / PyTorch (CUDA 13.x) / `mace-torch` / `cuequivariance` / `cuequivariance-torch` / `ase` / `pymatgen` / `pandas`
  2. **`~/lammps_mace_env`（`LAMMPS ML-MACE` 環境）**
     * Python 3.14 / `ACEsuit/lammps` (`PKG_ML-MACE=ON`, `BUILD_SHARED_LIBS=ON`) / C++ `LibTorch` (CUDA 12.6) / `ase` / `pymatgen` / `pandas`

---

## 5. モデルサイズ・精度と `cuEq` / `LAMMPS` の違い

### MACE-MP-0 のモデルサイズと演算精度
* **モデルサイズ (`small` / `medium` / `large`)**:
  * `small`: 軽量・高速だが多元素系や複雑な配位環境では精度がやや落ちる。
  * **`medium`（標準推奨）**: $L=1$・128 チャネルを備え、計算速度と DFT 再現精度のバランスに最も優れる。
  * `large`: 最高精度だが計算コストと VRAM 消費量が非常に大きい。
* **演算精度 (`float32` vs `float64`)**:
  * **`float32`（単精度）**: MD 計算および通常の構造スクリーニング（`fmax=0.01` 程度）向け。エネルギーは `float64` と小数点以下 7 桁目まで一致し、RTX 3060 Ti では **約 2.5 倍高速** に動作します（コンシューマー向け GeForce GPU はハードウェア仕様上 `fp64` 性能が `fp32` の 1/64 に制限されているため）。
  * **`float64`（倍精度）**: 力の収束判定を極端に厳しく追い込む精密構造緩和（`fmax <= 0.001` やフォノン計算前処理）向け。

### `cuEquivariance` (cuEq) と `LAMMPS` の技術的な違い
* **`cuEquivariance` (cuEq)**:
  NVIDIA が開発した等変ニューラルネットワーク専用のカスタム CUDA カーネルライブラリです。MACE の計算律速である「クレブシュ・ゴルダン係数を用いたテンソル積演算」を GPU 上で融合・最適化して実行するため、特に多元素系で圧倒的な速度を発揮します。
* **`LAMMPS (ML-MACE)`**:
  汎用分子動力学エンジン LAMMPS に C++ 版 LibTorch をリンクし、`pair_style mace` として TorchScript 化した MACE モデルを呼び出す仕組みです。C++ 側で高速なセルリスト法（隣接原子探索）と時間発展を完結できるほか、既存の LAMMPS 入力スクリプトや並列計算基盤と統合できる利点があります。

---

## 6. `cuEq` 環境の使い方（`*_cuEq.py`）

`mace_env` を有効化し、`enable_cueq=True` を指定したスクリプトを実行します。

```bash
# 仮想環境の有効化
source ~/mace_env/bin/activate

# 動作確認テスト
python install_tester_MACE_cuEq.py

# 構造最適化の実行
python Structure_optimizer_MACE_cuEq.py

# NVT-MD 計算の実行（新規 / リスタート）
python MD_calculator_MACE_cuEq.py
python MD_successer_MACE_cuEq.py
```

* **適している用途**: 数十〜数百原子規模の構造最適化スクリーニング（特に 6 元素など多元素系）や、ASE ベースの NVT-MD 計算において**最速**で動作します。

---

## 7. `LAMMPS` 環境はいつ使うのか（`*_LAMM.py`）

`lammps_mace_env` を有効化し、変換済みの `.model-lammps.pt` を指定して実行します。

```bash
# 仮想環境の有効化
source ~/lammps_mace_env/bin/activate

# 動作確認テスト
python install_tester_MACE_LAMM.py

# 構造最適化・MD 計算の実行
python Structure_optimizer_MACE_LAMM.py
python MD_calculator_MACE_LAMM.py
python MD_successer_MACE_LAMM.py
```

### LAMMPS 環境を使うべき場面
1. **研究室の PC 環境でそのまま動かしたいとき（最大の強み）**
   * 研究室の計算機は `LAMMPS (ML-MACE)` + `.model-lammps.pt` 構成で統一されているため、`*_LAMM.py` と変換済み `.pt` ファイルをそのまま研究室 PC へコピーするだけで、環境の差異を気にせず即座に実行・再現できます。
2. **大規模系（400 原子以上）や長時間の MD 計算を行うとき**
   * 原子数が増えると Python（ASE）側での隣接リスト構築や座標データの往復オーバーヘッドが大きくなります。純粋な LAMMPS 入力ファイル（`lmp -in in.lammps`）へ移行することで、Python を一切介さずに C++ 側だけで計算を完結できます。
3. **LAMMPS 固有のシミュレーション機能を使いたいとき**
   * NPT アンサンブル（定圧制御）や各種バイアス計算など、ASE よりも LAMMPS 標準コマンドが適している計算にそのまま流用できます。

> **⚠️ `*_LAMM.py` 編集時の重要な注意**:
> `LAMMPSlib` を使用する Python スクリプト内では、** `import torch` や `import mace` を記述しないでください**。Python 側の PyTorch と LAMMPS 側の C++ LibTorch がメモリ上で衝突し、`Segmentation fault` で強制終了します。

---

## 8. LAMMPS 用 `.pt` ファイルの入手・変換方法

LAMMPS (`pair_style mace`) は通常の `.model` ファイルを直接読み込めないため、`mace_create_lammps_model` コマンドで事前に **TorchScript 形式（`.model-lammps.pt`）** へ変換します。

### 変換コマンド（単精度 `fp32`・倍精度 `fp64` の作成例）

```bash
source ~/lammps_mace_env/bin/activate

# 1. 元となる MACE-MP-0 (medium) モデルを保存（初回のみ）
python -c 'import torch; from mace.calculators import mace_mp; torch.save(mace_mp(model="medium", return_raw_model=True), "mace_mp_medium.model")'

# 2. 単精度（float32）版の変換（MD計算・高速スクリーニング推奨）
mace_create_lammps_model mace_mp_medium.model --dtype float32
mv mace_mp_medium.model-lammps.pt mace_mp_medium_fp32.model-lammps.pt

# 3. 倍精度（float64）版の変換（精密構造最適化推奨）
mace_create_lammps_model mace_mp_medium.model --dtype float64
mv mace_mp_medium.model-lammps.pt mace_mp_medium_fp64.model-lammps.pt
```

* **モデルの汎用性**: 一度変換した `.model-lammps.pt` ファイルは**特定の結晶構造や元素に縛られず、あらゆる組成の計算に使い回せます**。また、OS や GPU 機種にも依存しないため、作成した `.pt` ファイルをそのまま研究室 PC にコピーして使用できます。
* **独自ファインチューニングモデルの場合**: 自作の `.model` ファイルに対しても同様に `mace_create_lammps_model your_model.model --dtype float32` を実行することで、LAMMPS で使用可能になります。
