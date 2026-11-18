# LAMMPS (ML-MACE) GPU環境 構築・運用マニュアル

研究室スクリプトとの互換性および大規模 MD 計算を目的として構築した、GPU 対応 LAMMPS（`ACEsuit/lammps` + `ML-MACE` + CUDA 版 `LibTorch`）の環境構成、構築手順、および日常的な計算実行手順をまとめたドキュメントです。

---

## 1. システム構成とディレクトリ概要

### ハードウェア・OS 環境
* **OS**: Ubuntu 26.04 (WSL2 on Windows)
* **GPU**: NVIDIA GeForce RTX 3060 Ti (VRAM 8 GB)
* **CUDA / LibTorch**: CUDA 12.6 対応 C++ LibTorch
* **Python**: Python 3.14（仮想環境で分離）

### 主要ディレクトリ一覧
* **`~/lammps_mace_env/`**: LAMMPS (ML-MACE) 実行用の Python 仮想環境（`ase`, `pymatgen`, `pandas`, `mace-torch` 等を導入）
* **`~/lammps_mace/`**: LAMMPS ソースコード・ビルド・インストール先
  * `~/lammps_mace/libtorch/`: C++ 版 LibTorch (CUDA 12.6)
  * `~/lammps_mace/lammps/install-gpu/bin/lmp`: GPU 対応 LAMMPS 実行バイナリ
  * `~/lammps_mace/lammps/install-gpu/lib/liblammps.so`: Python (`LAMMPSlib`) から呼び出される共有ライブラリ
* **`~/lammps_mace_test/`**: 動作検証・ベンチマーク・モデル格納用作業フォルダ
  * `~/lammps_mace_test/python/`: Python 連携スクリプト実行フォルダ

---

## 2. 環境構築手順（再現ログ）

※ `cuEquivariance` 環境（`~/mace_env`）とライブラリが競合しないよう、専用のディレクトリと仮想環境に分けて構築しています。

### 2.1 ビルドツールと Python 仮想環境の構築
```bash
# ビルドに必要なパッケージの導入
sudo apt update
sudo apt install -y cmake build-essential git wget unzip libopenmpi-dev openmpi-bin

# 専用Python仮想環境の作成と有効化
python3 -m venv ~/lammps_mace_env
source ~/lammps_mace_env/bin/activate

# 必要なPythonライブラリのインストール
pip install --upgrade pip
pip install mace-torch ase pymatgen pandas
```

### 2.2 C++ LibTorch (CUDA 12.6) の配置
```bash
mkdir -p ~/lammps_mace && cd ~/lammps_mace

# CUDA 12.6 対応 LibTorch (cxx11 ABI) のダウンロードと展開
wget https://download.pytorch.org/libtorch/cu126/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu126.zip -O libtorch.zip
unzip libtorch.zip
rm libtorch.zip
```

### 2.3 GPU 対応 LAMMPS (`ML-MACE`) のビルド
```bash
cd ~/lammps_mace
git clone --depth=1 https://github.com/ACEsuit/lammps.git
cd lammps
mkdir build-gpu && cd build-gpu

cmake ../cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$HOME/lammps_mace/lammps/install-gpu \
  -D CMAKE_CXX_STANDARD=17 \
  -D CMAKE_CXX_STANDARD_REQUIRED=ON \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_MPI=ON \
  -D PKG_ML-MACE=ON \
  -D CMAKE_PREFIX_PATH=$HOME/lammps_mace/libtorch

make -j$(nproc)
make install
```

### 2.4 パスと環境変数の設定
`~/.bashrc` に以下を追記し、`lmp` コマンドと `liblammps.so` をどこからでも認識できるようにします。

```bash
cat << 'EOF' >> ~/.bashrc

# LAMMPS (ML-MACE) GPU Environment
export PATH="$HOME/lammps_mace/lammps/install-gpu/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/lammps_mace/lammps/install-gpu/lib:$HOME/lammps_mace/libtorch/lib:$LD_LIBRARY_PATH"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
EOF

source ~/.bashrc
```

---

## 3. MACE モデルの変換手順（TorchScript 形式）

LAMMPS (`pair_style mace`) は通常の `.model` ファイルを直接読み込めないため、`mace_create_lammps_model` コマンドで事前に **`.model-lammps.pt`（TorchScript 形式）** へ変換しておく必要があります。

### 単精度（`float32`）と倍精度（`float64`）の両方を作成する場合
```bash
source ~/lammps_mace_env/bin/activate
cd ~/lammps_mace_test

# 1. 元モデル（mace_mp_medium.model）を用意（未取得の場合）
python -c 'import torch; from mace.calculators import mace_mp; torch.save(mace_mp(model="medium", return_raw_model=True), "mace_mp_medium.model")'

# 2. 単精度（float32）版の変換（MD計算・高速スクリーニング用）
mace_create_lammps_model mace_mp_medium.model --dtype float32
mv mace_mp_medium.model-lammps.pt mace_mp_medium_fp32.model-lammps.pt

# 3. 倍精度（float64）版の変換（精密構造最適化用）
mace_create_lammps_model mace_mp_medium.model --dtype float64
mv mace_mp_medium.model-lammps.pt mace_mp_medium_fp64.model-lammps.pt
```
* 変換した `.model-lammps.pt` ファイルは、一度作成すれば**どの元素・結晶構造の計算にも使い回せます**。
* ファインチューニングした独自モデル（`.model`）を使う場合も、同じコマンドで `.model-lammps.pt` に変換するだけで使用できます。

---

## 4. 普段の実行方法

計算を実行する際は、必ず `lammps_mace_env` を有効化してから実行します。

```bash
source ~/lammps_mace_env/bin/activate
```

### 4.1 構造最適化（複数 CIF の一括処理）
* **使用スクリプト**: `LAMMPS_MACE_Structure_optimizer.py`
* **特徴**: 元素構成が同じ構造に対しては LAMMPS インスタンスをキャッシュして使い回すため、モデルの読み込み・コンパイルは最初の 1 ファイル目のみ発生します。
* **実行手順**:
  1. スクリプト冒頭の個人設定を編集（`INPUT_CIF_PATTERN`, `MODEL_PT`, `ALLOW_CELL_DEFORMATION`, `OPTIMIZER_CLASS`）。
  2. 実行：
     ```bash
     python LAMMPS_MACE_Structure_optimizer.py
     ```
  3. 結果：`./output/` 内に `*_relaxed.cif` が保存され、`result.csv` に最終エネルギーと収束ステップ数が記録されます。

### 4.2 新規 NVT-MD 計算（段階昇温 ＋ 本計算）
* **使用スクリプト**: `LAMMPS_MACE_MD_calculator.py`
* **推奨モデル**: `mace_mp_medium_fp32.model-lammps.pt`（単精度）
* **実行手順**:
  1. スクリプト冒頭の個人設定を編集（`CIF_FILE`, `FILENAME`, `HEATING_SCHEDULE`, `STEPS_PER_TEMP`, `MAIN_STEPS`, `MODEL_PT`）。
  2. 実行：
     ```bash
     python LAMMPS_MACE_MD_calculator.py
     ```
  3. 結果：各温度の昇温トラジェクトリ（`*_heating_*.traj`）、本計算トラジェクトリ（`*_main_*.traj`）、MD ログ（`*_MD.log`）、進行状況ログ（`*_process.txt`）が出力されます。

### 4.3 継続 MD 計算（リスタート）
* **使用スクリプト**: `LAMMPS_MACE_MD_successer.py`
* **実行手順**:
  1. 前回の `.traj` ファイル（`TRAJ_FILE`）と目標温度（`HEATING_TEMPERATURE`）、追加ステップ数（`MAIN_STEPS`）を設定。
  2. 実行：
     ```bash
     python LAMMPS_MACE_MD_successer.py
     ```
  3. 結果：指定した `.traj` ファイルの最終フレームから計算が再開され、同ファイルに続きが追記されます。

### 4.4 純粋な LAMMPS コマンド実行（大規模系・Python オーバーヘッド完全排除）
400 原子以上のスーパーセルで長時間の MD を回す場合や、NPT アンサンブル等の LAMMPS 固有機能を使う場合は、Python を介さず `lmp` コマンドで直接実行します。

```bash
lmp -in in.lammps
```
*(※ 入力ファイル内で `pair_style mace no_domain_decomposition` および `pair_coeff * * モデルパス.model-lammps.pt 元素1 元素2 ...` を指定します)*

---

## 5. 実測ベンチマークと環境使い分けの指針

RTX 3060 Ti (VRAM 8 GB) 環境での実測比較結果です。

| 計算内容・条件 | 使用環境 / 手法 | 処理速度・結果 |
| :--- | :--- | :--- |
| **Si 64 原子 MD (20 steps)** | LAMMPS (`ML-MACE`, **fp64**) | 4.86 秒（約 243 ms/step） |
| **Si 64 原子 MD (20 steps)** | LAMMPS (`ML-MACE`, **fp32**) | **1.93 秒（約 96.6 ms/step / 約 2.5 倍高速）** |
| **Sillen 系 72 原子 構造最適化** | CHGNet | 約 10 秒 / model（約 150 step で収束） |
| **Sillen 系 72 原子 構造最適化** | MACE (fp64) + **cuEquivariance** (`mace_env`) | **約 4 秒 / model（約 50 step で収束）** |
| **Sillen 系 72 原子 構造最適化** | MACE (fp64) + **LAMMPS** (`lammps_mace_env`) | 約 16 秒 / model（モデル毎回ロード時、約 50 step） |

### おすすめの使い分け
1. **普段の構造最適化・数十〜数百原子の ASE ベース MD**
   * **`mace_env`（`cuEquivariance` 有効版）** が最速です。特に多元素系（6元素など）のテンソル積演算で圧倒的な速度が出ます。
2. **研究室スクリプトのそのまま実行・純粋な LAMMPS 入力 (`in.lammps`) での大規模 MD**
   * **`lammps_mace_env`（本マニュアルの `LAMMPS (ML-MACE)` 環境）** を使用します。

---

## 6. 運用上の重要ルールとトラブルシューティング

### ① 計算用 Python スクリプトに `import torch` / `import mace` を書かない
`LAMMPSlib` を使うスクリプト内で Python 版の `torch`（CUDA 13.x）をインポートすると、LAMMPS 側の C++ 版 `LibTorch`（CUDA 12.6）とメモリ上で衝突し、**`Segmentation fault (Signal 11)`** で強制終了します。
* **対策**: モデルの変換（`mace_create_lammps_model`）は必ず事前にコマンドラインで別プロセスとして実行し、計算用 `.py` ファイルには `ase` / `pymatgen` / `pandas` のみを記述してください。

### ② RTX 3060 Ti における FP32 と FP64 の使い分け
コンシューマー向け GPU（GeForce 系）はハードウェア仕様上、FP64（倍精度）の演算速度が FP32（単精度）の 1/64 に制限されています。
* **MD 計算・候補構造のスクリーニング（`fmax=0.01` 程度）**: **`fp32` モデル**を使用してください（エネルギーは小数点以下 7 桁目まで一致し、2.5 倍以上高速です）。
* **精密な最終緩和（`fmax=0.001` 以下やフォノン計算前）**: `fp32` で大まかに緩和した後の構造に対してのみ **`fp64` モデル**で仕上げを行うと最も効率的です。

### ③ Windows ⇔ Ubuntu 間のファイル確認・受け渡し
* Ubuntu ターミナルの作業フォルダで以下を実行すると、その場所を Windows のエクスプローラーで直接開けます（VESTA 等での `.cif` 確認に便利です）。
  ```bash
  explorer.exe .
  ```
* Windows からファイルをコピーした際に自動生成される `*:Zone.Identifier` ファイルは不要なため、以下のコマンドで一括削除できます。
  ```bash
  rm -f *:Zone.Identifier
  ```