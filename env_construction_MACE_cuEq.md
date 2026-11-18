# Windows 11 + WSL2 における MACE-MP-0 (medium) + cuEquivariance 環境構築・運用マニュアル

本ドキュメントは、Windows 11（NVIDIA GeForce RTX 3060 Ti 搭載機）の **WSL2 (Ubuntu)** 上に、NVIDIA の高速化カーネル **cuEquivariance** を有効化した **MACE-MP-0 (`medium`)** の計算環境を構築した記録と、日常の運用・別フォルダでの利用方法をまとめたものです。

---

## 1. 構築済み環境スペック（2026年11月17日 構築）

| 項目 | バージョン / パス / 仕様 | 備考 |
| :--- | :--- | :--- |
| **ホスト OS** | Windows 11 | NVIDIA Driver `621.14` (CUDA UMD `13.5`) |
| **GPU** | NVIDIA GeForce RTX 3060 Ti | VRAM 8GB (Ampere / SM86) |
| **WSL2 環境** | Ubuntu (WSL2) | ユーザー名: `ancient` / ホスト名: `DESKTOP-OWL` |
| **Python 仮想環境** | `~/mace_env` (`/home/ancient/mace_env`) | Python `3.14.4` (`venv`) |
| **PyTorch** | `2.15.0+cu132` | CUDA `13.2` 対応ビルド |
| **MACE** | `mace-torch 0.3.17` | `e3nn 0.4.4` |
| **cuEquivariance** | `0.13.0` | `cuequivariance`, `cuequivariance-torch`, `cuequivariance-ops-torch-cu13` |
| **原子シミュレーション関連** | `ase 3.29.0`, `pymatgen 2026.10.24`, `pandas 3.1.1`, `numpy 2.5.5` | |
| **モデルキャッシュ** | `~/.cache/mace/20231203mace128L1_epoch199model` | MACE-MP-0 (`medium`) 初回ダウンロード済み (42.4 MB) |
| **初期作業フォルダ** | `C:\MySetting\MACE` | WSL2 からは `/mnt/c/MySetting/MACE` としてアクセス |

---

## 2. 環境構築手順（再現用ログ）

将来別 PC への移行や環境再構築を行う際の手順です。`cuEquivariance` は Linux 専用パッケージのため、Windows ネイティブではなく WSL2 内に構築しています。

### Step 1: WSL2 と Ubuntu のインストール（Windows PowerShell）
1. **管理者として開いた PowerShell** で以下を実行し、Windows を再起動します。
   ```powershell
   wsl --install -d Ubuntu
   ```
2. 再起動後、通常の PowerShell で再度以下を実行して Ubuntu 本体を導入し、UNIX ユーザー名（例: `ancient`）とパスワードを設定します。
   ```powershell
   wsl --install -d Ubuntu
   ```
3. Ubuntu ターミナル内で GPU が認識されているか確認します。
   ```bash
   nvidia-smi
   ```

### Step 2: Python 仮想環境と PyTorch の導入（WSL2 Ubuntu）
Windows 側フォルダ（`/mnt/c/...`）ではなく、I/O が高速な Linux 側ホームディレクトリ（`~/mace_env`）に仮想環境を作成します。

```bash
# 仮想環境作成ツールの導入
sudo apt update && sudo apt install -y python3-venv python3-pip

# 仮想環境の作成と有効化
python3 -m venv ~/mace_env
source ~/mace_env/bin/activate

# pip 更新と CUDA 対応 PyTorch のインストール
pip install --upgrade pip
pip install torch

# GPU 認識テスト
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda, '| GPU:', torch.cuda.get_device_name(0))"
```

### Step 3: MACE と cuEquivariance の導入（WSL2 Ubuntu）
`(mace_env)` が有効な状態で以下を実行します。

```bash
# MACE および関連ライブラリのインストール
pip install mace-torch ase pymatgen pandas

# cuEquivariance (CUDA 13 カーネル) のインストール
# ※ CUDA 12 環境の PyTorch を入れた場合は末尾を -cu12 に変更する
pip install cuequivariance cuequivariance-torch cuequivariance-ops-torch-cu13

# インポート確認
python -c "import mace, cuequivariance, cuequivariance_torch; print('MACE:', mace.__version__, '| cuEq:', cuequivariance.__version__)"
```

---

## 3. 作成済みスクリプト一覧（`C:\MySetting\MACE`）

従来の CHGNet 用スクリプトと同等の入出力・個人設定項目を持つ MACE-MP-0 (`medium`) 版スクリプトです。

1. **`MACE_install_tester.py`**
   * Python / PyTorch / GPU / MACE / cuEquivariance のバージョン確認、および 64 原子 Si 結晶を用いた推論テスト・速度計測を行います。
2. **`MACE_Structure_optimizer.py`**
   * `target/*.cif` を一括で構造最適化し、`./output/*_relaxed.cif` と `result.csv` を出力します。
   * 力の収束精度（`fmax=0.01`）を確保するため **`default_dtype="float64"`** を使用します。
   * `ALLOW_CELL_DEFORMATION = True` にすると `FrechetCellFilter` による格子定数緩和も同時に行います。
3. **`MACE_MD_calculator.py`**
   * `CIF_FILE` を読み込み、`HEATING_SCHEDULE` に従った段階的昇温（`NoseHooverChainNVT`）の後、最終温度で本計算 MD を実行します。
   * 計算速度を重視し **`default_dtype="float32"`** かつ **`enable_cueq=True`** を設定しています（RTX 3060 Ti で約 30 ms/step 前後で動作）。
4. **`MACE_MD_successer.py`**
   * 既存のトラジェクトリファイル（`TRAJ_FILE`）の最終フレーム（`index=-1`）を読み込み、追記モード（`'a'`）で継続 NVT-MD 計算を行います。

---

## 4. 普段の実行方法（日常ルーチン）

PC を再起動したあとや、新しくターミナルを開いて計算を実行する際の手順です。

### 基本手順
1. Windows で **PowerShell**（または Windows ターミナル）を開き、Ubuntu を起動します。
   ```powershell
   wsl -d Ubuntu
   ```
2. Python 仮想環境を有効化し、作業フォルダへ移動します。
   ```bash
   source ~/mace_env/bin/activate
   cd /mnt/c/MySetting/MACE
   ```
   > **パスの変換ルール:** Windows の `C:\...` は WSL2 では `/mnt/c/...`、D ドライブの `D:\...` は `/mnt/d/...` に対応します。

3. 実行したいスクリプトを動かします。
   ```bash
   python MACE_Structure_optimizer.py
   # または
   python MACE_MD_calculator.py
   ```
   ※ 冒頭の `FutureWarning` も非表示にしたい場合は `-W ignore::FutureWarning` を付けて実行します：
   ```bash
   python -W ignore::FutureWarning MACE_MD_calculator.py
   ```

### 【便利設定】コマンド1つで仮想環境を有効化する
Ubuntu の設定ファイル（`~/.bashrc`）に短縮コマンド（エイリアス）を登録しておくと、次回から `mace` と打つだけで仮想環境が有効になります。

```bash
echo "alias mace='source ~/mace_env/bin/activate'" >> ~/.bashrc
source ~/.bashrc
```
登録後は、Ubuntu 起動後に `mace` と入力するだけで `(mace_env)` に切り替わります。

---

## 5. 新しいフォルダで MACE を使いたくなった場合

仮想環境（`~/mace_env`）とダウンロード済みの MACE モデル（`~/.cache/mace/`）は WSL2 全体で共有されているため、**パッケージの再インストールやモデルの再ダウンロードは一切不要**です。

用途に応じて以下の 2 通りの方法で運用してください。

### パターン A：Windows 側の新しいフォルダでそのまま使う（通常運用・おすすめ）
構造最適化や数万ステップ規模の MD 計算であれば、Windows 側のフォルダにスクリプトをコピーするだけでそのまま実行できます。

1. **Windows エクスプローラー上の操作**
   * 新しい作業フォルダ（例: `C:\Research\Sample_A` や `D:\Data\Sample_B`）を作成します。
   * `C:\MySetting\MACE` から必要なスクリプト（例: `MACE_MD_calculator.py`）と計算対象の `.cif` ファイルを新しいフォルダへコピーします。
   * スクリプト上部の `#====個人設定====`（`CIF_FILE`, `FILENAME`, 温度条件など）を編集します。
2. **WSL2 での実行**
   ```powershell
   wsl -d Ubuntu
   ```
   ```bash
   source ~/mace_env/bin/activate
   cd /mnt/c/Research/Sample_A
   python MACE_MD_calculator.py
   ```
   ※ D ドライブの場合は `cd /mnt/d/Data/Sample_B` と指定します。計算結果（`.traj`, `.log`, `.csv`, `output/`）はそのまま Windows 側のフォルダに保存されます。

### パターン B：数十万〜数百万ステップの長時間 MD を高速に回す場合
WSL2 から `/mnt/c/...`（Windows ドライブ）へ頻繁に `.traj` を書き込むと、ディスク I/O のオーバーヘッドが発生することがあります。超長時間 MD を行う場合は、WSL2 内の Linux ディスク上で計算し、完了後に Windows 側へコピーすると高速です。

```bash
# 1. 仮想環境を有効化
source ~/mace_env/bin/activate

# 2. WSL2 内に作業フォルダを作成し、Windows 側から CIF とスクリプトをコピー
mkdir -p ~/work/run01
cp /mnt/c/Research/Sample_A/target.cif ~/work/run01/
cp /mnt/c/MySetting/MACE/MACE_MD_calculator.py ~/work/run01/

# 3. WSL2 内で計算を実行
cd ~/work/run01
python MACE_MD_calculator.py

# 4. 計算完了後、結果ファイルを Windows 側フォルダへ戻す
cp *.traj *.log *.txt /mnt/c/Research/Sample_A/
```

---

## 6. 運用上の補足・注意事項

* **起動時に表示される `Failed to get GPU information from pynvml` について**
  * `cuEquivariance` が起動時に GPU クロック等の詳細情報を取得しようとした際に出る WSL2 特有のメッセージです。直後の行で RTX 3060 Ti 用のデフォルト設定が自動適用され、正常に CUDA カーネルが動作するため問題ありません。
* **`default_dtype` の使い分け（VRAM 8GB での推奨設定）**
  * **MD 計算 (`MACE_MD_calculator.py` / `MACE_MD_successer.py`):** `"float32"`（高速・省メモリ）
  * **構造最適化 (`MACE_Structure_optimizer.py`):** `"float64"`（力とエネルギーの収束判定を高精度化）
* **VRAM 不足（Out of Memory）が出た場合**
  * 数百〜千原子を超える非常に大きなスーパーセルを計算して VRAM 8GB を超過した場合は、個人設定の `MODEL_SIZE = "small"` に変更することでメモリ消費を抑えられます。