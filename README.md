![System Overview](docs/images/system_overview.png)

# 👗 Outfit Retargeting System

Blender上で素体メッシュに合わせて衣装モデルを自動的に調整・フィッティングする高度な自動リターゲットシステム。

---

## 🚀 主な機能

- 自動フィッティング: "ただの移動"ではなく、OBB(箱)とSVD(回転)を使って、衣装を素体に「賢く」重ね合わせます。
- 高度な変形: RBF(弾性補間)を使い、まるで布が引っ張られるように自然にフィットさせます。
- ウェイト転送: 素体の動き（ボーンウェイト）を、衣装へ自動的にコピーし、破綻を防ぎます。
- メッシュ最適化: エッジを自動分割し、変形によるカクつきを防ぎます。
- シェイプキー同期: 素体が痩せれば衣装も痩せる。体型変更に完全追従します。
- Unity連携強化: 標準出力によるリアルタイム進捗表示と、`sys.stdout`バッファリング対策済み。
- クラッシュ対策: エラー発生時に、その瞬間のシーンを`_error.blend`として自動保存します。

---

## 🛠 必要環境

- Blender 4.0+


---

## 📦 インストール

```bash
uv sync
```

---

## 📖 使用方法

```bash
# Taskfileを使用して実行
task run
```

---

## 📂 プロジェクト構造

### Core Logic

#### Application Layer (`application/`)
- `retargeter.py` - パイプライン制御・ユースケース
- `batch.py` - バッチ処理エントリポイント

#### Domain Layer (`domain/`)
- `models.py` - データモデル・設定定義

#### Infrastructure Layer (`infrastructure/`)
- `blender/` - Blender依存の具体的な実装
  - `ops.py` - 基本的なBlender操作
  - `mesh.py` - メッシュ編集・クリーンアップ
  - `armature.py` - アーマチュア・ボーン操作
  - `weights.py` - ウェイト転送・計算
  - `geometry.py` - 数学・幾何計算 (SVD, RBF)
  - `deformation.py` - メッシュ変形ロジック
  - `blendshapes.py` - シェイプキーツール



### [Documentation (docs/)](docs/README.md)
- [Architecture](docs/architecture/overview.md) - システム構成・フロー
- [Math Guide](docs/math/geometry.md) - 使用されている数学アルゴリズムの解説

---

## 📜 ライセンス

本プロジェクトは **GNU General Public License v3.0 (GPLv3)** の下で公開されています。
詳細は [LICENSE](LICENSE) ファイルをご確認ください。

---

## 🛠 開発者向け

### リンター & フォーマッタ (Ruff)

```bash
uv run ruff check src/  # チェック
uv run ruff format src/ # 整形
```
