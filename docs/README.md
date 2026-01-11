# ドキュメント

衣装リターゲットシステムの技術ドキュメント。

---

## 構成

```
docs/
├── [architecture/](architecture/overview.md)     # システム構造
│   ├── [overview.md](architecture/overview.md)     # モジュール構成 + 処理フロー
│   ├── [sequence.md](architecture/sequence.md)     # シーケンス図
│   ├── [dependencies.md](architecture/dependencies.md) # 依存関係
│   ├── [data_flow.md](architecture/data_flow.md)    # データフロー
│   └── [bones.md](architecture/bones.md)        # ボーン階層
├── [math/](math/geometry.md)             # 数学
│   ├── [geometry.md](math/geometry.md)     # OBB, SVD, RBF
│   ├── [weights.md](math/weights.md)      # ウェイト計算
│   ├── [transforms.md](math/transforms.md)   # 座標変換
│   └── [smoothing.md](math/smoothing.md)    # スムージング
└── [note/](note/project_concept.md)             # 読み物
    ├── [project_concept.md](note/project_concept.md)  # コンセプト
    ├── [rbf_intuition.md](note/rbf_intuition.md)    # RBFって何？（初心者向け）
    ├── [obb_intuition.md](note/obb_intuition.md)    # 自動仕分けの魔法（OBB）
    ├── [weights_intuition.md](note/weights_intuition.md) # 透明な糸の話（ウェイト）
    ├── [shapekeys_intuition.md](note/shapekeys_intuition.md) # 服が痩せる仕組み（シェイプキー）
    ├── [smoothing_intuition.md](note/smoothing_intuition.md) # ギザギザを消す技術（スムージング）
    ├── [consolidation_intuition.md](note/consolidation_intuition.md) # 靴がくしゃっとならない理由（ウェイト統合）
    └── [batch_intuition.md](note/batch_intuition.md)  # 寝ている間に終わる（バッチ処理）
```

---

## 推奨読み順

```mermaid
flowchart LR
    A["overview"] --> B["sequence"]
    B --> C["math/geometry"]
```

1. [overview.md](architecture/overview.md) - システム全体像
2. [sequence.md](architecture/sequence.md) - 処理の流れ
3. [geometry.md](math/geometry.md) - 内部アルゴリズム

---

## モジュール対応表

| モジュール | ドキュメント | 役割 |
|-------------|-------------|------|
| `retargeter.py` (App) | [overview](architecture/overview.md) | パイプライン制御 |
| `deformation.py` (Infra) | [geometry](math/geometry.md) | 品質パリティの中核 (SVD/RBF) |
| `weights.py` (Infra) | [weights](math/weights.md) | ウェイト伝播・正規化 |
| `armature.py` (Infra) | [transforms](math/transforms.md) | 座標空間逆変換 |
| `geometry.py` (Infra) | [geometry](math/geometry.md) | 数学的アルゴリズム |
| `mesh.py` (Infra) | [overview](architecture/overview.md) | メッシュクリーンアップ・属性転送 |
