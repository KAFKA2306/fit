# ドキュメント

衣装リターゲットシステムの技術ドキュメント。

## 構成

```text
docs/
├── architecture/                  # システム構造
│   ├── overview.md                # モジュール構成 + 処理フロー
│   ├── sequence.md                # シーケンス図
│   ├── dependencies.md            # 依存関係
│   ├── data_flow.md               # データフロー
│   └── bones.md                   # ボーン階層
├── math/                          # 数学
│   ├── geometry.md                # OBB, SVD, RBF
│   ├── weights.md                 # ウェイト計算
│   ├── transforms.md              # 座標変換
│   └── smoothing.md               # スムージング
├── note/                          # 補足説明
└── spec/
    └── req.md                     # 要件
```

## 推奨読み順

1. [overview.md](architecture/overview.md) - システム全体像
2. [sequence.md](architecture/sequence.md) - 処理の流れ
3. [geometry.md](math/geometry.md) - 内部アルゴリズム

## モジュール対応表

| モジュール | ドキュメント | 役割 |
|-------------|-------------|------|
| `retargeter.py` (App) | [overview](architecture/overview.md) | パイプライン制御 |
| `deformation.py` (Infra) | [geometry](math/geometry.md) | SVD/RBF変形 |
| `weights.py` (Infra) | [weights](math/weights.md) | ウェイト伝播・正規化 |
| `armature.py` (Infra) | [transforms](math/transforms.md) | 座標空間逆変換 |
| `geometry.py` (Infra) | [geometry](math/geometry.md) | 数学的アルゴリズム |
| `mesh.py` (Infra) | [overview](architecture/overview.md) | メッシュクリーンアップ・属性転送 |
