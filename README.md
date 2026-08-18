# fit — Blender衣装リターゲット研究基盤

**リポジトリ:** https://github.com/KAFKA2306/fit

Blender上で、衣装メッシュを別の素体へ配置・変形し、ウェイトやShape Keyの転送を補助するバッチ型リターゲットシステムです。

OBB、SVD、RBFなどの幾何処理を使った候補生成を行いますが、自動処理だけで身体への完全なフィット、貫通のなさ、販売品質、VRChat互換性を保証するものではありません。

## 主な処理

- OBBとSVDによる初期位置合わせ
- RBFを使った形状変形の試作
- アバターから衣装へのウェイト転送
- メッシュの分割・整理
- Shape Key同期の補助
- バッチ実行と進捗ログ
- 異常終了時のシーン保存

## 必要環境

```text
Blender 4.0以上
Python 3.11以上
uv
Task
```

実際に検証したBlender版は実行ログへ保存してください。Blender Python APIやモディファイア挙動は版によって変わります。

## セットアップ

```bash
uv sync --locked
```

## 実行

```bash
task run
```

タスクが入力・出力するファイル、Blender実行パス、設定値は`Taskfile.yml`を確認してください。

## 処理の流れ

```text
素体と衣装を読み込む
  → スケール・座標系・基準ポーズを検証
  → 初期位置合わせ
  → 形状変形
  → メッシュ整理
  → ウェイト転送
  → Shape Key候補を生成
  → 監査結果と.blendを保存
  → 人間が外観・ポーズを確認
```

## 主な構成

```text
src/
├── application/
│   ├── retargeter.py      # パイプライン制御
│   └── batch.py           # バッチ入口
├── domain/
│   └── models.py          # 設定・データモデル
└── infrastructure/
    └── blender/
        ├── ops.py
        ├── mesh.py
        ├── armature.py
        ├── weights.py
        ├── geometry.py
        ├── deformation.py
        └── blendshapes.py
docs/
```

ドキュメント:

- [ドキュメント目次](docs/README.md)
- [アーキテクチャ](docs/architecture/overview.md)
- [幾何処理](docs/math/geometry.md)

## 変形前に確認すること

- 素体と衣装の単位・スケール
- Armature Transform
- 基準ポーズ
- 左右・前後・上下の座標系
- 適用済みモディファイア
- 頂点数とトポロジ
- 衣装の厚みと身体からの距離

## 変形後の監査

- 非有限座標、重複頂点、ゼロ面積面
- 法線と裏面
- UVの破綻
- 身体との貫通
- ウェイト合計と最大ボーン影響数
- 肩、脇、股、膝、肘の変形
- Shape Key間の干渉
- FBX書出し・再読込
- Unity・VRChatでの実動作

## エラー保存

異常終了時に`_error.blend`を保存する機能がある場合でも、必ず保存に成功するとは限りません。元ファイルを直接上書きせず、作業コピーとGit管理外の出力先を使ってください。

## 静的検査

```bash
task check
```

`task check`はファイルを書き換えず、Ruffのlintとformat検査を実行します。静的検査の成功は、Blenderでの見た目やUnity互換性を証明しません。

## ライセンス

[GNU General Public License v3.0](LICENSE)

第三者アバター・衣装・テクスチャの権利は、このコードのライセンスとは別です。
