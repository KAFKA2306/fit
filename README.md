# Outfit Retargeting System

Blenderにおいて素体メッシュに合わせて衣装モデルを自動的に調整およびフィッティングする高度な自動リターゲットシステム。

---

## 主な機能

- 自動フィッティング: OBB（指向性境界箱）およびSVD（特異値分解）を用いて衣装を素体に対して最適に配置。
- 高度な変形: RBF（放射基底関数）による弾性補間を用い、自然な形状変形を実現。
- ウェイト転送: 素体のボーンウェイトを衣装へ自動転送し、動作時の形状破綻を防止。
- メッシュ最適化: エッジの自動分割により、変形時の視覚的な品質を維持。
- シェイプキー同期: 素体の体型変化に衣装のシェイプキーを完全追従。
- システム連携: 標準出力による進捗表示およびバッファリング対策を実施。
- クラッシュ対策: 異常終了時に現状のシーンを _error.blend として自動保存。

---

## 最小要件

- Blender 4.0 以上

---

## セットアップ

```bash
task sync
```

---

## 実行方法

```bash
task run
```

---

## プロジェクト構造

### コアロジック

#### アプリケーション層 (src/application/)
- retargeter.py: パイプライン制御およびユースケースの実装
- batch.py: バッチ処理のエントリポイント

#### ドメイン層 (src/domain/)
- models.py: データモデルおよび設定の定義

#### インフラストラクチャ層 (src/infrastructure/)
- blender/: Blender依存の具体実装
  - ops.py: 基本的な操作
  - mesh.py: メッシュ編集およびクリーンアップ
  - armature.py: アーマチュアおよびボーン操作
  - weights.py: ウェイト計算および転送
  - geometry.py: 数学的計算（SVD, RBF）
  - deformation.py: メッシュ変形ロジック
  - blendshapes.py: シェイプキー制御

### ドキュメント (docs/)
- [ドキュメント目次](file:///home/kafka/projects/fit/docs/README.md)
- [Architecture](file:///home/kafka/projects/fit/docs/architecture/overview.md): システム構成およびフロー
- [Math Guide](file:///home/kafka/projects/fit/docs/math/geometry.md): 数学アルゴリズムの解説
- [直感シリーズ (Intuition Series)](file:///home/kafka/projects/fit/docs/note/project_concept.md): 各機能の直感的解説

---

## ライセンス

本プロジェクトは GNU General Public License v3.0 (GPLv3) の下で公開されています。
詳細は LICENSE ファイルを参照してください。

---

## 開発

### 静的解析およびフォーマット

```bash
task check
```
