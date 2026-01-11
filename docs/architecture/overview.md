![Pipeline Flow Diagram](../images/pipeline_flow.png)

# システム概要

衣装リターゲットシステム。素体の骨格に合わせて衣装モデルを自動調整。

関連: [README](../README.md) | [シーケンス](sequence.md) | [依存関係](dependencies.md)

---

## モジュール構成

```mermaid
graph TB
    subgraph "Entry"
        Main["main.py"]
    end

    subgraph "Application"
        Retargeter["retargeter.py"]
        Batch["batch.py"]
    end

    subgraph "Domain"
        Models["models.py"]
    end

    subgraph "Infrastructure (Blender)"
        BlenderOps["ops.py"]
        MeshOps["mesh.py"]
        Deformation["deformation.py"]
        Armature["armature.py"]
        Weights["weights.py"]
        Geometry["geometry.py"]
        Blendshapes["blendshapes.py"]
    end

    Main --> Retargeter
    Batch --> Retargeter
    Retargeter --> Models
    Retargeter --> Infrastructure
    Deformation --> Geometry
    Deformation --> MeshOps
    Deformation --> Armature
    Deformation --> BlenderOps
```

---

## 実行フロー

```mermaid
flowchart TB
    Input["入力FBX"] --> Phase1
    
    subgraph Phase1["Phase 1: 初期化 (Initialization)"]
        A1["_step_initialize_scene"] --> A2["_step_process_avatars"]
    end
    
    Phase1 --> Phase2
    
    subgraph Phase2["Phase 2: 安定化 (Stabilization)"]
        B1["_step_apply_initial_pose"] --> B2["_step_setup_clothing"]
        B2 --> B3["_step_transfer_weights"]
    end
    
    Phase2 --> Phase3
    
    subgraph Phase3["Phase 3: 分類・変形 (Selection/Deformation)"]
        C1["_step_process_meshes"] --> C2["OBB干渉判定/分類"]
        C2 --> C3["Flexible/Rigid変形適用"]
    end
    
    Phase3 --> Phase4
    
    subgraph Phase4["Phase 4: 最終化 (Finalization)"]
        D1["_step_finalize"] --> D2["FBXエクスポート"]
    end
    
    Phase4 --> Output["出力FBX"]
```

---

## モジュール責務

| レイヤー | モジュール | 責務 | 主要関数 |
|---------|-----------|------|---------|
| Application | `retargeter.py` | パイプライン制御・ユースケース | `execute()`, `_run_pipeline()` |
| Application | `batch.py` | バッチ処理実行 | `main()` |
| Domain | `models.py` | データモデル・設定定義 | `RetargetConfig`, `RetargetContext` |
| Infra | `weights.py` | ウェイト転送・統合・波及 | `update_base_avatar_weights` |
| Infra | `mesh.py` | メッシュ加工・成分分離 | `cleanup_mesh`, `separate_and_combine_components` |
| Infra | `armature.py` | ボーン調整・逆ポーズ計算 | `calculate_inverse_pose_matrix` |
| Infra | `deformation.py` | リターゲット中核ロジック | `retarget_mesh`, `process_field_deformation` |
| Infra | `geometry.py` | RBF変形・相似変換計算 | `batch_process_vertices_multi_step` |
| Infra | `blendshapes.py` | BlendShape同期・正規化 | `sync_shape_key_names_from_file` |
| Infra | `ops.py` | Blender基本操作ラッパー | `apply_y_rotation_to_bone` |
