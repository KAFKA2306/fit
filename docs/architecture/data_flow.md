# データフロー図

関連: [README](../README.md) | [シーケンス](sequence.md) | [ボーン](bones.md)

リターゲット処理におけるデータの変換と流れを示します。

```mermaid
flowchart LR
    subgraph Input["入力"]
        I1[("服FBX")]
        I2[("ベースFBX")]
        I3[("設定データ (RetargetConfig)")]
        I4[("フィールドデータ (.npz)")]
    end

    subgraph "RetargetContext (State Management)"
        CTX["Context: config, cache, avatars"]
    end

    subgraph Pipeline["処理パイプライン"]
        P1["_step_process_avatars"]
        P2["_step_setup_clothing"]
        P3["_step_transfer_weights"]
        P4["_step_process_meshes"]
    end

    subgraph Output["出力"]
        O1[("リターゲット済FBX")]
    end

    I1 & I2 & I3 & I4 --> CTX
    CTX --> P1 --> P2 --> P3 --> P4 --> O1
```

## 設定データ構造

```mermaid
classDiagram
    class RetargetConfig {
        +str input_fbx
        +str output_fbx
        +str base_fbx
        +str base_avatar_data
        +str field_data
        +str shape_name_file
        +bool no_subdivision
        +bool no_triangle
    }

    class RetargetContext {
        +RetargetConfig config
        +dict cache
        +Object base_mesh
        +Object clothing_armature
        +list clothing_meshes
    }

    class AvatarData {
        +str meshName
        +list~BoneMapping~ humanoidBones
        +dict boneHierarchy
        +list~AuxBoneGroup~ auxiliaryBones
    }

    class BoneMapping {
        +str humanoidBoneName
        +str boneName
    }

    RetargetConfig --> AvatarData : references
    RetargetContext --> RetargetConfig : manages
    RetargetContext --> AvatarData : stores
```

## ウェイトデータ変換

```mermaid
flowchart TD
    subgraph "A: 検証 (fix_invalid_weights)"
        Step1["元の頂点ウェイト"] --> Decision{"NaN/Inf検出?"}
        Decision -->|Yes| Step2["隣接頂点から補間"]
        Decision -->|No| Step3["そのまま"]
    end
    
    subgraph "B: 伝播 (propagate_bone_weights)"
        P1["全0ウェイト頂点?"] -->|Yes| P2["隣接点からコピー"]
        P1 -->|No| P3["そのまま"]
    end
    
    Step2 --> P1
    Step3 --> P1
    P2 --> F["合計=1.0に正規化"]
    P3 --> F
    F --> G["ボーングループ割当"]
```

---

## ナビゲーション
- [ドキュメント目次](../README.md)
- [システム概要 (Overview)](overview.md)
- [処理シーケンス (Sequence)](sequence.md)
