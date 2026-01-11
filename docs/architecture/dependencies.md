# モジュール依存関係図

関連: [README](../README.md) | [overview](overview.md)

```mermaid
graph TD
    subgraph "Entry"
        Main["main.py"]
        Batch["batch.py"]
    end

    subgraph "Application"
        Retargeter["retargeter.py"]
    end

    subgraph "Domain"
        Models["models.py"]
    end

    subgraph "Infrastructure"
        Weights["weights.py"]
        MeshOps["mesh.py"]
        Armature["armature.py"]
        Blendshapes["blendshapes.py"]
        Deformation["deformation.py"]
        BlenderOps["ops.py"]
        Geometry["geometry.py"]
    end

    Main --> Retargeter
    Batch --> Retargeter
    Retargeter --> Models
    Retargeter --> Weights
    Retargeter --> MeshOps
    Retargeter --> Armature
    Retargeter --> Blendshapes
    Retargeter --> Deformation
    Retargeter --> BlenderOps
    
    Deformation --> Geometry
    Deformation --> MeshOps
    Deformation --> Armature
    Deformation --> BlenderOps
    
    Weights --> Armature
    Armature --> MeshOps
```

## インポート関係

| モジュール | 依存先 |
|-----------|-------|
| `weights.py` | `armature` |
| `mesh.py` | `numpy`, `bpy`, `bmesh` |
| `armature.py` | `mesh`, `models` |
| `deformation.py` | `armature`, `ops`, `geometry`, `mesh` |
| `geometry.py` | `numpy`, `scipy`, `mathutils` |
| `retargeter.py` | `armature`, `ops`, `blendshapes`, `deformation`, `mesh`, `weights`, `models` |
| `models.py` | (None) |

---

## ナビゲーション
- [ドキュメント目次](../README.md)
- [システム概要 (Overview)](overview.md)
