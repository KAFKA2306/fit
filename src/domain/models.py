from typing import Any

from pydantic import BaseModel, ConfigDict

LEG_FOOT_CHEST_BONES = {
    "RightUpperLeg",
    "LeftLowerLeg",
    "RightLowerLeg",
    "RightFoot",
    "LeftToes",
    "RightToes",
    "LeftBreast",
    "RightBreast",
    "LeftFootThumbIntermediate",
    "LeftFootThumbDistal",
    "LeftFootIndexIntermediate",
    "LeftFootIndexDistal",
    "LeftFootMiddleIntermediate",
    "LeftFootMiddleDistal",
    "LeftFootRingIntermediate",
    "LeftFootRingDistal",
    "LeftFootLittleIntermediate",
    "LeftFootLittleDistal",
    "RightFootThumbIntermediate",
    "RightFootThumbDistal",
    "RightFootIndexIntermediate",
    "RightFootIndexDistal",
    "RightFootMiddleIntermediate",
    "RightFootMiddleDistal",
    "RightFootRingIntermediate",
    "RightFootRingDistal",
    "RightFootLittleIntermediate",
    "RightFootLittleDistal",
}

RIGHT_GROUP_FINGERS = {
    "LeftThumbIntermediate",
    "LeftThumbDistal",
    "LeftMiddleIntermediate",
    "LeftMiddleDistal",
    "LeftLittleIntermediate",
    "LeftLittleDistal",
    "RightThumbIntermediate",
    "RightThumbDistal",
    "RightMiddleIntermediate",
    "RightMiddleDistal",
    "RightLittleIntermediate",
    "RightLittleDistal",
}

LEFT_GROUP_FINGERS = {
    "LeftIndexIntermediate",
    "LeftIndexDistal",
    "LeftRingIntermediate",
    "LeftRingDistal",
    "RightIndexIntermediate",
    "RightIndexDistal",
    "RightRingIntermediate",
    "RightRingDistal",
}

EXCLUDED_BONES = {"RightShoulder", "LeftUpperArm", "RightUpperArm", "RightLowerArm", "LeftHand", "RightHand"}
IGNORED_BONES = {"Head"}


class RetargetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_fbx: str
    input_fbx: str
    output_fbx: str
    config_path: str
    init_pose: str | None = None
    config_data: dict[str, Any] | None = None
    base_avatar_data: str | None = None
    base_avatar_data_content: dict[str, Any] | None = None
    hips_position: Any = None
    target_meshes: str | None = None
    mesh_renderers: str | None = None
    blend_shapes: str | None = None
    blend_shape_values: str | None = None
    blend_shape_mappings: str | None = None
    name_conv: str | None = None
    cloth_metadata: str | None = None
    mesh_material_data: str | None = None
    shape_name_file: str | None = None
    no_subdivision: bool = False
    no_triangle: bool = False
    field_data: str | None = None


class BatchConfig(BaseModel):
    blender_exe: str = "blender"
    input_dir: str = "Assets/HB_shop"
    output_dir: str = "Assets/OutfitRetargetingSystem/Outputs"
    base_fbx: str = "Assets/OutfitRetargetingSystem/Editor/Template.fbx"
    config_path: str = "Assets/OutfitRetargetingSystem/Editor/config_shinano2template.json"
    init_pose: str = "Assets/OutfitRetargetingSystem/Editor/pose_basis_template.json"


class RetargetContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: RetargetConfig
    base_mesh: Any = None
    base_armature: Any = None
    base_avatar_data: dict[str, Any] = {}
    clothing_meshes: list[Any] = []
    clothing_armature: Any = None
    clothing_avatar_data: dict[str, Any] = {}
    name_conv_data: dict[str, Any] | None = None
    pair_index: int = 0
    total_pairs: int = 1
    cache: dict[str, Any] = {}
