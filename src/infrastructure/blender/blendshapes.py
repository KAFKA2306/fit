import json

import bpy


def load_avatar_data_for_blendshape_analysis(avatar_data_path: str) -> dict:
    with open(avatar_data_path, encoding="utf-8") as f:
        return json.load(f)


def get_blendshape_groups(avatar_data: dict) -> dict:
    groups = {}
    blend_shape_groups = avatar_data.get("blendShapeGroups", [])
    for group in blend_shape_groups:
        group_name = group.get("name", "")
        blend_shape_fields = group.get("blendShapeFields", [])
        groups[group_name] = blend_shape_fields
    return groups


def get_deformation_fields_mapping(avatar_data: dict) -> tuple:
    blend_shape_fields = {}
    inverted_fields = {}
    for field in avatar_data.get("blendShapeFields", []):
        label = field.get("label", "")
        if label:
            blend_shape_fields[label] = field
    for field in avatar_data.get("invertedBlendShapeFields", []):
        label = field.get("label", "")
        if label:
            inverted_fields[label] = field
    return blend_shape_fields, inverted_fields


def reset_shape_keys(obj: bpy.types.Object) -> None:
    if obj.data.shape_keys is not None:
        for kb in obj.data.shape_keys.key_blocks:
            if kb.name != "Basis":
                kb.value = 0.0


def sync_shape_key_names_from_file(clothing_meshes: list, shape_name_filepath: str) -> None:
    with open(shape_name_filepath, encoding="utf-8") as f:
        shape_name_data = json.load(f)
    for mesh_obj in clothing_meshes:
        if mesh_obj.type != "MESH":
            continue
        mesh_name = mesh_obj.name
        if mesh_name not in shape_name_data:
            continue
        file_shape_names = shape_name_data[mesh_name]
        if not mesh_obj.data.shape_keys or not mesh_obj.data.shape_keys.key_blocks:
            continue
        shape_keys = mesh_obj.data.shape_keys.key_blocks
        current_shape_names = {sk.name for sk in shape_keys}
        for file_shape_name in file_shape_names:
            if file_shape_name == "Basis":
                continue
            if file_shape_name in current_shape_names:
                continue
            if "." in file_shape_name:
                after_period = file_shape_name.split(".", 1)[1]
                for shape_key in shape_keys:
                    if shape_key.name == "Basis":
                        continue
                    if shape_key.name == after_period:
                        shape_key.name = file_shape_name
                        break


def process_blendshape_transitions(current_config: dict, next_config: dict) -> None:
    blendshape_settings = next_config["config_data"].get("sourceBlendShapeSettings", [])
    current_config["next_blendshape_settings"] = blendshape_settings
    current_base_avatar_data = load_avatar_data_for_blendshape_analysis(current_config["base_avatar_data"])
    _blend_shape_groups = get_blendshape_groups(current_base_avatar_data)
    _blend_shape_fields, _inverted_blend_shape_fields = get_deformation_fields_mapping(current_base_avatar_data)
    all_transition_sets = []
    all_default_transition_sets = []
    all_target_settings = {}
    current_target_settings = current_config["config_data"].get("targetBlendShapeSettings", [])
    all_target_settings["Basis"] = current_target_settings
    current_blend_shape_fields = current_config["config_data"].get("blendShapeFields", [])
    for field in current_blend_shape_fields:
        field_label = field.get("label", "")
        field_target_settings = field.get("targetBlendShapeSettings", [])
        all_target_settings[field_label] = field_target_settings
    current_config["config_data"]["blend_shape_transition_sets"] = all_transition_sets
    current_config["config_data"]["blend_shape_default_transition_sets"] = all_default_transition_sets
