import json
import math
import os

import bpy
import mathutils
import numpy as np
from mathutils import Matrix


def get_shallowest_bone(armature: bpy.types.Object, avatar_data: dict = None) -> str:
    if not armature or armature.type != "ARMATURE":
        return None
    bones = armature.data.bones
    if not bones:
        return None
    root_bones = [bone for bone in bones if bone.parent is None]
    if not root_bones:
        return bones[0].name if bones else None
    if len(root_bones) == 1:
        return root_bones[0].name
    if avatar_data:
        hips_bone_name = None
        for bone_map in avatar_data.get("humanoidBones", []):
            if bone_map["humanoidBoneName"] == "Hips":
                hips_bone_name = bone_map["boneName"]
                break
        if hips_bone_name and hips_bone_name in bones:
            hips_bone = bones[hips_bone_name]
            for root_bone in root_bones:
                current = hips_bone
                while current:
                    if current == root_bone:
                        return root_bone.name
                    current = current.parent
    return root_bones[0].name


def record_armature_info(
    clothing_armature: bpy.types.Object, clothing_meshes: list, clothing_avatar_data: dict, record_key: str, cache: dict
) -> None:
    armature_name = clothing_armature.name if clothing_armature else None
    shallowest_bone = get_shallowest_bone(clothing_armature, clothing_avatar_data)
    mesh_names = [mesh.name for mesh in clothing_meshes] if clothing_meshes else []
    if "armature_record" not in cache:
        cache["armature_record"] = {}
    cache["armature_record"][record_key] = {
        "armature_name": armature_name,
        "shallowest_bone": shallowest_bone,
        "mesh_names": mesh_names,
    }


def export_armature_record_to_json(output_path: str, cache: dict) -> None:
    record = cache.get("armature_record")
    if not record:
        return
    json_output_path = output_path.rsplit(".", 1)[0] + "_armature_info.json"
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


def apply_y_rotation_to_bone(armature_obj: bpy.types.Object, bone_name: str, rotation_degrees: float) -> None:
    if bone_name and bone_name in armature_obj.pose.bones:
        bone = armature_obj.pose.bones[bone_name]
        current_world_matrix = armature_obj.matrix_world @ bone.matrix
        head_world_transformed = armature_obj.matrix_world @ bone.head
        offset_matrix = mathutils.Matrix.Translation(head_world_transformed * -1.0)
        rotation_matrix = mathutils.Matrix.Rotation(math.radians(rotation_degrees), 4, "Y")
        bone.matrix = (
            armature_obj.matrix_world.inverted()
            @ offset_matrix.inverted()
            @ rotation_matrix
            @ offset_matrix
            @ current_world_matrix
        )


def apply_y_rotation_to_leg_bones(
    armature: bpy.types.Object, avatar_data: dict, left_rotation_degrees: float, right_rotation_degrees: float
) -> None:
    left_upper_leg_bone = None
    right_upper_leg_bone = None
    for bone_map in avatar_data.get("humanoidBones", []):
        if bone_map.get("humanoidBoneName") == "LeftUpperLeg":
            left_upper_leg_bone = bone_map.get("boneName")
        elif bone_map.get("humanoidBoneName") == "RightUpperLeg":
            right_upper_leg_bone = bone_map.get("boneName")
    apply_y_rotation_to_bone(armature, left_upper_leg_bone, left_rotation_degrees)
    apply_y_rotation_to_bone(armature, right_upper_leg_bone, right_rotation_degrees)


def save_pose_state(armature_obj: bpy.types.Object) -> dict:
    if not armature_obj or armature_obj.type != "ARMATURE":
        return None
    pose_state = {}
    for bone in armature_obj.pose.bones:
        pose_state[bone.name] = {
            "matrix": bone.matrix.copy(),
            "location": bone.location.copy(),
            "rotation_euler": bone.rotation_euler.copy(),
            "rotation_quaternion": bone.rotation_quaternion.copy(),
            "scale": bone.scale.copy(),
        }
    return pose_state


def restore_pose_state(armature_obj: bpy.types.Object, pose_state: dict) -> None:
    if not armature_obj or armature_obj.type != "ARMATURE" or not pose_state:
        return
    for bone_name, state in pose_state.items():
        if bone_name in armature_obj.pose.bones:
            bone = armature_obj.pose.bones[bone_name]
            bone.matrix = state["matrix"]
            bone.location = state["location"]
            bone.rotation_euler = state["rotation_euler"]
            bone.rotation_quaternion = state["rotation_quaternion"]
            bone.scale = state["scale"]
    bpy.context.view_layer.update()


def build_bone_hierarchy(bone_node: dict, bone_parents: dict[str, str], current_path: list):
    bone_name = bone_node["name"]
    for child in bone_node.get("children", []):
        child_name = child["name"]
        bone_parents[child_name] = bone_name
        build_bone_hierarchy(child, bone_parents, [*current_path, bone_name])


def get_humanoid_bone_hierarchy(avatar_data: dict) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    bone_parents = {}
    build_bone_hierarchy(avatar_data["boneHierarchy"], bone_parents, [])
    humanoid_to_bone = {bone_map["humanoidBoneName"]: bone_map["boneName"] for bone_map in avatar_data["humanoidBones"]}
    bone_to_humanoid = {bone_map["boneName"]: bone_map["humanoidBoneName"] for bone_map in avatar_data["humanoidBones"]}
    return bone_parents, humanoid_to_bone, bone_to_humanoid


def apply_pose_from_json(armature_obj: bpy.types.Object, filepath: str, avatar_data: dict) -> None:
    if not armature_obj or armature_obj.type != "ARMATURE":
        return
    if not filepath or not os.path.exists(filepath):
        return
    with open(filepath, encoding="utf-8") as f:
        pose_data = json.load(f)

    bone_parents, humanoid_to_bone, bone_to_humanoid = get_humanoid_bone_hierarchy(avatar_data)

    def get_bone_hierarchy_order():
        order = []
        visited = set()

        def add_bone_and_children(humanoid_bone):
            if humanoid_bone in visited:
                return
            visited.add(humanoid_bone)
            order.append(humanoid_bone)
            for child_bone, parent_bone in bone_parents.items():
                if parent_bone == humanoid_bone and child_bone not in visited:
                    add_bone_and_children(child_bone)

        if "Hips" in humanoid_to_bone:
            add_bone_and_children(humanoid_to_bone["Hips"])
        return order

    bone_order = get_bone_hierarchy_order()
    original_bone_data = {}
    for humanoid_bone, bone_name in humanoid_to_bone.items():
        if bone_name in armature_obj.pose.bones:
            bone = armature_obj.pose.bones[bone_name]
            original_bone_data[humanoid_bone] = {
                "matrix": bone.matrix.copy(),
            }

    for bone_name in bone_order:
        if bone_name not in armature_obj.pose.bones:
            continue
        humanoid_bone = bone_to_humanoid.get(bone_name)
        if not humanoid_bone or humanoid_bone not in pose_data:
            continue

        bone = armature_obj.pose.bones[bone_name]
        original_data = original_bone_data.get(humanoid_bone)
        if not original_data:
            continue

        current_world_matrix = armature_obj.matrix_world @ original_data["matrix"]
        delta_matrix = Matrix(pose_data[humanoid_bone]["delta_matrix"])
        combined_matrix = delta_matrix @ current_world_matrix
        bone.matrix = armature_obj.matrix_world.inverted() @ combined_matrix
        bpy.context.view_layer.update()


def save_shape_key_state(mesh_obj: bpy.types.Object) -> dict:
    if not mesh_obj or not mesh_obj.data.shape_keys:
        return {}
    shape_key_state = {}
    for key_block in mesh_obj.data.shape_keys.key_blocks:
        shape_key_state[key_block.name] = key_block.value
    return shape_key_state


def restore_shape_key_state(mesh_obj: bpy.types.Object, shape_key_state: dict) -> None:
    if not mesh_obj or not mesh_obj.data.shape_keys or not shape_key_state:
        return
    for key_name, value in shape_key_state.items():
        if key_name in mesh_obj.data.shape_keys.key_blocks:
            mesh_obj.data.shape_keys.key_blocks[key_name].value = value


def apply_blend_shape_settings(
    mesh_obj: bpy.types.Object, blend_shape_settings: list, ignore_missing_shape_keys: bool = True
) -> bool:
    if not mesh_obj or not mesh_obj.data.shape_keys or not blend_shape_settings:
        return False
    for setting in blend_shape_settings:
        shape_name = setting.get("name")
        if shape_name not in mesh_obj.data.shape_keys.key_blocks:
            temp_shape_key_name = f"{shape_name}_temp"
            if temp_shape_key_name not in mesh_obj.data.shape_keys.key_blocks:
                if not ignore_missing_shape_keys:
                    print(f"Required shape key does not exist: {shape_name}")
                    return False
    for setting in blend_shape_settings:
        shape_name = setting.get("name")
        shape_value = setting.get("value", 0.0)
        if shape_name in mesh_obj.data.shape_keys.key_blocks:
            mesh_obj.data.shape_keys.key_blocks[shape_name].value = shape_value
        else:
            temp_shape_key_name = f"{shape_name}_temp"
            if temp_shape_key_name in mesh_obj.data.shape_keys.key_blocks:
                mesh_obj.data.shape_keys.key_blocks[temp_shape_key_name].value = shape_value
    return True


def store_pose_globally(armature_obj: bpy.types.Object, cache: dict) -> None:
    cache["global_pose"] = save_pose_state(armature_obj)


def restore_global_pose(armature_obj: bpy.types.Object, cache: dict) -> None:
    pose = cache.get("global_pose")
    if pose:
        restore_pose_state(armature_obj, pose)


def store_current_pose_as_previous(armature_obj: bpy.types.Object, cache: dict) -> None:
    cache["previous_pose"] = save_pose_state(armature_obj)


def restore_previous_pose(armature_obj: bpy.types.Object, cache: dict) -> None:
    pose = cache.get("previous_pose")
    if pose:
        restore_pose_state(armature_obj, pose)


def get_vertex_groups_and_weights(mesh_obj, vertex_index):
    groups = {}
    vertex = mesh_obj.data.vertices[vertex_index]
    for g in vertex.groups:
        group_name = mesh_obj.vertex_groups[g.group].name
        groups[group_name] = g.weight
    return groups


def get_armature_from_modifier(mesh_obj):
    for modifier in mesh_obj.modifiers:
        if modifier.type == "ARMATURE":
            return modifier.object
    return None


def calculate_inverse_pose_matrix(mesh_obj, armature_obj, vertex_index):
    weights = get_vertex_groups_and_weights(mesh_obj, vertex_index)
    if not weights:
        return None
    final_matrix = Matrix.Identity(4)
    final_matrix.zero()
    total_weight = 0
    for bone_name, weight in weights.items():
        if weight > 0 and bone_name in armature_obj.data.bones:
            bone = armature_obj.data.bones[bone_name]
            pose_bone = armature_obj.pose.bones.get(bone_name)
            if bone and pose_bone:
                mat = (
                    armature_obj.matrix_world
                    @ pose_bone.matrix
                    @ bone.matrix_local.inverted()
                    @ armature_obj.matrix_world.inverted()
                )
                final_matrix += mat * weight
                total_weight += weight
    if total_weight > 0:
        final_matrix = final_matrix * (1.0 / total_weight)
    return final_matrix.inverted()


def inverse_bone_deform_all_vertices(armature_obj, mesh_obj):
    vertices = [v.co.copy() for v in mesh_obj.data.vertices]
    inverse_transformed_vertices = []
    for vertex_index in range(len(vertices)):
        pos = vertices[vertex_index]
        weights = get_vertex_groups_and_weights(mesh_obj, vertex_index)
        if not weights:
            inverse_transformed_vertices.append(pos)
            continue
        combined_matrix = Matrix.Identity(4)
        combined_matrix.zero()
        total_weight = 0.0
        for bone_name, weight in weights.items():
            if weight > 0 and bone_name in armature_obj.data.bones:
                bone = armature_obj.data.bones[bone_name]
                pose_bone = armature_obj.pose.bones.get(bone_name)
                if bone and pose_bone:
                    bone_matrix = pose_bone.matrix @ bone.matrix_local.inverted()
                    combined_matrix += bone_matrix * weight
                    total_weight += weight
        if total_weight > 0:
            combined_matrix = combined_matrix * (1.0 / total_weight)
        else:
            combined_matrix = Matrix.Identity(4)
        inverse_matrix = combined_matrix.inverted()
        rest_pose_pos = inverse_matrix @ pos
        inverse_transformed_vertices.append(rest_pose_pos)
    if mesh_obj.data.shape_keys:
        for shape_key in mesh_obj.data.shape_keys.key_blocks:
            if shape_key.name != "Basis":
                for i, vert in enumerate(shape_key.data):
                    vert.co += inverse_transformed_vertices[i] - vertices[i]
        basis_shape_key = mesh_obj.data.shape_keys.key_blocks["Basis"]
        for i, vert in enumerate(basis_shape_key.data):
            vert.co = inverse_transformed_vertices[i]
    for vertex_index, pos in enumerate(inverse_transformed_vertices):
        mesh_obj.data.vertices[vertex_index].co = pos
    result = np.array([[v[0], v[1], v[2]] for v in inverse_transformed_vertices])
    return result


def get_child_bones_recursive(bone_name: str, armature_obj: bpy.types.Object, clothing_avatar_data: dict = None) -> set:
    children = set()
    if bone_name not in armature_obj.data.bones:
        return children
    humanoid_bones = set()
    if clothing_avatar_data:
        for bone_map in clothing_avatar_data.get("humanoidBones", []):
            if "boneName" in bone_map:
                humanoid_bones.add(bone_map["boneName"])
    bone = armature_obj.data.bones[bone_name]
    for child in bone.children:
        if child.name in humanoid_bones:
            continue
        children.add(child.name)
        children.update(get_child_bones_recursive(child.name, armature_obj, clothing_avatar_data))
    return children


def create_blendshape_mask(target_obj, mask_bones, clothing_avatar_data, field_name="", store_debug_mask=True):
    mask_weights = np.zeros(len(target_obj.data.vertices))
    armature_obj = None
    for modifier in target_obj.modifiers:
        if modifier.type == "ARMATURE":
            armature_obj = modifier.object
            break
    if not armature_obj:
        return mask_weights
    humanoid_to_bone = {}
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
    auxiliary_bones = {}
    for aux_set in clothing_avatar_data.get("auxiliaryBones", []):
        humanoid_bone = aux_set["humanoidBoneName"]
        auxiliary_bones[humanoid_bone] = aux_set["auxiliaryBones"]
    target_bones = set()
    for humanoid_bone in mask_bones:
        bone_name = humanoid_to_bone.get(humanoid_bone)
        if bone_name:
            target_bones.add(bone_name)
            target_bones.update(get_child_bones_recursive(bone_name, armature_obj, clothing_avatar_data))
        if humanoid_bone in auxiliary_bones:
            for aux_bone in auxiliary_bones[humanoid_bone]:
                target_bones.add(aux_bone)
                target_bones.update(get_child_bones_recursive(aux_bone, armature_obj, clothing_avatar_data))
    target_groups = {g.index for g in target_obj.vertex_groups if g.name in target_bones}
    for vert in target_obj.data.vertices:
        for g in vert.groups:
            if g.group in target_groups:
                mask_weights[vert.index] += g.weight
    mask_weights = np.clip(mask_weights, 0.0, 1.0)
    if store_debug_mask:
        group_name = f"DEBUG_Mask_{field_name}" if field_name else "DEBUG_Mask"
        if group_name in target_obj.vertex_groups:
            target_obj.vertex_groups.remove(target_obj.vertex_groups[group_name])
        debug_group = target_obj.vertex_groups.new(name=group_name)
        for vert_idx, weight in enumerate(mask_weights):
            if weight > 0:
                debug_group.add([vert_idx], weight, "REPLACE")
    return mask_weights
