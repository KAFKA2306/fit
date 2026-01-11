import math

import bmesh
import bpy

from infrastructure.blender import armature


def merge_vertex_group_weights(mesh_obj: bpy.types.Object, source_group: str, target_group: str) -> None:
    if source_group not in mesh_obj.vertex_groups or target_group not in mesh_obj.vertex_groups:
        return
    source_idx = mesh_obj.vertex_groups[source_group].index
    target_idx = mesh_obj.vertex_groups[target_group].index
    for vert in mesh_obj.data.vertices:
        source_w = 0.0
        target_w = 0.0
        for g in vert.groups:
            if g.group == source_idx:
                source_w = g.weight
            elif g.group == target_idx:
                target_w = g.weight
        if source_w > 0:
            mesh_obj.vertex_groups[target_group].add([vert.index], source_w + target_w, "REPLACE")
    mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[source_group])


def process_bone_weight_consolidation(mesh_obj: bpy.types.Object, avatar_data: dict) -> None:
    upper_chest = armature.get_bone_name_from_humanoid(avatar_data, "UpperChest")
    chest = armature.get_bone_name_from_humanoid(avatar_data, "Chest")
    if upper_chest and chest and upper_chest in mesh_obj.vertex_groups:
        if chest not in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.new(name=chest)
        merge_vertex_group_weights(mesh_obj, upper_chest, chest)

    for breasts_humanoid in ["LeftBreasts", "RightBreasts"]:
        breasts_bone = armature.get_bone_name_from_humanoid(avatar_data, breasts_humanoid)
        if chest and breasts_bone and breasts_bone in mesh_obj.vertex_groups:
            if chest not in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups.new(name=chest)
            merge_vertex_group_weights(mesh_obj, breasts_bone, chest)

    for side in ["Left", "Right"]:
        foot = armature.get_bone_name_from_humanoid(avatar_data, f"{side}Foot")
        if not foot:
            continue
        toes_list = [f"{side}Toes"] + [
            f"{side}Foot{t}{suffix}"
            for t in ["Thumb", "Index", "Middle", "Ring", "Little"]
            for suffix in ["Proximal", "Intermediate", "Distal"]
        ]
        for toe_humanoid in toes_list:
            toe_bone = armature.get_bone_name_from_humanoid(avatar_data, toe_humanoid)
            if toe_bone and toe_bone in mesh_obj.vertex_groups:
                if foot not in mesh_obj.vertex_groups:
                    mesh_obj.vertex_groups.new(name=foot)
                merge_vertex_group_weights(mesh_obj, toe_bone, foot)


def get_bone_parent_map(bone_hierarchy: dict) -> dict:
    parent_map = {}

    def traverse_hierarchy(node, parent=None):
        current_bone = node["name"]
        parent_map[current_bone] = parent
        for child in node.get("children", []):
            traverse_hierarchy(child, current_bone)

    traverse_hierarchy(bone_hierarchy)
    return parent_map


def remove_empty_vertex_groups(mesh_obj: bpy.types.Object) -> None:
    if mesh_obj.type != "MESH" or not mesh_obj.vertex_groups:
        return
    used_vertex_groups = dict.fromkeys(range(len(mesh_obj.vertex_groups)), False)
    for vert in mesh_obj.data.vertices:
        for g in vert.groups:
            if g.weight > 0.0005:
                used_vertex_groups[g.group] = True
    total_vertex_count = len(mesh_obj.data.vertices)
    if total_vertex_count > 500:
        group_vertex_info = {i: [] for i in range(len(mesh_obj.vertex_groups))}
        for vert in mesh_obj.data.vertices:
            for g in vert.groups:
                if g.weight > 0:
                    group_vertex_info[g.group].append(g.weight)
        for group_idx, weights in group_vertex_info.items():
            if 0 < len(weights) <= 4 and all(w <= 0.01 for w in weights):
                used_vertex_groups[group_idx] = False
    groups_to_remove = []
    for i, used in used_vertex_groups.items():
        if not used:
            groups_to_remove.append(mesh_obj.vertex_groups[i].name)
    for group_name in groups_to_remove:
        if group_name in mesh_obj.vertex_groups:
            mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[group_name])

    bpy.ops.object.vertex_group_normalize_all(group_select_mode="BONE_DEFORM", lock_active=False)


def propagate_bone_weights(
    mesh_obj: bpy.types.Object, temp_group_name: str = "PropagatedWeightsTemp", max_iterations: int = 500
) -> str | None:
    armature_obj = None
    for modifier in mesh_obj.modifiers:
        if modifier.type == "ARMATURE":
            armature_obj = modifier.object
            break
    if not armature_obj:
        return None
    deform_groups = {bone.name for bone in armature_obj.data.bones}
    bm = bmesh.new()
    bm.from_mesh(mesh_obj.data)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    vertex_weights = {}
    vertices_without_weights = set()
    for vert in mesh_obj.data.vertices:
        weights = {}
        for group in mesh_obj.vertex_groups:
            if group.name in deform_groups:
                weight = 0.0
                for g in vert.groups:
                    if g.group == group.index:
                        weight = g.weight
                        break
                if weight > 0:
                    weights[group.name] = weight
        vertex_weights[vert.index] = weights
        if not weights:
            vertices_without_weights.add(vert.index)
    if not vertices_without_weights:
        return None
    if temp_group_name in mesh_obj.vertex_groups:
        mesh_obj.vertex_groups.remove(mesh_obj.vertex_groups[temp_group_name])
    temp_group = mesh_obj.vertex_groups.new(name=temp_group_name)
    iteration = 0
    while iteration < max_iterations and vertices_without_weights:
        propagated_this_iteration = 0
        remaining_vertices = set()
        for vert_idx in vertices_without_weights:
            vert = bm.verts[vert_idx]
            neighbors = set()
            for edge in vert.link_edges:
                other = edge.other_vert(vert)
                if vertex_weights[other.index]:
                    neighbors.add(other)
            if neighbors:
                closest_vert = min(neighbors, key=lambda v: (v.co - vert.co).length)
                vertex_weights[vert_idx] = vertex_weights[closest_vert.index].copy()
                temp_group.add([vert_idx], 1.0, "REPLACE")
                propagated_this_iteration += 1
            else:
                remaining_vertices.add(vert_idx)
        if propagated_this_iteration == 0:
            break
        vertices_without_weights = remaining_vertices
        iteration += 1
    if vertices_without_weights:
        total_weights = {}
        weight_count = 0
        for _vert_idx, weights in vertex_weights.items():
            if weights:
                weight_count += 1
                for group_name, weight in weights.items():
                    if group_name not in total_weights:
                        total_weights[group_name] = 0.0
                    total_weights[group_name] += weight
        if weight_count > 0:
            average_weights = {group_name: weight / weight_count for group_name, weight in total_weights.items()}
            for vert_idx in vertices_without_weights:
                vertex_weights[vert_idx] = average_weights.copy()
                temp_group.add([vert_idx], 1.0, "REPLACE")
    for vert_idx, weights in vertex_weights.items():
        for group_name, weight in weights.items():
            if group_name in mesh_obj.vertex_groups:
                mesh_obj.vertex_groups[group_name].add([vert_idx], weight, "REPLACE")
    bm.free()
    return temp_group_name


def process_missing_bone_weights(
    base_mesh: bpy.types.Object,
    clothing_armature: bpy.types.Object,
    base_avatar_data: dict,
    clothing_avatar_data: dict,
    preserve_optional_humanoid_bones: bool,
) -> None:
    clothing_bone_names = {bone.name for bone in clothing_armature.data.bones}
    base_humanoid_to_bone = {}
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            base_humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
    clothing_humanoid_to_bone = {}
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if "humanoidBoneName" in bone_map and "boneName" in bone_map:
            clothing_humanoid_to_bone[bone_map["humanoidBoneName"]] = bone_map["boneName"]
    parent_map = get_bone_parent_map(base_avatar_data["boneHierarchy"])
    for humanoid_name, bone_name in base_humanoid_to_bone.items():
        if clothing_humanoid_to_bone.get(humanoid_name) in clothing_bone_names:
            continue
        if preserve_optional_humanoid_bones:
            should_preserve = False
            if (
                (
                    humanoid_name == "UpperChest"
                    and "Chest" in clothing_humanoid_to_bone
                    and clothing_humanoid_to_bone["Chest"] in clothing_bone_names
                    and "UpperChest" not in clothing_humanoid_to_bone
                    and "UpperChest" in base_humanoid_to_bone
                )
                or (
                    humanoid_name == "LeftFoot"
                    and "LeftLowerLeg" in clothing_humanoid_to_bone
                    and clothing_humanoid_to_bone["LeftLowerLeg"] in clothing_bone_names
                    and "LeftFoot" not in clothing_humanoid_to_bone
                    and "LeftFoot" in base_humanoid_to_bone
                )
                or (
                    humanoid_name == "RightFoot"
                    and "RightLowerLeg" in clothing_humanoid_to_bone
                    and clothing_humanoid_to_bone["RightLowerLeg"] in clothing_bone_names
                    and "RightFoot" not in clothing_humanoid_to_bone
                    and "RightFoot" in base_humanoid_to_bone
                )
            ):
                should_preserve = True
            if should_preserve:
                continue
        parent_bone = parent_map.get(bone_name)
        if (
            parent_bone
            and parent_bone in clothing_bone_names
            and bone_name in base_mesh.vertex_groups
            and parent_bone in base_mesh.vertex_groups
        ):
            bone_group = base_mesh.vertex_groups[bone_name]
            parent_group = base_mesh.vertex_groups[parent_bone]
            for vert in base_mesh.data.vertices:
                bone_weight = 0.0
                for g in vert.groups:
                    if g.group == bone_group.index:
                        bone_weight = g.weight
                        break
                if bone_weight > 0:
                    parent_weight = 0.0
                    for g in vert.groups:
                        if g.group == parent_group.index:
                            parent_weight = g.weight
                            break
                    parent_group.add([vert.index], parent_weight + bone_weight, "REPLACE")
            base_mesh.vertex_groups.remove(bone_group)


def update_base_avatar_weights(
    base_mesh: bpy.types.Object,
    clothing_armature: bpy.types.Object,
    base_avatar_data: dict,
    clothing_avatar_data: dict,
    preserve_optional_humanoid_bones: bool = True,
) -> None:
    process_missing_bone_weights(
        base_mesh, clothing_armature, base_avatar_data, clothing_avatar_data, preserve_optional_humanoid_bones
    )


def fix_invalid_weights(mesh_obj: bpy.types.Object) -> None:
    if mesh_obj.type != "MESH":
        return
    mesh = mesh_obj.data
    vertex_groups = mesh_obj.vertex_groups
    if len(vertex_groups) == 0:
        return
    vertices_without_weights = set()
    for vert in mesh.vertices:
        has_valid_weight = False
        for g in vert.groups:
            weight = g.weight
            if not math.isfinite(weight) or math.isnan(weight):
                group_name = vertex_groups[g.group].name
                vertex_groups[group_name].add([vert.index], 0.0, "REPLACE")
            elif weight > 0.0:
                has_valid_weight = True
        if not has_valid_weight:
            vertices_without_weights.add(vert.index)
    if len(vertices_without_weights) == 0:
        return
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    def get_adjacent_vertices(vert_idx: int) -> set:
        adjacent = set()
        bm_vert = bm.verts[vert_idx]
        for edge in bm_vert.link_edges:
            other_vert = edge.other_vert(bm_vert)
            adjacent.add(other_vert.index)
        return adjacent

    def get_vertex_weights(vert_idx: int) -> dict:
        result = {}
        vert = mesh.vertices[vert_idx]
        for g in vert.groups:
            weight = g.weight
            if math.isfinite(weight) and not math.isnan(weight) and weight > 0.0:
                group_name = vertex_groups[g.group].name
                result[group_name] = weight
        return result

    def has_valid_weights(vert_idx: int) -> bool:
        vert = mesh.vertices[vert_idx]
        for g in vert.groups:
            weight = g.weight
            if math.isfinite(weight) and not math.isnan(weight) and weight > 0.0:
                return True
        return False

    for vert_idx in vertices_without_weights:
        first_level_neighbors = get_adjacent_vertices(vert_idx)
        valid_neighbors = []
        for neighbor_idx in first_level_neighbors:
            if has_valid_weights(neighbor_idx):
                valid_neighbors.append((neighbor_idx, 1))
        if not valid_neighbors:
            second_level_neighbors = set()
            for neighbor_idx in first_level_neighbors:
                second_level = get_adjacent_vertices(neighbor_idx)
                second_level_neighbors.update(second_level - {vert_idx} - first_level_neighbors)
            for neighbor_idx in second_level_neighbors:
                if has_valid_weights(neighbor_idx):
                    valid_neighbors.append((neighbor_idx, 2))
        if not valid_neighbors:
            continue
        min_distance = min(n[1] for n in valid_neighbors)
        closest_neighbors = [n[0] for n in valid_neighbors if n[1] == min_distance]
        combined_weights = {}
        for neighbor_idx in closest_neighbors:
            neighbor_weights = get_vertex_weights(neighbor_idx)
            for group_name, weight in neighbor_weights.items():
                if group_name not in combined_weights:
                    combined_weights[group_name] = []
                combined_weights[group_name].append(weight)
        for group_name, weight_list in combined_weights.items():
            avg_weight = sum(weight_list) / len(weight_list)
            if group_name in vertex_groups:
                vertex_groups[group_name].add([vert_idx], avg_weight, "REPLACE")
    bm.free()
