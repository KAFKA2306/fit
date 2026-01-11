import bmesh
import bpy
import numpy as np
from mathutils import Vector
from scipy.spatial import cKDTree

from infrastructure.blender.mesh import get_humanoid_and_auxiliary_bone_groups


def get_evaluated_mesh(obj: bpy.types.Object):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_obj = obj.evaluated_get(depsgraph)
    evaluated_mesh = evaluated_obj.data
    bm = bmesh.new()
    bm.from_mesh(evaluated_mesh)
    bm.transform(obj.matrix_world)
    return bm


def adjust_armature_hips_position(
    armature_obj: bpy.types.Object, target_position: Vector, clothing_avatar_data: dict
) -> None:
    if not armature_obj or armature_obj.type != "ARMATURE":
        return
    hips_bone_name = None
    for bone_map in clothing_avatar_data.get("humanoidBones", []):
        if bone_map["humanoidBoneName"] == "Hips":
            hips_bone_name = bone_map["boneName"]
            break
    if not hips_bone_name:
        return
    pose_bone = armature_obj.pose.bones.get(hips_bone_name)
    if not pose_bone:
        return
    current_position = armature_obj.matrix_world @ pose_bone.head
    offset = target_position - current_position
    if offset.length < 0.0001:
        return
    current_active = bpy.context.active_object
    current_mode = current_active.mode if current_active else "OBJECT"
    bpy.ops.object.mode_set(mode="OBJECT")
    children = []
    for child in bpy.data.objects:
        if child.parent == armature_obj:
            children.append(child)
    for child in children:
        bpy.ops.object.select_all(action="DESELECT")
        child.select_set(True)
        bpy.context.view_layer.objects.active = child
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    armature_obj.location += offset
    for child in children:
        bpy.ops.object.select_all(action="DESELECT")
        armature_obj.select_set(True)
        child.select_set(True)
        bpy.context.view_layer.objects.active = armature_obj
        bpy.ops.object.parent_set(type="OBJECT", keep_transform=True)
    bpy.ops.object.select_all(action="DESELECT")
    if current_active:
        current_active.select_set(True)
        bpy.context.view_layer.objects.active = current_active
        if current_mode != "OBJECT":
            bpy.ops.object.mode_set(mode=current_mode)
    bpy.context.view_layer.update()


def apply_pose_as_rest(armature: bpy.types.Object) -> None:
    original_active = bpy.context.active_object
    if not armature or armature.type != "ARMATURE":
        return
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.mode_set(mode="POSE")
    bpy.ops.pose.select_all(action="SELECT")
    bpy.ops.pose.armature_apply()
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.context.view_layer.objects.active = original_active


def create_hinge_bone_group(obj: bpy.types.Object, armature: bpy.types.Object, avatar_data: dict) -> None:
    bone_groups = get_humanoid_and_auxiliary_bone_groups(avatar_data)
    all_deform_groups = set(bone_groups)
    if armature:
        all_deform_groups.update(bone.name for bone in armature.data.bones)
    original_non_humanoid_groups = all_deform_groups - bone_groups
    cloth_bm = get_evaluated_mesh(obj)
    cloth_bm.verts.ensure_lookup_table()
    cloth_bm.faces.ensure_lookup_table()
    vertex_coords = np.array([v.co for v in cloth_bm.verts])
    kdtree = cKDTree(vertex_coords)
    hinge_bone_group = obj.vertex_groups.new(name="HingeBone")
    for bone_name in original_non_humanoid_groups:
        bone = armature.pose.bones.get(bone_name)
        if bone.parent and bone.parent.name in bone_groups:
            group_index = obj.vertex_groups.find(bone_name)
            if group_index != -1:
                bone_head = armature.matrix_world @ bone.head
                neighbor_indices = kdtree.query_ball_point(bone_head, 0.01)
                for index in neighbor_indices:
                    for g in obj.data.vertices[index].groups:
                        if g.group == group_index:
                            weight = g.weight
                            hinge_bone_group.add([index], weight, "REPLACE")
                            break


def get_bone_name_from_humanoid(avatar_data: dict, humanoid_bone_name: str) -> str | None:
    for mapping in avatar_data.get("humanoidBones", []):
        if mapping.get("humanoidBoneName") == humanoid_bone_name:
            return mapping.get("boneName")
    return None


def calculate_inverse_pose_matrix(
    obj: bpy.types.Object, armature_obj: bpy.types.Object, vert_idx: int
) -> bpy.types.mathutils.Matrix:
    from mathutils import Matrix

    vert = obj.data.vertices[vert_idx]
    bone_weight_sum = 0
    matrix_sum = Matrix.Zero(4)
    matrix_sum.zero()  # Ensure all zeros
    deform_bones = {b.name for b in armature_obj.data.bones}
    for g in vert.groups:
        group_name = obj.vertex_groups[g.group].name
        if g.weight > 0 and group_name in deform_bones:
            pose_bone = armature_obj.pose.bones[group_name]
            bind_bone = armature_obj.data.bones[group_name]
            if pose_bone and bind_bone:
                # armature_world @ pose_matrix @ bind_matrix_inv @ armature_world_inv
                mat = (
                    armature_obj.matrix_world
                    @ pose_bone.matrix
                    @ bind_bone.matrix_local.inverted()
                    @ armature_obj.matrix_world.inverted()
                )
                matrix_sum += g.weight * mat
                bone_weight_sum += g.weight
    if bone_weight_sum > 0:
        final_mat = matrix_sum * (1.0 / bone_weight_sum)
        try:
            return final_mat.inverted()
        except Exception:
            return Matrix.Identity(4)
    return Matrix.Identity(4)
