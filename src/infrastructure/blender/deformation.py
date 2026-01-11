import os

import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from infrastructure.blender import armature, geometry
from infrastructure.blender import mesh as mesh_ops
from infrastructure.blender import ops as blender_ops


class TransitionCache:
    def __init__(self):
        self.cache = {}

    def get_cache_key(self, blendshape_values):
        sorted_items = sorted(blendshape_values.items())
        return hash(tuple(sorted_items))

    def store_result(self, blendshape_values, vertices, all_blendshape_values):
        cache_key = self.get_cache_key(blendshape_values)
        if cache_key in self.cache:
            return
        self.cache[cache_key] = {"vertices": vertices.copy(), "blendshape_values": all_blendshape_values.copy()}

    def find_interpolation_candidates(self, target_blendshape_values, changing_blendshape, blendshape_groups=None):
        candidates = []
        group_blendshapes = set()
        if blendshape_groups:
            for group in blendshape_groups:
                if changing_blendshape in group.get("blendShapeFields", []):
                    group_blendshapes = set(group.get("blendShapeFields", []))
                    break
        for cached_data in self.cache.values():
            cached_values = cached_data["blendshape_values"]
            match = True
            if group_blendshapes:
                for name in group_blendshapes:
                    if (
                        name != changing_blendshape
                        and abs(cached_values.get(name, 0.0) - target_blendshape_values.get(name, 0.0)) > 1e-6
                    ):
                        match = False
                        break
            if match:
                candidates.append(
                    {
                        "cached_val": cached_values.get(changing_blendshape, 0.0),
                        "target_val": target_blendshape_values.get(changing_blendshape, 0.0),
                        "vertices": cached_data["vertices"],
                        "diff": abs(
                            cached_values.get(changing_blendshape, 0.0)
                            - target_blendshape_values.get(changing_blendshape, 0.0)
                        ),
                    }
                )
        return candidates

    def interpolate_result(self, target_blendshape_values, changing_blendshape, blendshape_groups=None):
        candidates = self.find_interpolation_candidates(
            target_blendshape_values, changing_blendshape, blendshape_groups
        )
        if len(candidates) < 2:
            return None
        target_v = target_blendshape_values.get(changing_blendshape, 0.0)
        valid_pairs = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                v1, v2 = candidates[i]["cached_val"], candidates[j]["cached_val"]
                if (v1 <= target_v <= v2) or (v2 <= target_v <= v1):
                    if abs(v2 - v1) < 1e-6:
                        continue
                    valid_pairs.append(
                        {"interval": abs(v2 - v1), "c1": candidates[i], "c2": candidates[j], "v1": v1, "v2": v2}
                    )
        if not valid_pairs:
            return None
        best = min(valid_pairs, key=lambda x: x["interval"])
        t = (target_v - best["v1"]) / (best["v2"] - best["v1"])
        return best["c1"]["vertices"] + t * (best["c2"]["vertices"] - best["c1"]["vertices"])


def process_field_deformation(
    target_obj,
    field_data_path,
    blend_shape_labels=None,
    clothing_avatar_data=None,
    shape_key_name="SymmetricDeformed",
    ignore_blendshape=None,
    target_shape_key=None,
    cache=None,
    label=None,
):
    applied_base_shape = None
    if label and target_obj.data.shape_keys:
        base_shape_name = f"{label}_BaseShape"
        if base_shape_name in target_obj.data.shape_keys.key_blocks:
            for sk in target_obj.data.shape_keys.key_blocks:
                sk.value = 0.0
            applied_base_shape = target_obj.data.shape_keys.key_blocks[base_shape_name]
            applied_base_shape.value = 1.0

    if target_shape_key:
        for sk in target_obj.data.shape_keys.key_blocks:
            sk.value = 0.0
        target_shape_key.value = 1.0

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_mesh = target_obj.evaluated_get(depsgraph).data
    blend_positions = np.array([v.co for v in eval_mesh.vertices])

    field_info = geometry.get_deformation_field_multi_step(field_data_path, cache)
    final_positions = geometry.batch_process_vertices_multi_step(
        blend_positions,
        field_info["all_field_points"],
        field_info["all_delta_positions"],
        field_info["field_weights"],
        field_info["world_matrix"],
        field_info["world_matrix_inv"],
        target_obj.matrix_world,
        target_obj.matrix_world.inverted(),
        k=field_info["kdtree_query_k"],
    )

    # find armature
    clothing_armature = next((mod.object for mod in target_obj.modifiers if mod.type == "ARMATURE"), None)

    # Revert BaseShape application
    if applied_base_shape:
        applied_base_shape.value = 0.0

    active_sk = target_obj.data.shape_keys.key_blocks.get(shape_key_name) or target_obj.shape_key_add(
        name=shape_key_name
    )
    active_sk.value = 1.0

    for i in range(len(final_positions)):
        p_vec = Vector(final_positions[i])
        # inv_pose is (Pose World -> Bind World)
        inv_pose = armature.calculate_inverse_pose_matrix(target_obj, clothing_armature, i)
        # Math: Bind Local = Mesh Matrix Inv @ Bind World
        active_sk.data[i].co = target_obj.matrix_world.inverted() @ inv_pose @ p_vec

    return active_sk


def find_intersecting_faces_bvh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    tree = BVHTree.FromBMesh(bm)
    intersections = tree.overlap(tree)
    bm.free()
    return intersections


def retarget_mesh(obj, ctx):
    mesh_ops.cleanup_mesh(obj)

    # Pre-subdivision
    if not ctx.config.no_subdivision and ctx.pair_index == 0:
        mesh_ops.subdivide_long_edges(obj)
        mesh_ops.subdivide_breast_faces(obj, ctx.clothing_avatar_data)

    mesh_ops.calculate_distance_based_weights(obj, ctx.base_mesh)

    clothing_armature = ctx.clothing_armature
    field_data_path = ctx.config.field_data
    blend_shapes = ctx.config.blend_shapes.split(";") if ctx.config.blend_shapes else None

    config_data = {}
    if field_data_path and os.path.exists(field_data_path):
        import json

        with open(field_data_path) as f:
            config_data = json.load(f)

    # Identify non-separation list
    temp_sep, _ = mesh_ops.separate_and_combine_components(obj, clothing_armature)
    do_not_separate = []
    for s_obj in temp_sep:
        apply_field_delta_with_rigid_transform_single(s_obj, field_data_path, ctx)
        obb = geometry.calculate_obb_from_object(s_obj)
        if geometry.check_obb_intersection(ctx.base_mesh, obb):
            do_not_separate.append(s_obj.name)
        bpy.data.objects.remove(s_obj, do_unlink=True)

    # Actual processing
    separated_objs, non_separated_objs = mesh_ops.separate_and_combine_components(
        obj, clothing_armature, do_not_separate=do_not_separate
    )
    processed = []

    for s_obj in separated_objs:
        apply_field_delta_with_rigid_transform_single(s_obj, field_data_path, ctx)
        # Also apply blendshape fields rigidly if needed
        if config_data and "blendShapeFields" in config_data:
            for bf in config_data["blendShapeFields"]:
                label = bf["label"]
                if blend_shapes and label not in blend_shapes:
                    continue
                bf_path = os.path.join(os.path.dirname(field_data_path), bf["path"])
                if os.path.exists(bf_path):
                    apply_field_delta_with_rigid_transform_single(s_obj, bf_path, ctx, label=label)
        processed.append(s_obj)

    for n_obj in non_separated_objs:
        apply_symmetric_field_delta(
            n_obj,
            field_data_path,
            blend_shapes,
            ctx.clothing_avatar_data,
            ctx.base_avatar_data,
            subdivision=False,
            cache=ctx.cache,
            config_data=config_data,
        )
        processed.append(n_obj)

    # Join back
    if processed:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.delete(type="VERT")
        bpy.ops.object.mode_set(mode="OBJECT")

        for part in processed:
            bpy.ops.object.select_all(action="DESELECT")
            part.select_set(True)
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.join()

    # Post-triangulation
    if not ctx.config.no_triangle and ctx.pair_index == ctx.total_pairs - 1:
        mesh_ops.triangulate_mesh(obj)


def apply_field_delta_with_rigid_transform_single(obj, field_data_path, ctx, label=None, shape_key_name=None):
    if shape_key_name is None:
        shape_key_name = label if label else "RigidTransformed"

    # apply label_BaseShape before retargeting
    applied_base_shape = None
    if label and obj.data.shape_keys:
        base_shape_name = f"{label}_BaseShape"
        if base_shape_name in obj.data.shape_keys.key_blocks:
            applied_base_shape = obj.data.shape_keys.key_blocks[base_shape_name]
            applied_base_shape.value = 1.0

    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_mesh = obj.evaluated_get(depsgraph).data
    positions = np.array([v.co for v in eval_mesh.vertices])

    field_info = geometry.get_deformation_field_multi_step(field_data_path, ctx.cache)
    deformed_positions = geometry.batch_process_vertices_multi_step(
        positions,
        field_info["all_field_points"],
        field_info["all_delta_positions"],
        field_info["field_weights"],
        field_info["world_matrix"],
        field_info["world_matrix_inv"],
        obj.matrix_world,
        obj.matrix_world.inverted(),
        k=field_info["kdtree_query_k"],
    )

    source_p = np.array([obj.matrix_world @ Vector(v) for v in positions])
    target_p = np.array(deformed_positions)
    s, R, t = geometry.calculate_optimal_similarity_transform(source_p, target_p)

    # Revert BaseShape application
    if applied_base_shape:
        applied_base_shape.value = 0.0

    sk = obj.data.shape_keys.key_blocks.get(shape_key_name) or obj.shape_key_add(name=shape_key_name)
    sk.value = 1.0

    for i in range(len(positions)):
        # Calculate similarity transformed point in world coordinates
        p_vec = Vector(source_p[i])
        world_p_vec = s * (R @ p_vec) + t

        # inv_pose is (Pose World -> Bind World)
        inv_pose = armature.calculate_inverse_pose_matrix(obj, ctx.clothing_armature, i)

        # sk.data[i].co is Bind Local
        # Math: Bind Local = Mesh Matrix Inv @ Bind World
        sk.data[i].co = obj.matrix_world.inverted() @ inv_pose @ world_p_vec

    return sk


def apply_symmetric_field_delta(
    target_obj,
    field_data_path,
    blend_shape_labels=None,
    clothing_avatar_data=None,
    base_avatar_data=None,
    subdivision=True,
    shape_key_name="SymmetricDeformed",
    config_data=None,
    cache=None,
):
    iteration = 0
    MAX_ITERATIONS = 0
    shape_key = None

    # Process Basis
    basis_path = field_data_path
    while iteration <= MAX_ITERATIONS:
        original_sk_state = blender_ops.save_shape_key_state(target_obj)
        shape_key = process_field_deformation(
            target_obj, basis_path, blend_shape_labels, clothing_avatar_data, shape_key_name, cache=cache
        )
        blender_ops.restore_shape_key_state(target_obj, original_sk_state)

        if not subdivision:
            break
        intersections = find_intersecting_faces_bvh(target_obj)
        if not intersections or iteration == MAX_ITERATIONS:
            break
        iteration += 1

    # Process Config BlendShapes
    if config_data and "blendShapeFields" in config_data:
        for bf in config_data["blendShapeFields"]:
            label = bf["label"]
            if blend_shape_labels and label not in blend_shape_labels:
                continue
            bf_path = os.path.join(os.path.dirname(field_data_path), bf["path"])
            if os.path.exists(bf_path):
                process_field_deformation(
                    target_obj, bf_path, None, clothing_avatar_data, shape_key_name=label, cache=cache, label=label
                )

    return shape_key
