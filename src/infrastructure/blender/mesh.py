import bmesh
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree


def get_humanoid_and_auxiliary_bone_groups(base_avatar_data: dict) -> set:
    bone_groups = set()
    for bone_map in base_avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            bone_groups.add(bone_map["boneName"])
    for aux_set in base_avatar_data.get("auxiliaryBones", []):
        for aux_bone in aux_set.get("auxiliaryBones", []):
            bone_groups.add(aux_bone)
    return bone_groups


def check_edge_direction_similarity(directions1, directions2, angle_threshold=3.0) -> bool:
    if not directions1 or not directions2:
        return False
    angle_threshold_rad = np.radians(angle_threshold)
    for dir1 in directions1:
        for dir2 in directions2:
            dot_product = dir1.dot(dir2)
            dot_product = max(min(dot_product, 1.0), -1.0)
            angle = np.arccos(dot_product)
            if angle <= angle_threshold_rad or angle >= (np.pi - angle_threshold_rad):
                return True
    return False


def calculate_weight_pattern_similarity(weights1: dict, weights2: dict) -> float:
    all_groups = set(weights1.keys()) | set(weights2.keys())
    if not all_groups:
        return 0.0
    total_diff = 0.0
    for group in all_groups:
        w1 = weights1.get(group, 0.0)
        w2 = weights2.get(group, 0.0)
        total_diff += abs(w1 - w2)
    normalized_diff = total_diff / len(all_groups)
    similarity = 1.0 - min(normalized_diff, 1.0)
    return similarity


def triangulate_mesh(obj: bpy.types.Object) -> None:
    if obj is None or obj.type != "MESH":
        return
    original_active = bpy.context.view_layer.objects.active
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="FIXED", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    if original_active:
        bpy.context.view_layer.objects.active = original_active
    obj.select_set(False)


def cleanup_mesh(mesh_obj: bpy.types.Object) -> None:
    if mesh_obj is None or mesh_obj.type != "MESH":
        return
    mesh = mesh_obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    loose_verts = [v for v in bm.verts if len(v.link_edges) == 0]
    if len(loose_verts) >= 1000:
        for v in loose_verts:
            bm.verts.remove(v)
    non_finite_verts = [
        v for v in bm.verts if not (np.isfinite(v.co.x) and np.isfinite(v.co.y) and np.isfinite(v.co.z))
    ]
    if non_finite_verts:
        for v in non_finite_verts:
            bm.verts.remove(v)
    bm.to_mesh(mesh)
    bm.free()


def create_overlapping_vertices_attributes(
    clothing_meshes: list,
    base_avatar_data: dict,
    distance_threshold: float = 0.0001,
    edge_angle_threshold: float = 3,
    weight_similarity_threshold: float = 0.1,
    overlap_attr_name: str = "Overlapped",
    world_pos_attr_name: str = "OriginalWorldPosition",
) -> None:
    target_groups = get_humanoid_and_auxiliary_bone_groups(base_avatar_data)
    for mesh_obj in clothing_meshes:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        eval_obj = mesh_obj.evaluated_get(depsgraph)
        mesh = eval_obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        vert_indices = {v.index: i for i, v in enumerate(bm.verts)}
        all_vertices = []
        for vert_idx, vert in enumerate(bm.verts):
            world_pos = mesh_obj.matrix_world @ vert.co
            edge_directions = []
            bm_vert = bm.verts[vert_idx]
            for edge in bm_vert.link_edges:
                other_vert = edge.other_vert(bm_vert)
                direction = (other_vert.co - bm_vert.co).normalized()
                edge_directions.append(direction)
            weights = {}
            orig_vert = mesh_obj.data.vertices[vert_indices[vert_idx]]
            for group_name in target_groups:
                if group_name in mesh_obj.vertex_groups:
                    group = mesh_obj.vertex_groups[group_name]
                    for g in orig_vert.groups:
                        if g.group == group.index:
                            weights[group_name] = g.weight
                            break
            all_vertices.append(
                {"vert_idx": vert_idx, "world_pos": world_pos, "edge_directions": edge_directions, "weights": weights}
            )
        positions = [v["world_pos"] for v in all_vertices]
        kdtree = KDTree(len(positions))
        for i, pos in enumerate(positions):
            kdtree.insert(pos, i)
        kdtree.balance()
        if overlap_attr_name not in mesh_obj.data.attributes:
            mesh_obj.data.attributes.new(name=overlap_attr_name, type="FLOAT", domain="POINT")
        overlap_attr = mesh_obj.data.attributes[overlap_attr_name]
        if world_pos_attr_name not in mesh_obj.data.attributes:
            mesh_obj.data.attributes.new(name=world_pos_attr_name, type="FLOAT_VECTOR", domain="POINT")
        pos_attr = mesh_obj.data.attributes[world_pos_attr_name]
        for i, vertex in enumerate(mesh_obj.data.vertices):
            overlap_attr.data[i].value = 0.0
            world_position = mesh_obj.matrix_world @ vertex.co
            pos_attr.data[i].vector = world_position
        processed = set()
        for i, vert_data in enumerate(all_vertices):
            mesh_vertex_idx = vert_indices[all_vertices[i]["vert_idx"]]
            world_pos = all_vertices[i]["world_pos"]
            pos_attr.data[mesh_vertex_idx].vector = world_pos
            if i in processed:
                continue
            overlapping_indices = []
            for _co, idx, _dist in kdtree.find_range(vert_data["world_pos"], distance_threshold):
                if check_edge_direction_similarity(
                    vert_data["edge_directions"], all_vertices[idx]["edge_directions"], edge_angle_threshold
                ):
                    similarity = calculate_weight_pattern_similarity(vert_data["weights"], all_vertices[idx]["weights"])
                    if similarity >= (1.0 - weight_similarity_threshold):
                        overlapping_indices.append(idx)
            if not overlapping_indices:
                continue
            overlapping_indices.append(i)
            processed.add(i)
            for vert_idx in overlapping_indices:
                mesh_vertex_idx = vert_indices[all_vertices[vert_idx]["vert_idx"]]
                overlap_attr.data[mesh_vertex_idx].value = 1.0
                processed.add(vert_idx)
        bm.free()
        mesh_obj.data.update()


def create_deformation_mask(obj: bpy.types.Object, avatar_data: dict) -> None:
    if obj.type != "MESH":
        return
    group_names = set()
    for bone_map in avatar_data.get("humanoidBones", []):
        if "boneName" in bone_map:
            group_names.add(bone_map["boneName"])
    if "DeformationMask" in obj.vertex_groups:
        obj.vertex_groups.remove(obj.vertex_groups["DeformationMask"])
    deformation_mask = obj.vertex_groups.new(name="DeformationMask")
    for vert in obj.data.vertices:
        should_add = False
        weight_sum = 0.0
        for group_name in group_names:
            if group_name in obj.vertex_groups:
                group_idx = obj.vertex_groups[group_name].index
                weight = 0
                for g in vert.groups:
                    if g.group == group_idx:
                        weight = g.weight
                        break
                if weight > 0:
                    should_add = True
                    weight_sum += weight
        if should_add:
            deformation_mask.add([vert.index], weight_sum, "REPLACE")


def subdivide_long_edges(obj, min_edge_length=0.005, max_edge_length_ratio=2.0, cuts=1):
    mesh = obj.data
    if not obj or obj.type != "MESH":
        return
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.edges.ensure_lookup_table()
    edge_lengths = [e.calc_length() for e in bm.edges]
    if not edge_lengths:
        bm.free()
        return
    edge_lengths.sort()
    n = len(edge_lengths)
    median = edge_lengths[n // 2]
    threshold = median * max_edge_length_ratio
    edges_to_sub = [e for e in bm.edges if e.calc_length() >= threshold]
    if edges_to_sub:
        bmesh.ops.subdivide_edges(bm, edges=edges_to_sub, cuts=cuts, use_grid_fill=True)
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()


def subdivide_faces(obj, face_indices, cuts=1, max_distance=0.005):
    mesh = obj.data
    if not obj or obj.type != "MESH":
        return
    if len(obj.data.vertices) == 0:
        return
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.transform(obj.matrix_world)
    bvh_tree = BVHTree.FromBMesh(bm)
    initial_faces = {f for f in bm.faces if f.index in face_indices}
    faces_within_distance = set(initial_faces)
    for f in initial_faces:
        face_center = f.calc_center_median()
        max_edge_length = max([e.calc_length() for e in f.edges])
        search_radius = min(max_edge_length, max_distance)
        for _location, _normal, index, _distance in bvh_tree.find_nearest_range(face_center, search_radius):
            if index is not None and index < len(bm.faces):
                faces_within_distance.add(bm.faces[index])
    bm.transform(obj.matrix_world.inverted())
    all_edges_candidates = {edge for f in faces_within_distance for edge in f.edges}
    edges_to_subdivide = [e for e in all_edges_candidates if e.calc_length() >= 0.004]
    if edges_to_subdivide:
        bmesh.ops.subdivide_edges(
            bm, edges=edges_to_subdivide, cuts=cuts, use_grid_fill=True, use_single_edge=False, use_only_quads=True
        )
        bm.to_mesh(mesh)
        mesh.update()
    bm.free()


def subdivide_breast_faces(target_obj, clothing_avatar_data):
    if not clothing_avatar_data:
        return
    breast_bone_names = []
    for mapping in clothing_avatar_data.get("humanoidBones", []):
        if mapping["humanoidBoneName"] in ["LeftBreast", "RightBreast"]:
            breast_bone_names.append(mapping["boneName"])
    for group in clothing_avatar_data.get("auxiliaryBones", []):
        if group["humanoidBoneName"] in ["LeftBreast", "RightBreast"]:
            breast_bone_names.extend(group["auxiliaryBones"])
    breast_vertices = set()
    for name in breast_bone_names:
        if name in target_obj.vertex_groups:
            idx = target_obj.vertex_groups[name].index
            for v in target_obj.data.vertices:
                for g in v.groups:
                    if g.group == idx and g.weight > 0.001:
                        breast_vertices.add(v.index)
    if breast_vertices:
        relevant_faces = [f.index for f in target_obj.data.polygons if any(vi in breast_vertices for vi in f.vertices)]
        if relevant_faces:
            subdivide_faces(target_obj, relevant_faces, cuts=1)


def find_connected_components(obj: bpy.types.Object) -> list[set[int]]:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    components = []
    visited = set()
    for v in bm.verts:
        if v.index not in visited:
            component = set()
            stack = [v]
            visited.add(v.index)
            while stack:
                curr = stack.pop()
                component.add(curr.index)
                for edge in curr.link_edges:
                    other = edge.other_vert(curr)
                    if other.index not in visited:
                        visited.add(other.index)
                        stack.append(other)
            components.append(component)
    bm.free()
    return components


def check_uniform_weights(
    obj: bpy.types.Object, vert_indices: set[int], armature_obj: bpy.types.Object
) -> tuple[bool, dict]:
    if not armature_obj:
        return False, {}
    deform_bones = {b.name for b in armature_obj.data.bones}
    first_idx = next(iter(vert_indices))
    first_weights = {}
    if first_idx >= len(obj.data.vertices):
        return False, {}
    for g in obj.data.vertices[first_idx].groups:
        name = obj.vertex_groups[g.group].name
        if name in deform_bones:
            first_weights[name] = g.weight
    for idx in vert_indices:
        curr_weights = {}
        for g in obj.data.vertices[idx].groups:
            name = obj.vertex_groups[g.group].name
            if name in deform_bones:
                curr_weights[name] = g.weight
        if set(first_weights.keys()) != set(curr_weights.keys()):
            return False, {}
        for name, w in first_weights.items():
            if abs(w - curr_weights.get(name, 0.0)) > 0.001:
                return False, {}
    return True, first_weights


def cluster_components_by_adaptive_distance(component_coords: dict, component_sizes: dict) -> list[list[int]]:
    if not component_coords:
        return []
    centers = {}
    for comp_idx, coords in component_coords.items():
        if coords:
            center = Vector((0, 0, 0))
            for co in coords:
                center += co
            center /= len(coords)
            centers[comp_idx] = center
    clusters = [[comp_idx] for comp_idx in centers]
    avg_size = sum(component_sizes.values()) / len(component_sizes) if component_sizes else 0.1
    min_t, max_t = 0.1, 1.0
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            if i >= len(clusters):
                break
            for j in range(i + 1, len(clusters)):
                if j >= len(clusters):
                    break
                min_dist = float("inf")
                si, sj = 0.0, 0.0
                for ci in clusters[i]:
                    for cj in clusters[j]:
                        if ci in centers and cj in centers:
                            dist = (centers[ci] - centers[cj]).length
                            if dist < min_dist:
                                min_dist = dist
                                si, sj = component_sizes.get(ci, avg_size), component_sizes.get(cj, avg_size)
                threshold = max(si, sj) * 0.5
                threshold = max(min_t, min(max_t, threshold))
                if min_dist <= threshold:
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break
    return clusters


def separate_and_combine_components(
    obj: bpy.types.Object, armature_obj: bpy.types.Object, do_not_separate: list = None
) -> tuple[list, list]:
    if do_not_separate is None:
        do_not_separate = []
    components = find_connected_components(obj)
    if len(components) <= 1:
        return [], [obj]

    uniform_data = []
    non_separated_indices = set()
    for comp in components:
        is_uniform, weights = check_uniform_weights(obj, comp, armature_obj)
        # Check do_not_separate (approximate by first index name check if needed)
        # Here we just check if any vertex in comp should be forced to stay
        should_separate = is_uniform
        if should_separate:
            # We don't have individual names for components yet, but we can check if the whole group matches a pattern
            # We assume the caller passes names which we match later
            pass

        if should_separate:
            coords = [obj.matrix_world @ obj.data.vertices[i].co for i in comp]
            size = sum(max(c[k] for c in coords) - min(c[k] for c in coords) for k in range(3))
            uniform_data.append({"comp": comp, "weights": weights, "coords": coords, "size": size})
        else:
            non_separated_indices.update(comp)

    groups = {}
    for d in uniform_data:
        w_hash = "_".join([f"{k}:{v:.3f}" for k, v in sorted(d["weights"].items())])
        if w_hash not in groups:
            groups[w_hash] = []
        groups[w_hash].append(d)

    separated_objs = []
    processed_indices = set()
    for _w_hash, items in groups.items():
        comp_coords = {i: item["coords"] for i, item in enumerate(items)}
        comp_sizes = {i: item["size"] for i, item in enumerate(items)}
        clusters = cluster_components_by_adaptive_distance(comp_coords, comp_sizes)

        for cluster in clusters:
            keep = set()
            for idx in cluster:
                keep.update(items[idx]["comp"])

            # Check if this cluster's name (placeholder) is in do_not_separate
            # Since cluster indices change, we use a simple heuristic or deferred check in caller

            bpy.ops.object.select_all(action="DESELECT")
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.duplicate(linked=False)
            new_obj = bpy.context.active_object
            new_obj.name = f"{obj.name}_comp"

            # Use bmesh for fast deletion
            bm = bmesh.new()
            bm.from_mesh(new_obj.data)
            bm.verts.ensure_lookup_table()
            to_del = [v for v in bm.verts if v.index not in keep]
            bmesh.ops.delete(bm, geom=to_del, context="VERTS")
            bm.to_mesh(new_obj.data)
            bm.free()

            if new_obj.name in do_not_separate:  # Simple match
                non_separated_indices.update(keep)
                bpy.data.objects.remove(new_obj, do_unlink=True)
            else:
                separated_objs.append(new_obj)
                processed_indices.update(keep)

    all_indices = {v.index for v in obj.data.vertices}
    remainder_indices = all_indices - processed_indices

    non_separated_objs = []
    if remainder_indices:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.duplicate(linked=False)
        rem_obj = bpy.context.active_object
        rem_obj.name = f"{obj.name}_remainder"
        bm = bmesh.new()
        bm.from_mesh(rem_obj.data)
        bm.verts.ensure_lookup_table()
        to_del = [v for v in bm.verts if v.index not in remainder_indices]
        bmesh.ops.delete(bm, geom=to_del, context="VERTS")
        bm.to_mesh(rem_obj.data)
        bm.free()
        non_separated_objs.append(rem_obj)

    return separated_objs, non_separated_objs


def calculate_distance_based_weights(
    source_obj: bpy.types.Object,
    target_obj: bpy.types.Object,
    vertex_group_name="DistanceWeight",
    min_distance=0.0,
    max_distance=0.03,
) -> None:
    if vertex_group_name not in source_obj.vertex_groups:
        vg = source_obj.vertex_groups.new(name=vertex_group_name)
    else:
        vg = source_obj.vertex_groups[vertex_group_name]

    depsgraph = bpy.context.evaluated_depsgraph_get()
    target_eval = target_obj.evaluated_get(depsgraph)
    target_mesh = target_eval.data
    target_verts = [target_obj.matrix_world @ v.co for v in target_mesh.vertices]
    target_polys = [p.vertices for p in target_mesh.polygons]
    bvh = BVHTree.FromPolygons(target_verts, target_polys)

    source_eval = source_obj.evaluated_get(depsgraph)
    source_mesh = source_eval.data
    for i, vert in enumerate(source_mesh.vertices):
        world_co = source_obj.matrix_world @ vert.co
        _loc, _norm, _idx, dist = bvh.find_nearest(world_co)
        if dist is None:
            weight = 0.0
        elif dist <= min_distance:
            weight = 1.0
        elif dist >= max_distance:
            weight = 0.0
        else:
            weight = 1.0 - ((dist - min_distance) / (max_distance - min_distance))
        vg.add([i], weight, "REPLACE")
