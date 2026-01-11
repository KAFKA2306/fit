import math

import bpy
import numpy as np
from mathutils import Matrix, Vector
from scipy.spatial import cKDTree


def calculate_obb_from_points(points):
    if len(points) < 3:
        return None
    points_np = np.array([[p.x, p.y, p.z] for p in points])
    center = np.mean(points_np, axis=0)
    centered_points = points_np - center
    cov_matrix = np.cov(centered_points, rowvar=False)
    if np.linalg.matrix_rank(cov_matrix) < 3:
        return None
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    if np.any(np.abs(eigenvalues) < 1e-10):
        return None
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    axes = eigenvectors
    projections = np.abs(np.dot(centered_points, axes))
    radii = np.max(projections, axis=0)
    return {"center": center, "axes": axes, "radii": radii}


def calculate_obb_from_object(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_mesh = obj.evaluated_get(depsgraph).data
    points = [obj.matrix_world @ v.co for v in eval_mesh.vertices]
    return calculate_obb_from_points(points)


def check_obb_intersection(mesh_obj, obb):
    if obb is None:
        return False
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_mesh = mesh_obj.evaluated_get(depsgraph).data
    for v in eval_mesh.vertices:
        world_p = mesh_obj.matrix_world @ v.co
        rel_p = world_p - Vector(obb["center"])
        projs = [abs(rel_p.dot(Vector(obb["axes"][:, i]))) for i in range(3)]
        if all(p <= r for p, r in zip(projs, obb["radii"], strict=False)):
            return True
    return False


def cross2d(u: Vector, v: Vector) -> float:
    return u.y * v.x - u.x * v.y


def point_in_triangle2d(p: Vector, a: Vector, b: Vector, c: Vector) -> bool:
    pab = cross2d(p - a, b - a)
    pbc = cross2d(p - b, c - b)
    if pab * pbc < 0:
        return False
    pca = cross2d(p - c, a - c)
    if pab * pca < 0:
        return False
    return True


def signed_2d_tri_area(a: Vector, b: Vector, c: Vector) -> float:
    return (a.x - c.x) * (b.y - c.y) - (a.y - c.y) * (b.x - c.x)


def test_2d_segment_segment(a: Vector, b: Vector, c: Vector, d: Vector) -> bool:
    a1 = signed_2d_tri_area(a, b, d)
    a2 = signed_2d_tri_area(a, b, c)
    if a1 * a2 < 0.0:
        a3 = signed_2d_tri_area(c, d, a)
        a4 = a3 + a2 - a1
        if a3 * a4 < 0.0:
            return True
    return False


def project_triangle_2d(triangle: list[Vector], normal: Vector) -> list[Vector]:
    if abs(normal.x) >= abs(normal.y) and abs(normal.x) >= abs(normal.z):
        return [Vector((v.y, v.z)) for v in triangle]
    elif abs(normal.y) >= abs(normal.z):
        return [Vector((v.x, v.z)) for v in triangle]
    else:
        return [Vector((v.x, v.y)) for v in triangle]


def triangle_area(triangle: list[Vector]) -> float:
    a = (triangle[1] - triangle[0]).length
    b = (triangle[2] - triangle[1]).length
    c = (triangle[0] - triangle[2]).length
    s = (a + b + c) / 2
    area_val = max(s * (s - a) * (s - b) * (s - c), 0)
    area = math.sqrt(area_val)
    return area


def is_degenerate_triangle(triangle: list[Vector], epsilon: float = 1e-6) -> bool:
    area = triangle_area(triangle)
    return area < epsilon


def calc_triangle_normal(triangle: list[Vector]) -> Vector:
    v1 = triangle[1] - triangle[0]
    v2 = triangle[2] - triangle[0]
    normal = v1.cross(v2)
    length = normal.length
    if length > 1e-8:
        return normal / length
    return Vector((0, 0, 0))


def intersect_triangle_triangle(t1: list[Vector], t2: list[Vector]) -> bool:
    EPSILON2 = 1e-6
    if is_degenerate_triangle(t1, EPSILON2) or is_degenerate_triangle(t2, EPSILON2):
        return False
    n1 = calc_triangle_normal(t1)
    n2 = calc_triangle_normal(t2)
    if n1.length < EPSILON2 or n2.length < EPSILON2:
        return False
    d1_const = -n1.dot(t1[0])
    d2_const = -n2.dot(t2[0])
    dist1 = [n2.dot(v) + d2_const for v in t1]
    dist2 = [n1.dot(v) + d1_const for v in t2]
    if all(d >= 0 for d in dist1) or all(d <= 0 for d in dist1):
        return False
    if all(d >= 0 for d in dist2) or all(d <= 0 for d in dist2):
        return False

    def compute_intersection_points(triangle, dists):
        pts = []
        for i in range(3):
            j = (i + 1) % 3
            di = dists[i]
            dj = dists[j]
            if abs(di) < 1e-8:
                pts.append(triangle[i])
            if di * dj < 0:
                t = di / (di - dj)
                pt = triangle[i] + t * (triangle[j] - triangle[i])
                pts.append(pt)
            elif abs(dj) < 1e-8:
                pts.append(triangle[j])
        unique_pts = []
        for p in pts:
            if not any((p - q).length < 1e-8 for q in unique_pts):
                unique_pts.append(p)
        return unique_pts

    pts1 = compute_intersection_points(t1, dist1)
    pts2 = compute_intersection_points(t2, dist2)
    if len(pts1) < 2 or len(pts2) < 2:
        return False
    d = n1.cross(n2)
    if d.length < 1e-8:
        return False
    d.normalize()
    s1 = [d.dot(p) for p in pts1]
    s2 = [d.dot(p) for p in pts2]
    seg1_min, seg1_max = min(s1), max(s1)
    seg2_min, seg2_max = min(s2), max(s2)
    if seg1_max < seg2_min or seg2_max < seg1_min:
        return False
    return True


def calculate_optimal_rigid_transform(source_points, target_points):
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = centroid_target - R @ centroid_source
    return R, t


def apply_rigid_transform_to_points(points, R, t):
    return (R @ points.T).T + t


def calculate_optimal_similarity_transform(source_points, target_points):
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    source_scale = np.sum(source_centered**2)
    H = source_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    trace_RSH = np.sum(S)
    s = trace_RSH / source_scale if source_scale > 0 else 1.0
    t = centroid_target - s * (R @ centroid_source)
    return s, R, t


def apply_similarity_transform_to_points(points, s, R, t):
    return s * (R @ points.T).T + t


def calculate_optimal_similarity_transform_weighted(source_points, target_points, weights):
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
    centroid_source = np.sum(source_points * weights[:, np.newaxis], axis=0)
    centroid_target = np.sum(target_points * weights[:, np.newaxis], axis=0)
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target
    source_scale = np.sum(weights[:, np.newaxis] * source_centered**2)
    H = (source_centered * weights[:, np.newaxis]).T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    trace_RSH = np.sum(S)
    s = trace_RSH / source_scale if source_scale > 0 else 1.0
    t = centroid_target - s * (R @ centroid_source)
    return s, R, t


def batch_process_vertices_multi_step(
    vertices,
    all_field_points,
    all_delta_positions,
    field_weights,
    field_matrix,
    field_matrix_inv,
    target_matrix,
    target_matrix_inv,
    deform_weights=None,
    rbf_epsilon=0.00001,
    batch_size=1000,
    k=8,
):
    num_vertices = len(vertices)
    num_steps = len(all_field_points)
    cumulative_displacements = np.zeros((num_vertices, 3))
    current_world_positions = np.array([target_matrix @ Vector(v) for v in vertices])
    if deform_weights is None:
        deform_weights = np.ones(num_vertices)
    for step in range(num_steps):
        field_points = all_field_points[step]
        delta_positions = all_delta_positions[step]
        kdtree = cKDTree(field_points, balanced_tree=False, compact_nodes=False)
        step_displacements = np.zeros((num_vertices, 3))
        for start_idx in range(0, num_vertices, batch_size):
            end_idx = min(start_idx + batch_size, num_vertices)
            batch_weights = deform_weights[start_idx:end_idx]
            batch_world = current_world_positions[start_idx:end_idx]
            batch_field = np.array([field_matrix_inv @ Vector(v) for v in batch_world])
            k_use = min(k, len(field_points))
            distances, indices = kdtree.query(batch_field, k=k_use)
            weights = 1.0 / np.sqrt(distances**2 + rbf_epsilon**2)
            weights /= np.sum(weights, axis=1, keepdims=True)
            weighted_deltas = delta_positions[indices] * weights[..., np.newaxis]
            batch_displacements = np.sum(weighted_deltas, axis=1) * batch_weights[:, np.newaxis]
            world_displacements = np.array([field_matrix.to_3x3() @ Vector(v) for v in batch_displacements])
            step_displacements[start_idx:end_idx] = world_displacements
            current_world_positions[start_idx:end_idx] += world_displacements
        cumulative_displacements += step_displacements
    final_world_positions = np.array([target_matrix @ Vector(v) for v in vertices]) + cumulative_displacements
    return final_world_positions


def batch_process_vertices_with_custom_range(
    vertices,
    all_field_points,
    all_delta_positions,
    field_weights,
    field_matrix,
    field_matrix_inv,
    target_matrix,
    target_matrix_inv,
    start_value,
    end_value,
    deform_weights=None,
    rbf_epsilon=0.00001,
    batch_size=1000,
    k=8,
):
    num_vertices = len(vertices)
    num_steps = len(all_field_points)
    cumulative_displacements = np.zeros((num_vertices, 3))
    current_world_positions = np.array([target_matrix @ Vector(v) for v in vertices])
    if deform_weights is None:
        deform_weights = np.ones(num_vertices)
    step_size = 1.0 / num_steps
    processed_steps = []
    for step in range(num_steps):
        step_start = step * step_size
        step_end = (step + 1) * step_size
        if step_start + 0.00001 <= end_value and step_end - 0.00001 >= start_value:
            processed_steps.append((step, step_start, step_end))
    for _step_idx, (step, step_start, step_end) in enumerate(processed_steps):
        field_points = all_field_points[step].copy()
        delta_positions = all_delta_positions[step].copy()
        original_delta_positions = all_delta_positions[step]
        if start_value != step_start:
            if start_value >= step_start + 0.00001:
                adjustment_factor = (start_value - step_start) / step_size
                adjustment_delta = original_delta_positions * adjustment_factor
                field_points += adjustment_delta
                delta_positions -= adjustment_delta
        if end_value != step_end:
            if end_value <= step_end - 0.00001:
                adjustment_factor = (step_end - end_value) / step_size
                adjustment_delta = original_delta_positions * adjustment_factor
                delta_positions -= adjustment_delta
        kdtree = cKDTree(field_points, balanced_tree=False, compact_nodes=False)
        step_displacements = np.zeros((num_vertices, 3))
        for start_idx in range(0, num_vertices, batch_size):
            end_idx = min(start_idx + batch_size, num_vertices)
            batch_weights = deform_weights[start_idx:end_idx]
            batch_world = current_world_positions[start_idx:end_idx]
            batch_field = np.array([field_matrix_inv @ Vector(v) for v in batch_world])
            k_use = min(k, len(field_points))
            distances, indices = kdtree.query(batch_field, k=k_use)
            weights = 1.0 / np.sqrt(distances**2 + rbf_epsilon**2)
            weights /= np.sum(weights, axis=1, keepdims=True)
            weighted_deltas = delta_positions[indices] * weights[..., np.newaxis]
            batch_displacements = np.sum(weighted_deltas, axis=1) * batch_weights[:, np.newaxis]
            world_displacements = np.array([field_matrix.to_3x3() @ Vector(v) for v in batch_displacements])
            step_displacements[start_idx:end_idx] = world_displacements
            current_world_positions[start_idx:end_idx] += world_displacements
        cumulative_displacements += step_displacements
    final_world_positions = np.array([target_matrix @ Vector(v) for v in vertices]) + cumulative_displacements
    return final_world_positions


def batch_process_vertices(
    vertices,
    kdtree,
    field_points,
    delta_positions,
    field_weights,
    field_matrix,
    field_matrix_inv,
    target_matrix,
    target_matrix_inv,
    deform_weights=None,
    batch_size=1000,
    k=8,
):
    num_vertices = len(vertices)
    results = np.zeros((num_vertices, 3))
    if deform_weights is None:
        deform_weights = np.ones(num_vertices)
    rbf_epsilon = 0.00001
    for start_idx in range(0, num_vertices, batch_size):
        end_idx = min(start_idx + batch_size, num_vertices)
        batch_vertices = vertices[start_idx:end_idx]
        batch_weights = deform_weights[start_idx:end_idx]
        batch_world = np.array([target_matrix @ Vector(v) for v in batch_vertices])
        batch_field = np.array([field_matrix_inv @ Vector(v) for v in batch_world])
        distances, indices = kdtree.query(batch_field, k=27)
        for i, (_vert_field, dist, idx) in enumerate(zip(batch_field, distances, indices, strict=False)):
            weights = 1.0 / np.sqrt(dist**2 + rbf_epsilon**2)
            if weights.sum() > 0.0:
                weights /= weights.sum()
            else:
                weights *= 0
            deltas = delta_positions[idx]
            displacement = (deltas * weights[:, np.newaxis]).sum(axis=0) * batch_weights[i]
            world_displacement = field_matrix.to_3x3() @ Vector(displacement)
            results[start_idx + i] = batch_world[i] + world_displacement
    return results


def get_deformation_field_multi_step(field_data_path: str, cache: dict | None = None) -> dict:
    if cache is None:
        cache = {}
    multi_step_key = field_data_path + "_multi_step"
    if multi_step_key in cache:
        return cache[multi_step_key]
    data = np.load(field_data_path, allow_pickle=True)
    if "all_field_points" in data:
        all_field_points = data["all_field_points"]
        all_delta_positions = data["all_delta_positions"]
        num_steps = int(data.get("num_steps", len(all_delta_positions)))
        enable_x_mirror = data.get("enable_x_mirror", False)
        if enable_x_mirror:
            mirrored_field_points = []
            mirrored_delta_positions = []
            for step in range(num_steps):
                field_points = all_field_points[step].copy()
                delta_positions = all_delta_positions[step].copy()
                if len(field_points) > 0:
                    x_positive_mask = field_points[:, 0] > 0.0
                    if np.any(x_positive_mask):
                        mirror_field_points = field_points[x_positive_mask].copy()
                        mirror_delta_positions = delta_positions[x_positive_mask].copy()
                        mirror_field_points[:, 0] *= -1.0
                        mirror_delta_positions[:, 0] *= -1.0
                        combined_field_points = np.vstack([field_points, mirror_field_points])
                        combined_delta_positions = np.vstack([delta_positions, mirror_delta_positions])
                        mirrored_field_points.append(combined_field_points)
                        mirrored_delta_positions.append(combined_delta_positions)
                    else:
                        mirrored_field_points.append(field_points)
                        mirrored_delta_positions.append(delta_positions)
                else:
                    mirrored_field_points.append(field_points)
                    mirrored_delta_positions.append(delta_positions)
            all_field_points = mirrored_field_points
            all_delta_positions = mirrored_delta_positions
    elif "field_points" in data and "all_delta_positions" in data:
        field_points = data["field_points"]
        all_delta_positions = data["all_delta_positions"]
        num_steps = int(data.get("num_steps", len(all_delta_positions)))
        all_field_points = [field_points for _ in range(num_steps)]
        num_steps = 1
    if "weights" in data:
        field_weights = data["weights"]
    else:
        field_weights = np.ones(len(all_field_points[0]) if len(all_field_points) > 0 else 0)
    world_matrix = Matrix(data["world_matrix"])
    world_matrix_inv = world_matrix.inverted()
    k_neighbors = 8
    rbf_epsilon = float(data.get("rbf_epsilon", 0.00001))
    field_info = {
        "data": data,
        "all_field_points": all_field_points,
        "all_delta_positions": all_delta_positions,
        "num_steps": num_steps,
        "field_weights": field_weights,
        "world_matrix": world_matrix,
        "world_matrix_inv": world_matrix_inv,
        "kdtree_query_k": k_neighbors,
        "rbf_epsilon": rbf_epsilon,
        "use_multi_step": num_steps > 1,
    }
    cache[multi_step_key] = field_info
    return field_info
