import json
import logging
import os

import bpy
from mathutils import Vector

from domain.models import RetargetConfig, RetargetContext
from infrastructure.blender import armature, blendshapes, weights
from infrastructure.blender import deformation as mesh_deformation
from infrastructure.blender import mesh as mesh_ops
from infrastructure.blender import ops as blender_ops

logger = logging.getLogger(__name__)


class OutfitRetargeter:
    def execute(self, args):
        config_pairs = self._load_config_pairs(args)
        total_pairs = len(config_pairs)
        for pair_index, config_pair in enumerate(config_pairs):
            context = RetargetContext(config=config_pair, pair_index=pair_index, total_pairs=total_pairs)
            self._run_pipeline(args, context)

    @staticmethod
    def _report_progress(progress: float, message: str = ""):
        logger.info(f"Progress: {progress:.2f}")
        if message:
            logger.info(f"Status: {message}")

    def _run_pipeline(self, args, ctx: RetargetContext):
        steps = [
            (self._step_initialize_scene, "Initializing Scene", 0.05),
            (self._step_process_avatars, "Processing Avatars", 0.10),
            (self._step_apply_initial_pose, "Applying Initial Pose", 0.15),
            (self._step_setup_clothing, "Setting up Clothing", 0.20),
            (self._step_transfer_weights, "Transferring Weights", 0.30),
            (self._step_process_meshes, "Processing Meshes", 0.40),
            (self._step_finalize, "Finalizing", 0.95),
        ]

        for step_func, status_msg, progress_val in steps:
            self._report_progress(progress_val, status_msg)
            step_func(args, ctx)

        self._report_progress(1.00, "Done")

    def _step_initialize_scene(self, args, ctx: RetargetContext):
        bpy.ops.wm.open_mainfile(filepath=args.base)

    def _step_process_avatars(self, args, ctx: RetargetContext):
        with open(ctx.config.base_avatar_data, encoding="utf-8") as f:
            ctx.base_avatar_data = json.load(f)
        bpy.ops.import_scene.fbx(filepath=ctx.config.base_fbx, use_anim=False)
        ctx.base_mesh = bpy.data.objects.get(ctx.base_avatar_data.get("meshName"))
        ctx.base_armature = next((mod.object for mod in ctx.base_mesh.modifiers if mod.type == "ARMATURE"), None)
        if ctx.base_armature:
            for bone in ctx.base_armature.data.bones:
                bone.inherit_scale = "ALIGNED"

        bpy.ops.import_scene.fbx(filepath=ctx.config.input_fbx, use_anim=True)
        bpy.ops.object.mode_set(mode="OBJECT")
        for action in bpy.data.actions:
            bpy.data.actions.remove(action)

        clothing_avatar_data_path = ctx.config.config_data.get("clothingAvatarData")
        clothing_dir = os.path.dirname(os.path.abspath(ctx.config.config_path))
        if clothing_avatar_data_path and not os.path.isabs(clothing_avatar_data_path):
            clothing_avatar_data_path = os.path.normpath(os.path.join(clothing_dir, clothing_avatar_data_path))

        with open(clothing_avatar_data_path, encoding="utf-8") as f:
            ctx.clothing_avatar_data = json.load(f)

        ctx.clothing_armature = next(
            (obj for obj in bpy.data.objects if obj.type == "ARMATURE" and obj.name != ctx.base_armature.name), None
        )
        ctx.clothing_meshes = [
            obj
            for obj in bpy.data.objects
            if obj.type == "MESH"
            and obj.name != ctx.base_mesh.name
            and any(mod.type == "ARMATURE" and mod.object == ctx.clothing_armature for mod in obj.modifiers)
        ]

        if ctx.pair_index == 0 and ctx.config.name_conv:
            with open(ctx.config.name_conv, encoding="utf-8") as f:
                ctx.name_conv_data = json.load(f)

    def _step_apply_initial_pose(self, args, ctx: RetargetContext):
        if ctx.config.init_pose:
            blender_ops.apply_pose_from_json(ctx.base_armature, ctx.config.init_pose, ctx.base_avatar_data)
            armature.apply_pose_as_rest(ctx.base_armature)

    def _step_setup_clothing(self, args, ctx: RetargetContext):
        if ctx.config.shape_name_file:
            blendshapes.sync_shape_key_names_from_file(ctx.clothing_meshes, ctx.config.shape_name_file)

        if ctx.pair_index == 0:
            for mesh_obj in ctx.clothing_meshes:
                weights.fix_invalid_weights(mesh_obj)
            blender_ops.record_armature_info(
                ctx.clothing_armature, ctx.clothing_meshes, ctx.clothing_avatar_data, "before", ctx.cache
            )

        if ctx.config.hips_position:
            armature.adjust_armature_hips_position(
                ctx.clothing_armature, ctx.config.hips_position, ctx.clothing_avatar_data
            )

        for mesh_obj in ctx.clothing_meshes:
            mesh_ops.cleanup_mesh(mesh_obj)

    def _step_transfer_weights(self, args, ctx: RetargetContext):
        if hasattr(bpy.context.scene, "robust_weight_transfer_settings"):
            bpy.context.scene.robust_weight_transfer_settings.source_object = ctx.base_mesh

        weights.remove_empty_vertex_groups(ctx.base_mesh)
        weights.update_base_avatar_weights(
            ctx.base_mesh, ctx.clothing_armature, ctx.base_avatar_data, ctx.clothing_avatar_data
        )
        mesh_ops.create_overlapping_vertices_attributes(ctx.clothing_meshes, ctx.base_avatar_data)

    def _step_process_meshes(self, args, ctx: RetargetContext):
        for obj in ctx.clothing_meshes:
            armature.create_hinge_bone_group(obj, ctx.clothing_armature, ctx.clothing_avatar_data)
            blendshapes.reset_shape_keys(obj)
            weights.remove_empty_vertex_groups(obj)
            weights.process_bone_weight_consolidation(obj, ctx.clothing_avatar_data)
            weights.propagate_bone_weights(obj)

            if ctx.config.field_data:
                mesh_deformation.retarget_mesh(obj, ctx)

    def _step_finalize(self, args, ctx: RetargetContext):
        if ctx.pair_index == ctx.total_pairs - 1 and ctx.config.output_fbx:
            bpy.ops.export_scene.fbx(filepath=ctx.config.output_fbx, use_selection=True)

    def _load_config_pairs(self, args):
        base_fbx_paths = [path.strip() for path in args.base_fbx.split(";")]
        config_paths = [path.strip() for path in args.config.split(";")]
        pairs = []
        for i in range(len(base_fbx_paths)):
            config_path = config_paths[i]
            base_fbx = base_fbx_paths[i]
            with open(config_path, encoding="utf-8") as f:
                config_data = json.load(f)
            config_dir = os.path.dirname(os.path.abspath(config_path))

            def resolve_path(path_key):
                p = config_data.get(path_key)
                if p and not os.path.isabs(p):
                    return os.path.normpath(os.path.join(config_dir, p))
                return p

            pair = RetargetConfig(
                config_path=config_path,
                config_data=config_data,
                base_fbx=base_fbx,
                input_fbx=resolve_path("inputClothingFbxPath") or args.input,
                output_fbx=args.output,
                init_pose=args.init_pose,
                base_avatar_data=resolve_path("baseAvatarData"),
                target_meshes=args.target_meshes,
                mesh_renderers=args.mesh_renderers,
                blend_shapes=args.blend_shapes,
                blend_shape_values=args.blend_shape_values,
                blend_shape_mappings=args.blend_shape_mappings,
                name_conv=args.name_conv,
                cloth_metadata=args.cloth_metadata,
                mesh_material_data=args.mesh_material_data,
                shape_name_file=args.shape_name_file,
                no_subdivision=args.no_subdivision,
                no_triangle=args.no_triangle,
                field_data=resolve_path("fieldData") or resolve_path("field_data"),
            )

            hips_pos_str = args.hips_position
            if hips_pos_str:
                parts = hips_pos_str.split(",")
                if len(parts) == 3:
                    pair.hips_position = Vector((float(parts[0]), float(parts[1]), float(parts[2])))

            with open(pair.base_avatar_data, encoding="utf-8") as f:
                pair.base_avatar_data_content = json.load(f)
            pairs.append(pair)

        if len(pairs) > 1:
            for i in range(len(pairs) - 1):
                blendshapes.process_blendshape_transitions(pairs[i], pairs[i + 1])
        return pairs
