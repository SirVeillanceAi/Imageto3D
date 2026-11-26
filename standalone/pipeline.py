"""Builds ComfyUI API graphs for the Imageto3D Gradio app.

The node settings mirror two tested workflows:
  - ComfyUI's official template "Pixal3D: Image to Model" (TRELLIS.2 / Pixal3D + PBR texture)
  - PixelArtistry-Hi3D 01/03 (double remesh, then Hi3D 512 -> 1024 -> 1536)
Node ids are readable names so progress messages can show what is running.
"""

import os

TRELLIS2_UNET = "trellis_2_int8_convrot.safetensors"
PIXAL3D_UNET = "pixal3d_int8_convrot.safetensors"
PIXAL3D_MV_UNET = "pixal3d_multiview_int8_convrot.safetensors"
SHAPE_VAE = "trellis_2_shape_vae_bf16.safetensors"
TEXTURE_VAE = "trellis_2_texture_vae_bf16.safetensors"
DINO_V3 = "dino_v3_L_naf_fp32.safetensors"
MOGE = "moge_2_vitl_normal_fp16.safetensors"
BIREFNET = "birefnet.safetensors"
DINO_V2 = "dinov2-large.safetensors"
CLIP_L = "clip-vit-large-patch14.safetensors"
HI3D_CKPT = "hi3d_v1.safetensors" if os.environ.get("HI3D_PRECISION", "int8") == "fp16" else "hi3d_v1_int8.safetensors"
HI3D_LADDER = ["512", "1024", "1536"]

LABELS = {}  # node id -> human label, filled while building


class Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, node_id, class_type, label=None, **inputs):
        self.nodes[node_id] = {"class_type": class_type, "inputs": inputs}
        LABELS[node_id] = label or class_type
        return node_id

    @staticmethod
    def out(node_id, index=0):
        return [node_id, index]


def _remesh(g, node_id, mesh, resolution, label):
    # DynamicCombo sub-inputs are sent as "<combo>.<field>"
    return g.add(node_id, "RemeshMesh", label, mesh=mesh, resolution=resolution,
                 sign_mode="solid", **{"sign_mode.fill": 20.0, "sign_mode.qef": False},
                 band=1.0, project_back=0.0, fix_poles=False, smooth_iters=0,
                 drop_small_components=0.01, precluster_max_verts=20000000)


def _image_and_mask(g, image_name, remove_background):
    g.add("load_image", "LoadImage", "Load image", image=image_name)
    if remove_background:
        g.add("bg_model", "LoadBackgroundRemovalModel", "Load BiRefNet", bg_removal_name=BIREFNET)
        g.add("remove_bg", "RemoveBackground", "Remove background",
              bg_removal_model=g.out("bg_model"), image=g.out("load_image"))
        return g.out("load_image"), g.out("remove_bg")
    return g.out("load_image"), g.out("load_image", 1)


def _hi3d_chain(g, coarse_mesh, image, mask, target, seed, prefix):
    """Double remesh, then Hi3D passes up to `target` (512/1024/1536). Returns save node ids."""
    g.add("hi3d_crop", "ImageCropToMask", "Crop for Hi3D", images=image, masks=mask,
          width=1024, height=1024, pad_factor=1.2, grow_mask=0, background="#ffffff")
    g.add("hi3d_ckpt", "Hi3DCheckpointLoader", "Load Hi3D", ckpt_name=HI3D_CKPT)
    g.add("hi3d_dino", "CLIPVisionLoader", "Load DINOv2", clip_name=DINO_V2)
    g.add("hi3d_clip", "CLIPVisionLoader", "Load CLIP ViT-L", clip_name=CLIP_L)
    g.add("hi3d_cond", "Hi3DConditioning", "Hi3D conditioning",
          dino_vision=g.out("hi3d_dino"), clip_vision=g.out("hi3d_clip"), image=g.out("hi3d_crop"))
    _remesh(g, "hi3d_remesh_1", coarse_mesh, 1024, "Remesh 1/2")
    _remesh(g, "hi3d_remesh_2", g.out("hi3d_remesh_1"), 1024, "Remesh 2/2")

    mesh, saves = g.out("hi3d_remesh_2"), []
    for res in HI3D_LADDER[:HI3D_LADDER.index(str(target)) + 1]:
        g.add(f"hi3d_stage_{res}", "Hi3DRefineStage", f"Hi3D stage {res}",
              positive=g.out("hi3d_cond"), negative=g.out("hi3d_cond", 1), mesh=mesh, resolution=res)
        g.add(f"hi3d_sample_{res}", "Hi3DSampler", f"Hi3D sampling {res}", model=g.out("hi3d_ckpt"),
              positive=g.out(f"hi3d_stage_{res}"), negative=g.out(f"hi3d_stage_{res}", 1),
              latent=g.out(f"hi3d_stage_{res}", 2), seed=seed, steps=25)
        g.add(f"hi3d_decode_{res}", "VAEDecodeHi3D", f"Hi3D decode {res}",
              samples=g.out(f"hi3d_sample_{res}"), vae=g.out("hi3d_ckpt", 1))
        saves.append(g.add(f"save_hi3d_{res}", "SaveGLB", f"Save Hi3D {res}",
                           mesh=g.out(f"hi3d_decode_{res}"), filename_prefix=f"{prefix}_hi3d_{res}"))
        mesh = g.out(f"hi3d_decode_{res}")
    return saves


def image_to_3d(image_name, model="trellis2", remove_background=True, seed=42, resolution=1024,
                texture=True, texture_size=2048, face_count=700000, hi3d="off", prefix="imageto3d/out"):
    """TRELLIS.2 or Pixal3D -> coarse mesh, optional PBR texture, optional Hi3D refinement."""
    LABELS.clear()
    g = Graph()
    image, mask = _image_and_mask(g, image_name, remove_background)

    g.add("crop", "ImageCropToMask", "Crop image", images=image, masks=mask,
          width=1024, height=1024, pad_factor=1.1, grow_mask=0, background="#000000")
    g.add("unet", "UNETLoader", "Load TRELLIS.2" if model == "trellis2" else "Load Pixal3D",
          unet_name=TRELLIS2_UNET if model == "trellis2" else PIXAL3D_UNET, weight_dtype="default")
    g.add("shape_vae", "VAELoader", "Load shape VAE", vae_name=SHAPE_VAE)
    g.add("dino_v3", "CLIPVisionLoader", "Load DINOv3", clip_name=DINO_V3)

    if model == "trellis2":
        g.add("cond", "Trellis2Conditioning", "TRELLIS.2 conditioning",
              clip_vision_model=g.out("dino_v3"), image=g.out("crop"))
    else:
        g.add("moge", "LoadMoGeModel", "Load MoGe", model_name=MOGE)
        g.add("moge_run", "MoGeInference", "Estimate camera (MoGe)", moge_model=g.out("moge"),
              image=g.out("crop"), resolution_level=9, fov_x_degrees=0.0, batch_size=4,
              force_projection=True, apply_mask=True, refine_steps=3)
        g.add("fov", "MoGeGeometryToFOV", "Camera FOV", moge_geometry=g.out("moge_run"),
              axis="horizontal", unit="degrees")
        g.add("cond", "Pixal3DConditioning", "Pixal3D conditioning", clip_vision_model=g.out("dino_v3"),
              image=g.out("crop"), camera_angle_x=g.out("fov"))
    pos, neg = g.out("cond"), g.out("cond", 1)

    _generate(g, pos, neg, seed, resolution, texture, texture_size, face_count, prefix)
    if hi3d != "off":
        _hi3d_chain(g, g.out("shape_decode"), image, mask, hi3d, seed, prefix)
    return g.nodes


def multiview_to_3d(views, remove_background=True, seed=42, fov=20.0, resolution=1024, texture=True,
                    texture_size=2048, face_count=700000, hi3d="off", prefix="imageto3d/multiview"):
    """Pixal3D multi-view: {"front"|"left"|"back"|"right": image name}, views 90 degrees apart."""
    LABELS.clear()
    g = Graph()
    if remove_background:
        g.add("bg_model", "LoadBackgroundRemovalModel", "Load BiRefNet", bg_removal_name=BIREFNET)
    cropped = {}
    for side in ("front", "left", "back", "right"):
        if not views.get(side):
            continue
        g.add(f"load_{side}", "LoadImage", f"Load {side} view", image=views[side])
        mask = g.out(f"load_{side}", 1)
        if remove_background:
            mask = g.out(g.add(f"remove_bg_{side}", "RemoveBackground", f"Remove background ({side})",
                               bg_removal_model=g.out("bg_model"), image=g.out(f"load_{side}")))
        cropped[side] = g.out(g.add(f"crop_{side}", "ImageCropToMask", f"Crop {side} view",
                                    images=g.out(f"load_{side}"), masks=mask, width=1024, height=1024,
                                    pad_factor=1.1, grow_mask=0, background="#000000"))
    g.add("unet", "UNETLoader", "Load Pixal3D multi-view", unet_name=PIXAL3D_MV_UNET, weight_dtype="default")
    g.add("shape_vae", "VAELoader", "Load shape VAE", vae_name=SHAPE_VAE)
    g.add("dino_v3", "CLIPVisionLoader", "Load DINOv3", clip_name=DINO_V3)
    g.add("cond", "Pixal3DMultiViewConditioning", "Pixal3D multi-view conditioning",
          clip_vision_model=g.out("dino_v3"), fov=float(fov), **cropped)
    _generate(g, g.out("cond"), g.out("cond", 1), seed, resolution, texture, texture_size, face_count, prefix)
    if hi3d != "off":
        first = next(side for side in ("front", "left", "back", "right") if side in cropped)
        mask = g.out(f"remove_bg_{first}") if remove_background else g.out(f"load_{first}", 1)
        _hi3d_chain(g, g.out("shape_decode"), g.out(f"load_{first}"), mask, hi3d, seed, prefix)
    return g.nodes


def _generate(g, pos, neg, seed, resolution, texture, texture_size, face_count, prefix):
    """Shared TRELLIS.2-family stages: structure -> shape 512 -> upsample -> optional PBR texture.
    Expects "unet" and "shape_vae" nodes in the graph."""
    # Structure stage
    g.add("struct_cfg", "CFGOverride", model=g.out("unet"), cfg=1.0, start_percent=0.667, end_percent=1.0)
    g.add("struct_rescale", "RescaleCFG", model=g.out("struct_cfg"), multiplier=0.7)
    g.add("struct_shift", "ModelSamplingSD3", model=g.out("struct_rescale"), shift=5.0)
    g.add("struct_empty", "EmptyTrellis2LatentStructure", batch_size=1)
    g.add("struct_sample", "KSampler", "Sampling structure", model=g.out("struct_shift"), seed=seed + 14,
          steps=12, cfg=7.5, sampler_name="euler", scheduler="normal",
          positive=pos, negative=neg, latent_image=g.out("struct_empty"), denoise=1.0)
    g.add("struct_decode", "VaeDecodeStructureTrellis2", "Decode structure",
          samples=g.out("struct_sample"), vae=g.out("shape_vae"), resolution="32")

    # Shape stage 512, then upsample
    g.add("shape_cfg", "CFGOverride", model=g.out("unet"), cfg=1.0, start_percent=0.769, end_percent=1.0)
    g.add("shape_rescale", "RescaleCFG", model=g.out("shape_cfg"), multiplier=0.5)
    g.add("shape_stage", "Trellis2ShapeStage", positive=pos, negative=neg, voxel=g.out("struct_decode"))
    g.add("shape_sample", "KSampler", "Sampling shape 512", model=g.out("shape_rescale"), seed=seed,
          steps=20, cfg=7.5, sampler_name="euler", scheduler="normal",
          positive=g.out("shape_stage"), negative=g.out("shape_stage", 1),
          latent_image=g.out("shape_stage", 2), denoise=1.0)
    g.add("upsample", "Trellis2UpsampleStage", f"Upsample to {resolution}", positive=g.out("shape_stage"),
          negative=g.out("shape_stage", 1), shape_latent=g.out("shape_sample"), vae=g.out("shape_vae"),
          target_resolution=int(resolution))
    g.add("hires_sample", "KSampler", f"Sampling shape {resolution}", model=g.out("shape_rescale"), seed=seed,
          steps=12, cfg=7.5, sampler_name="euler", scheduler="simple",
          positive=g.out("upsample"), negative=g.out("upsample", 1),
          latent_image=g.out("upsample", 2), denoise=1.0)
    g.add("shape_decode", "VaeDecodeShapeTrellis", "Decode mesh", samples=g.out("hires_sample"), vae=g.out("shape_vae"))
    g.add("save_shape", "SaveGLB", "Save untextured mesh", mesh=g.out("shape_decode"), filename_prefix=f"{prefix}_shape")

    if texture:
        g.add("tex_vae", "VAELoader", "Load texture VAE", vae_name=TEXTURE_VAE)
        g.add("tex_stage", "Trellis2TextureStage", positive=g.out("upsample"), negative=g.out("upsample", 1),
              shape_latent=g.out("hires_sample"))
        g.add("tex_sample", "KSampler", "Sampling texture", model=g.out("unet"), seed=seed + 1,
              steps=12, cfg=1.0, sampler_name="euler", scheduler="normal",
              positive=g.out("tex_stage"), negative=g.out("tex_stage", 1),
              latent_image=g.out("tex_stage", 2), denoise=1.0)
        g.add("tex_decode", "VaeDecodeTextureTrellis", "Decode texture", samples=g.out("tex_sample"),
              vae=g.out("tex_vae"), shape_subdivides=g.out("shape_decode", 1))
        _remesh(g, "tex_remesh", g.out("shape_decode"), 768, "Remesh")
        g.add("tex_decimate", "DecimateMesh", "Decimate", mesh=g.out("tex_remesh"),
              target_face_count=int(face_count), placement_mode="midpoint")
        g.add("tex_smooth", "MeshSmoothNormals", mesh=g.out("tex_decimate"), crease_angle=180.0)
        g.add("tex_unwrap", "UnwrapMesh", "UV unwrap", mesh=g.out("tex_smooth"), segmenter="pec",
              resolution=int(texture_size), padding=1, weld_distance=0.0002)
        g.add("tex_bake", "BakeTextureFromVoxel", "Bake PBR textures", mesh=g.out("tex_unwrap"),
              voxel_colors=g.out("tex_decode"), texture_size=int(texture_size), reference_mesh=g.out("shape_decode"))
        g.add("tex_normal", "BakeNormalMapFromMesh", "Bake normal map", low_poly=g.out("tex_unwrap"),
              high_poly=g.out("tex_remesh"), resolution=int(texture_size), cage_distance=0.05, ignore_backfaces=True)
        g.add("tex_ao", "BakeAmbientOcclusion", "Bake AO", low_poly=g.out("tex_unwrap"), high_poly=g.out("tex_remesh"),
              resolution=1024, samples=64, max_distance=0.71, strength=1.0, bias=0.01)
        g.add("tex_apply", "ApplyTextureToMesh", "Apply textures", mesh=g.out("tex_unwrap"),
              base_color=g.out("tex_bake"), metallic=g.out("tex_bake", 1), roughness=g.out("tex_bake", 2),
              occlusion=g.out("tex_ao"), normal_map=g.out("tex_normal"))
        g.add("tex_final", "MeshSmoothNormals", mesh=g.out("tex_apply"), crease_angle=180.0)
        g.add("save_textured", "SaveGLB", "Save textured mesh", mesh=g.out("tex_final"), filename_prefix=f"{prefix}_textured")



def refine_mesh(image_name, mesh_name, target="1536", remove_background=True, seed=42, prefix="imageto3d/refine"):
    """Hi3D only: your own coarse mesh (in ComfyUI/input/3d) + its reference image."""
    LABELS.clear()
    g = Graph()
    image, mask = _image_and_mask(g, image_name, remove_background)
    g.add("load_mesh", "Load3DAdvanced", "Load mesh", model_file=f"3d/{mesh_name}",
          viewport_state={}, width=1024, height=1024)
    g.add("mesh", "Get3DComponents", "Read mesh", model_3d=g.out("load_mesh"))
    _hi3d_chain(g, g.out("mesh"), image, mask, target, seed, prefix)
    return g.nodes
