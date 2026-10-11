"""Picture -> TRELLIS -> a GLB and a sheet of six views. Run by generate.sh with TRELLIS's own Python.

    generate.py PICTURE OUT_DIR SEED --home ~/trellis [--steps N] [--simplify 0.95] [--texture 1024] [--more ...]

It imports nothing from the server: the only Python it can see is the TRELLIS environment's.
The "=== stage" lines it prints are read by the server; their first word is the contract.
"""
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("picture")
parser.add_argument("out_dir")
parser.add_argument("seed", type=int)
parser.add_argument("--home", required=True, help="where TRELLIS is installed, e.g. /home/me/trellis")
parser.add_argument("--steps", type=int, default=0, help="sampler steps for shape and surface; 0 keeps TRELLIS's default")
parser.add_argument("--simplify", type=float, default=0.95, help="share of triangles removed on export")
parser.add_argument("--texture", type=int, default=1024, help="texture size in pixels")
parser.add_argument("--more", nargs="*", default=[], help="further pictures of the SAME object, from other sides")
args = parser.parse_args()

os.environ["ATTN_BACKEND"] = "xformers"
os.environ["SPCONV_ALGO"] = "native"   # spconv's default algorithm crashes on RTX 50-series
sys.path.insert(0, os.path.join(args.home, "TRELLIS"))


def stage(name):
    print("=== %s %s" % (name, time.strftime("%H:%M:%S")), flush=True)


def has_card(vertices):
    """True when the mesh looks like a picture on a flat sheet, which TRELLIS sometimes makes of a flat drawing.

    Either the whole mesh is almost flat, or some axis-aligned plane has vertices in three of the
    bounding box's four corners, which a solid object very rarely does.
    """
    v = vertices.detach().cpu().numpy()
    low, high = v.min(0), v.max(0)
    if (high - low).min() < 0.1 * (high - low).max():
        return True
    for normal in range(3):
        u, w = [axis for axis in range(3) if axis != normal]
        corners = 0
        for cu in (low[u], high[u]):
            for cw in (low[w], high[w]):
                near = ((np.abs(v[:, u] - cu) < 0.04 * (high[u] - low[u]))
                        & (np.abs(v[:, w] - cw) < 0.04 * (high[w] - low[w])))
                corners += bool(near.any())
        if corners >= 3:
            return True
    return False


stage("load")
import numpy as np  # noqa: E402
import torch  # noqa: E402
from xformers.ops.fmha import dispatch as _xd  # noqa: E402

_xd._set_use_fa3(False)   # the flash3 kernel errors on RTX 50-series

from PIL import Image  # noqa: E402
from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: E402
from trellis.utils import postprocessing_utils, render_utils  # noqa: E402

stem = os.path.splitext(os.path.basename(args.picture))[0] + "_s%d" % args.seed
os.makedirs(args.out_dir, exist_ok=True)
result = {}

t = time.time()
pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipe.cuda()
result["load_seconds"] = round(time.time() - t, 1)

stage("generate")
t = time.time()
sampling = {}
if args.steps:
    sampling = {"sparse_structure_sampler_params": {"steps": args.steps},
                "slat_sampler_params": {"steps": args.steps}}
if args.more:
    # TRELLIS's multi-picture mode: every picture conditions the same model, in turn, a step each.
    # It was not trained for this, so it helps only when the pictures really are one object.
    pictures = [Image.open(path) for path in [args.picture] + args.more]
    out = pipe.run_multi_image(pictures, seed=args.seed, mode="stochastic", **sampling)
else:
    out = pipe.run(Image.open(args.picture), seed=args.seed, **sampling)
vertices = out["mesh"][0].vertices
result["generate_seconds"] = round(time.time() - t, 1)
result["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
result["flat_card"] = bool(has_card(vertices))
result["size"] = [round(x, 3) for x in (vertices.max(0).values - vertices.min(0).values).tolist()]

stage("views")
t = time.time()
yaws = [np.pi * k / 4 for k in (0, 1, 2, 3, 4, 6)]
ext, intr = render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, [0.25] * 6, [1.6] * 6, [40] * 6)
opt = {"resolution": 512, "bg_color": (0.35, 0.35, 0.38)}
color = render_utils.render_frames(out["gaussian"][0], ext, intr, opt, verbose=False)["color"]
normal = render_utils.render_frames(out["mesh"][0], ext, intr, opt, verbose=False)["normal"]
rows = [np.concatenate(color, 1), np.concatenate(normal, 1)]
Image.fromarray(np.concatenate(rows, 0)).resize((1920, 640)).save(os.path.join(args.out_dir, stem + "_views.png"))
result["views_seconds"] = round(time.time() - t, 1)

stage("export")
t = time.time()
glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0], simplify=args.simplify, texture_size=args.texture)
glb.export(os.path.join(args.out_dir, stem + ".glb"))
result["export_seconds"] = round(time.time() - t, 1)

with open(os.path.join(args.out_dir, stem + "_result.json"), "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
