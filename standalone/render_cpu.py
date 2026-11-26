"""CPU-only preview renderer for GLB/GLTF/OBJ/STL meshes.

No GPU, OpenGL or WebGL needed: the surface is sampled into a dense point cloud
(with base-color texture or vertex colors), lit, and splatted into a z-buffer with
numpy. Used for preview images when the viewing device has no WebGL.
"""
import io
import math
import threading
from collections import OrderedDict

import numpy as np
import trimesh
from PIL import Image

BACKGROUND = (38, 38, 38)
_cache = OrderedDict()   # path -> (mtime, points, normals, colors)
_lock = threading.Lock()


def _mesh_colors(mesh, faces, bary):
    """Linear-RGB color for each sample (face index + barycentric coords)."""
    vis = mesh.visual
    n = len(faces)
    col = None
    try:
        if vis.kind == "texture" and getattr(vis, "uv", None) is not None:
            mat = vis.material
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            factor = getattr(mat, "baseColorFactor", None)
            if img is not None:
                uv = np.einsum("ij,ijk->ik", bary, vis.uv[mesh.faces[faces]])
                col = trimesh.visual.color.uv_to_color(uv, img)[:, :3] / 255.0
            elif factor is not None:
                col = np.tile(np.asarray(factor, float)[:3] / 255.0, (n, 1))
            if col is not None and img is not None and factor is not None:
                col = col * (np.asarray(factor, float)[:3] / 255.0)
        elif vis.kind == "vertex":
            vc = vis.vertex_colors[:, :3].astype(float) / 255.0
            col = np.einsum("ij,ijk->ik", bary, vc[mesh.faces[faces]])
        elif vis.kind == "face":
            col = vis.face_colors[faces, :3].astype(float) / 255.0
    except Exception:
        col = None
    if col is None:  # untextured: light clay
        col = np.full((n, 3), 0.78)
    return np.clip(col, 0, 1) ** 2.2


def memory_budget_mb():
    """RAM the renderer may use: cgroup limit minus what is in use, else /proc/meminfo."""
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        with open("/sys/fs/cgroup/memory.current") as f:
            used = int(f.read().strip())
        if limit != "max":
            return max(0, (int(limit) - used) // 2**20)
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
        return info.get("MemAvailable", 4 * 2**20) // 1024
    except OSError:
        return 4096


def sample_count(requested=1_200_000):
    """Fewer points on small pods (RunPod's 0-GPU mode has very little RAM)."""
    mb = memory_budget_mb()
    if mb < 600:
        return 250_000
    if mb < 1500:
        return 500_000
    return requested


class NotEnoughMemory(RuntimeError):
    pass


def _check_memory(path, count):
    """Refuse instead of getting the whole app OOM-killed on tiny pods."""
    import os
    with _lock:
        hit = _cache.get(path)
    if hit and hit[0] == os.path.getmtime(path):
        return
    need = os.path.getsize(path) / 2**20 * 8 + count * 64 / 2**20 + 60
    have = memory_budget_mb()
    if have < need:
        raise NotEnoughMemory(
            f"This pod has about {have} MB of free RAM; the CPU preview of this {os.path.getsize(path) // 2**20} MB "
            f"model needs about {need:.0f} MB. Use the 3D viewer in your browser, download the file, "
            "or open it on a pod with a GPU (more RAM).")


def _sample(path, count):
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = [g for g in loaded.dump() if isinstance(g, trimesh.Trimesh) and len(g.faces)]
    del loaded
    if not meshes:
        raise ValueError("No triangle meshes in file")
    # face areas in float32, chunked, to keep peak memory low on big meshes
    areas_per_mesh = []
    for m in meshes:
        v = np.asarray(m.vertices, np.float32)
        f = np.asarray(m.faces)
        a = np.empty(len(f), np.float32)
        for i in range(0, len(f), 500_000):
            t = v[f[i:i + 500_000]]
            a[i:i + 500_000] = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        areas_per_mesh.append(a)
    totals = np.array([a.sum() for a in areas_per_mesh], float)
    per_mesh = np.maximum((count * totals / max(totals.sum(), 1e-12)).astype(int), 1)
    rng = np.random.default_rng(0)
    pts, nrm, col = [], [], []
    for mesh, a, k in zip(meshes, areas_per_mesh, per_mesh):
        cdf = np.cumsum(a, dtype=np.float64)
        faces = np.minimum(np.searchsorted(cdf, rng.random(k) * cdf[-1]), len(a) - 1)
        r1, r2 = rng.random(k, np.float32), rng.random(k, np.float32)
        s = np.sqrt(r1)
        bary = np.stack([1 - s, s * (1 - r2), s * r2], axis=1)
        tri = np.asarray(mesh.vertices, np.float32)[mesh.faces[faces]]
        pts.append(np.einsum("ij,ijk->ik", bary, tri))
        nrm.append(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))   # flat normals: no full-mesh pass
        col.append(_mesh_colors(mesh, faces, bary).astype(np.float32))
        del tri, cdf
    pts, nrm, col = np.concatenate(pts), np.concatenate(nrm), np.concatenate(col)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    lo, hi = pts.min(0), pts.max(0)
    pts = (pts - (lo + hi) / 2) / (np.linalg.norm(hi - lo) / 2 + 1e-12)   # fit a unit sphere
    if np.mean(np.sum(nrm * pts, axis=1)) < 0:   # normals point inward: flip so culling works
        nrm = -nrm
    return pts.astype(np.float32), nrm.astype(np.float32), col.astype(np.float32)


def _points(path, count):
    import os
    mtime = os.path.getmtime(path)
    with _lock:
        hit = _cache.get(path)
        if hit and hit[0] == mtime and len(hit[1]) >= 0.9 * count:
            _cache.move_to_end(path)
            return hit[1:]
    data = _sample(path, count)
    with _lock:
        _cache[path] = (mtime, *data)
        while len(_cache) > (2 if count >= 1_000_000 else 1):
            _cache.popitem(last=False)
    return data


def render(path, azimuth=0.0, elevation=15.0, size=768, count=None):
    """Render one view. Azimuth 0 looks at the model's front (+Z, glTF convention)."""
    count = count or sample_count()
    if count < 1_000_000:
        size = min(size, 640)
    _check_memory(path, count)
    pts, nrm, col = _points(path, count)
    a, e = math.radians(azimuth), math.radians(elevation)
    # camera orbit: rotate the model by -azimuth about Y, then tilt by elevation about X
    ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]], np.float32)
    rx = np.array([[1, 0, 0], [0, math.cos(e), -math.sin(e)], [0, math.sin(e), math.cos(e)]], np.float32)
    rot = rx @ ry
    p = pts @ rot.T
    n = nrm @ rot.T

    ss = 2                         # supersample, then average down
    res = size * ss
    scale = res * 0.46
    front = n[:, 2] > -0.15                   # back-face culling (keeps near-silhouette points)
    p, n, col = p[front], n[front], col[front]
    x = np.round(p[:, 0] * scale + res / 2).astype(np.int64)
    y = np.round(res / 2 - p[:, 1] * scale).astype(np.int64)
    z = p[:, 2]

    # lighting (linear space): headlight + key light from upper left, two-sided
    key = np.array([-0.5, 0.7, 0.6], np.float32); key /= np.linalg.norm(key)
    head = np.array([0, 0, 1], np.float32)
    shade = 0.30 + 0.45 * np.abs(n @ head) + 0.45 * np.clip(n @ key, 0, None)
    rgb = np.clip(col * shade[:, None], 0, 1) ** (1 / 2.2)

    # 2x2 splat for hole-free coverage
    xs = np.concatenate([x, x + 1, x, x + 1])
    ys = np.concatenate([y, y, y + 1, y + 1])
    zs = np.tile(z, 4)
    idx = np.tile(np.arange(len(x)), 4)
    ok = (xs >= 0) & (xs < res) & (ys >= 0) & (ys < res)
    lin = ys[ok] * res + xs[ok]
    zs, idx = zs[ok], idx[ok]
    order = np.argsort(zs)                    # far to near: nearer points are written last and win
    img = np.empty((res * res, 3), np.float32)
    img[:] = np.asarray(BACKGROUND, np.float32) / 255
    img[lin[order]] = rgb[idx[order]]
    img = img.reshape(res, res, 3)
    hit = np.zeros(res * res, bool)
    hit[lin] = True
    img = _fill_holes(img, hit.reshape(res, res))
    img = img.reshape(size, ss, size, ss, 3).mean((1, 3))
    return Image.fromarray((img * 255 + 0.5).astype(np.uint8))


def _fill_holes(img, hit, passes=2):
    """Fill empty pixels that are mostly surrounded by surface (sampling gaps)."""
    for _ in range(passes):
        pad_i = np.pad(img, ((1, 1), (1, 1), (0, 0)))
        pad_h = np.pad(hit, 1).astype(np.float32)
        acc = np.zeros_like(img)
        cnt = np.zeros(hit.shape, np.float32)
        h, w = hit.shape
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                if dy == 1 and dx == 1:
                    continue
                m = pad_h[dy:dy + h, dx:dx + w]
                acc += pad_i[dy:dy + h, dx:dx + w] * m[..., None]
                cnt += m
        fill = (~hit) & (cnt >= 5)
        img[fill] = acc[fill] / cnt[fill][:, None]
        hit = hit | fill
    return img


def render_views(path, azimuths=(0, 90, 180, 270), size=640):
    names = {0: "front", 180: "back"}
    return [(render(path, az, size=size), names.get(az % 360, f"{az}°")) for az in azimuths]


def render_png(path, azimuth=0.0, elevation=15.0, size=768):
    buf = io.BytesIO()
    render(path, azimuth, elevation, size).save(buf, "PNG", optimize=True)
    return buf.getvalue()
