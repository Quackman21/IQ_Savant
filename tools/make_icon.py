"""Turn a square logo into a macOS-style app icon: rounded corners, Apple's margin, soft shadow.

    python tools/make_icon.py "path/to/logo.png" tools/IQ_icon.png
"""
import sys
from PIL import Image, ImageFilter
import numpy as np
src = Image.open(sys.argv[1]).convert("RGBA")
S, B = 1024, 824
off = (S - B) // 2
w, h = src.size; z = 0.07                                              # zoom in 7% so the IQ fills more
src = src.crop((int(w * z), int(h * z), int(w * (1 - z)), int(h * (1 - z))))
body = np.asarray(src.resize((B, B), Image.LANCZOS)).astype(np.float32)

# macOS icon shape: straight sides, corner radius 22.5% of the body, smooth corners; 4x supersampled
k = 4; R = B * k / 2.0; rad = 0.225 * B * k
c = np.abs(np.arange(B * k, dtype=np.float64) + 0.5 - R) - (R - rad)
dx, dy = np.maximum(c, 0)[None, :], np.maximum(c, 0)[:, None]
m = ((dx ** 2.6 + dy ** 2.6) ** (1 / 2.6) <= rad).astype(np.float32)
mask = m.reshape(B, k, B, k).mean(axis=(1, 3))                      # antialiased edge, 0..1

# soft top highlight
hl = np.clip(1 - np.linspace(0, 1, B) / 0.45, 0, 1)[:, None] * 0.10
body[..., :3] = body[..., :3] * (1 - hl[..., None]) + 255 * hl[..., None]
# thin light rim
inner = np.asarray(Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))).astype(np.float32) / 255
rim = np.clip(mask - inner, 0, 1) * 0.18
body[..., :3] = body[..., :3] * (1 - rim[..., None]) + 255 * rim[..., None]
body[..., 3] = mask * 255
body_img = Image.fromarray(body.clip(0, 255).astype(np.uint8), "RGBA")

sh = np.zeros((S, S), np.float32); sh[off + 12: off + 12 + B, off: off + B] = mask * 0.35
sh_img = Image.fromarray((sh * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(18))
canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0)); canvas.putalpha(sh_img)
canvas.alpha_composite(body_img, (off, off))
canvas.save(sys.argv[2])
print("icon saved:", sys.argv[2])
