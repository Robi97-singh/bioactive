import ast, pathlib, random
import numpy as np
from PIL import Image
R = pathlib.Path("/shared/hdd/data/bioactive/brightfield_images")
N, C = 3000, 448
random.seed(0); rng = np.random.default_rng(0)
files = random.sample(list(R.glob("*/*.png")), N)
s = s2 = 0.0; n = 0
for i, p in enumerate(files, 1):
    a = np.asarray(Image.open(p), dtype=np.float64) / 255.0
    t, l = (int(x) for x in rng.integers(0, a.shape[0] - C + 1, size=2))
    c = a[t:t+C, l:l+C]
    s += c.sum(); s2 += (c**2).sum(); n += c.size
    if i % 500 == 0: print(i, flush=True)
m = s / n; sd = (s2 / n - m**2) ** 0.5
print(f"mean={m:.4f} std={sd:.4f}")
f = pathlib.Path("defaults/datasets.py"); x = f.read_text()
a1, a2 = "self.mean = (0.4928,)", "self.std  = (0.2288,)"
assert x.count(a1) == 1 and x.count(a2) == 1
x = x.replace(a1, f"self.mean = ({m:.4f},)").replace(a2, f"self.std  = ({sd:.4f},)")
ast.parse(x); f.write_text(x); print("PATCHED_OK")
