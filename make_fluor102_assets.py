import ast, json, pathlib, re
P = pathlib.Path("params"); J = pathlib.Path("jobs")
for name in ("dinov3_base", "celldino", "resnet"):
    d = json.load(open(P / f"params_brightfield_{name}_cluster.json"))
    d["dataset_params"]["dataset"] = "BioAct"
    d["dataset_params"]["data_location"] = "/shared/hdd/data/bioactive/images"
    json.dump(d, open(P / f"params_fluor102_{name}_cluster.json", "w"), indent=2)
    print("wrote params_fluor102_%s_cluster.json" % name)
    print("  leftover brightfield strings:", re.findall(r'"[^"]*brightfield[^"]*"', json.dumps(d)))
n = 0
for f in sorted(J.glob("robin-bioact-brightfield-*.yaml")):
    if "adaptcheck" in f.name or "smoke" in f.name: continue
    t = f.read_text().replace("brightfield", "fluor102")
    t = t.replace("bf_dinov3_base", "f102_dinov3_base").replace("bf_celldino", "f102_celldino")
    (J / f.name.replace("brightfield", "fluor102")).write_text(t); n += 1
print("wrote", n, "job files")
cl = pathlib.Path("classification.py"); t = cl.read_text()
if "f102_dinov3_base" not in t:
    anchor = '    "bf_celldino":    ("celldino",     "bioact_brightfield_celldino"),\n'
    assert t.count(anchor) == 1
    add = ('    "f102_dinov3_base": ("dinov3_base", "bioact_fluor102_dinov3_base"),\n'
           '    "f102_celldino":    ("celldino",    "bioact_fluor102_celldino"),\n')
    t = t.replace(anchor, anchor + add)
    ast.parse(t); cl.write_text(t); print("MODEL_MAP patched")
else:
    print("MODEL_MAP already patched")
