#!/usr/bin/env python3
"""Generate Kubernetes Job YAMLs for the 6-fold CV campaign.

Each job: train (auto-tests, since is_supervised) -> make_plots_v3.py

Matrix:  224 -> resnet, dino, clip, biomedclip, lora_vit_s   (30 jobs)
         448 -> resnet, dino, lora_vit_s                     (18 jobs)
CLIP/BiomedCLIP are locked to 224 by their pretrained positional embeddings.

Models with extra model_params (LoRA rank/alpha/target-modules, Cell-DINO's
checkpoint path, etc.) that the shared BASE file can't express via --model
alone get their own dedicated base file in MODEL_BASE.

  python3 make_jobs.py --models resnet --res 448 --folds 0   # smoke test
  python3 make_jobs.py                                       # all 48
"""
import argparse, os

NS       = "b-r-singh1"
CODE     = "/shared/ssd/home/b-r-singh1/bioactive"
HOME     = "/shared/ssd/home/b-r-singh1"
DATA     = "/shared/hdd/data/bioactive"
LOGS     = "/shared/ssd/logs/b-r-singh1"
VENV     = "/shared/ssd/home/b-r-singh1/venv"
IMAGE    = "abrainone/ai-linux:cu12.6.3-latest"
BASE     = "params/params_cluster_base.json"
MODEL_BASE = {"lora_vit_s": "params/params_lora_vit_small_cluster.json"}
UID = GID = 1376

# 4090s have 24GB; these came from a 16GB 4080, so they are conservative.
BATCH = {"resnet": 64, "dino": 32, "clip": 16, "biomedclip": 16, "lora_vit_s": 32}
MATRIX = {224: ["resnet", "dino", "clip", "biomedclip", "lora_vit_s"],
          448: ["resnet", "dino", "lora_vit_s"]}
FOLDS = [0, 1, 2, 3, 4, 5]

TPL = """apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: {ns}
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      securityContext:
        runAsUser: {uid}
        runAsGroup: {gid}
        fsGroup: {gid}
      containers:
      - name: train
        image: {image}
        workingDir: {code}
        command: ["bash", "-lc"]
        args:
        - >-
          source {venv}/bin/activate &&
          python classification.py --params_path {base}
          --model {model} --res {res} --fold {fold}
          --batch_size {batch} --model_name {mname} &&
          python make_plots_v3.py {mname}_fold{fold}
        env:
        - {{ name: WANDB_MODE,      value: "disabled" }}
        - {{ name: BIOACT_SAVE_DIR, value: "{logs}" }}
        - {{ name: HF_HOME,         value: "{home}/.cache/huggingface" }}
        - {{ name: PYTHONUNBUFFERED, value: "1" }}
        resources:
          requests: {{ nvidia.com/gpu: 1, cpu: 8, memory: 32Gi }}
          limits:   {{ nvidia.com/gpu: 1, cpu: 16, memory: 40Gi }}
        volumeMounts:
        - {{ mountPath: {home}, name: home }}
        - {{ mountPath: {data}, name: data, readOnly: true }}
        - {{ mountPath: {logs}, name: logs }}
        - {{ mountPath: /dev/shm, name: dshm }}
      volumes:
      - name: home
        hostPath: {{ path: {home}, type: Directory }}
      - name: data
        hostPath: {{ path: {data}, type: Directory }}
      - name: logs
        hostPath: {{ path: {logs}, type: Directory }}
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: 16Gi }}
"""

def jobname(m, r, f):
    return f"robin-bioact-{m.replace('_','-')}-r{r}-f{f}"

def render(m, r, f):
    return TPL.format(name=jobname(m, r, f), ns=NS, uid=UID, gid=GID, image=IMAGE,
                      code=CODE, home=HOME, data=DATA, logs=LOGS, venv=VENV,
                      base=MODEL_BASE.get(m, BASE), model=m, res=r, fold=f,
                      batch=BATCH.get(m, 32), mname=f"bioact_{m}_r{r}")

ap = argparse.ArgumentParser()
ap.add_argument("--outdir", default="jobs")
ap.add_argument("--models", nargs="*")
ap.add_argument("--res", nargs="*", type=int)
ap.add_argument("--folds", nargs="*", type=int)
ap.add_argument("--dry", action="store_true")
a = ap.parse_args()

resl = a.res or sorted(MATRIX)
fol  = a.folds or FOLDS
if not a.dry: os.makedirs(a.outdir, exist_ok=True)

n = 0
for r in resl:
    if r not in MATRIX:
        print(f"  skip res={r}: no models defined"); continue
    for m in (x for x in MATRIX[r] if not a.models or x in a.models):
        for f in fol:
            nm = jobname(m, r, f)
            if a.dry:
                print(f"  would write {nm}.yaml")
            else:
                open(os.path.join(a.outdir, f"{nm}.yaml"), "w").write(render(m, r, f))
            n += 1

if a.dry: raise SystemExit
print(f"Wrote {n} job file(s) to ./{a.outdir}/")
print(f"\nSmoke test:\n  kubectl apply -f {a.outdir}/{jobname('resnet',448,0)}.yaml")
print("  kubectl get pods\n  kubectl logs -f <pod>")
