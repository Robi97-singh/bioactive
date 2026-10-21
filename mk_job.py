#!/usr/bin/env python3
"""mk_job.py NAME "BASH COMMAND" [KEY=VAL ...]  -> writes jobs/NAME.yaml (does NOT apply it).
Same pod spec as your other jobs (GPU, venv, mounts)."""
import sys

name, cmd = sys.argv[1], sys.argv[2]
extra = [kv.split("=", 1) for kv in sys.argv[3:]]
env = [("WANDB_MODE", "disabled"), ("BIOACT_SAVE_DIR", "/shared/ssd/logs/b-r-singh1"),
       ("HF_HOME", "/shared/ssd/home/b-r-singh1/hf_cache"), ("PYTHONUNBUFFERED", "1")] + extra
env_txt = "\n".join('        - { name: %s, value: "%s" }' % (k, v) for k, v in env)
yaml = f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {name}
  namespace: b-r-singh1
spec:
  backoffLimit: 0
  template:
    spec:
      restartPolicy: Never
      runtimeClassName: nvidia
      securityContext:
        runAsUser: 1376
        runAsGroup: 1376
        fsGroup: 1376
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - {{ key: kubernetes.io/hostname, operator: NotIn, values: ["w4-mi"] }}
      containers:
      - name: train
        image: abrainone/ai-linux:cu12.6.3-latest
        workingDir: /shared/ssd/home/b-r-singh1/bioactive
        command: ["bash", "-lc"]
        args:
        - |
          source /shared/ssd/home/b-r-singh1/venv/bin/activate
          {cmd}
        env:
{env_txt}
        resources:
          requests: {{ nvidia.com/gpu: 1, cpu: 4, memory: 24Gi }}
          limits:   {{ nvidia.com/gpu: 1, cpu: 8, memory: 32Gi }}
        volumeMounts:
        - {{ mountPath: /shared/ssd/home/b-r-singh1, name: home }}
        - {{ mountPath: /shared/hdd/data/bioactive, name: data, readOnly: true }}
        - {{ mountPath: /shared/ssd/logs/b-r-singh1, name: logs }}
        - {{ mountPath: /dev/shm, name: dshm }}
      volumes:
      - name: home
        hostPath: {{ path: /shared/ssd/home/b-r-singh1, type: Directory }}
      - name: data
        hostPath: {{ path: /shared/hdd/data/bioactive, type: Directory }}
      - name: logs
        hostPath: {{ path: /shared/ssd/logs/b-r-singh1, type: Directory }}
      - name: dshm
        emptyDir: {{ medium: Memory, sizeLimit: 8Gi }}
"""
open(f"jobs/{name}.yaml", "w").write(yaml)
print(f"wrote jobs/{name}.yaml")
