"""
Generates the 6 per-fold K8s job YAMLs for the ResNet-50 brightfield
fine-tune arm (r448), mirroring jobs/robin-bioact-resnet-r224-f1.yaml's
exact structure/resources/mounts -- only the params file, resolution,
fold, and model_name differ.

    python3 make_resnet_bf_jobs.py
"""
import pathlib

JOBS_DIR = pathlib.Path("jobs")

TEMPLATE = """apiVersion: batch/v1
kind: Job
metadata:
  name: robin-bioact-brightfield-resnet-r448-f{fold}
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
      containers:
      - name: train
        image: abrainone/ai-linux:cu12.6.3-latest
        workingDir: /shared/ssd/home/b-r-singh1/bioactive
        command: ["bash", "-lc"]
        args:
        - >-
          source /shared/ssd/home/b-r-singh1/venv/bin/activate &&
          python classification.py --params_path params/params_brightfield_resnet_cluster.json
          --model resnet --res 448 --fold {fold}
          --batch_size 64 --model_name bioact_brightfield_resnet_r448 &&
          python make_plots_v3.py bioact_brightfield_resnet_r448_fold{fold}
        env:
        - {{ name: WANDB_MODE,      value: "disabled" }}
        - {{ name: BIOACT_SAVE_DIR, value: "/shared/ssd/logs/b-r-singh1" }}
        - {{ name: HF_HOME,         value: "/shared/ssd/home/b-r-singh1/hf_cache" }}
        - {{ name: PYTHONUNBUFFERED, value: "1" }}
        resources:
          requests: {{ nvidia.com/gpu: 1, cpu: 8, memory: 32Gi }}
          limits:   {{ nvidia.com/gpu: 1, cpu: 16, memory: 40Gi }}
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
        emptyDir: {{ medium: Memory, sizeLimit: 16Gi }}
"""


def main() -> None:
    for fold in range(6):
        out_path = JOBS_DIR / f"robin-bioact-brightfield-resnet-r448-f{fold}.yaml"
        out_path.write_text(TEMPLATE.format(fold=fold))
        print(f"[ok] wrote {out_path}")


if __name__ == "__main__":
    main()
