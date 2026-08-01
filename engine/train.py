"""
RoboPacer training module — called by server.py or standalone.
"""
import json, os, random, subprocess, sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CALIB_N       = 200

ENGINE_DIR = Path(__file__).parent
ROOT_DIR   = ENGINE_DIR.parent


# ── Dataset ───────────────────────────────────────────────────────────────────

class SteeringDataset(Dataset):
    def __init__(self, records, root, augment=False):
        self.records = records
        self.root    = Path(root)
        self.augment = augment
        self.resize  = T.Resize((IMG_SIZE, IMG_SIZE), antialias=True)
        self.tt      = T.ToTensor()
        self.norm    = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        self.jitter  = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec   = self.records[idx]
        img   = Image.open(self.root / rec["image_path"]).convert("RGB")
        angle = float(rec["steering_angle"])
        img   = self.resize(img)
        if self.augment:
            if random.random() < 0.5:
                img   = T.functional.hflip(img)
                angle = -angle
            img = self.jitter(img)
        img = self.norm(self.tt(img))
        return img, torch.tensor(angle, dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Sequential(
        nn.Linear(512, 64),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )
    return m


# ── Train / eval loops ────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, device, push=None):
    model.train()
    total    = 0.0
    samples  = 0
    n        = len(loader)
    log_step = max(1, n // 10)
    for i, (imgs, angles) in enumerate(loader):
        imgs, angles = imgs.to(device), angles.to(device)
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(imgs).squeeze(1), angles)
        loss.backward()
        optimizer.step()
        batch_n  = len(imgs)
        total   += loss.item() * batch_n
        samples += batch_n
        if push and (i + 1) % log_step == 0:
            push({"type": "log",
                  "text": f"  train batch {i+1}/{n}  loss={total/samples:.5f}",
                  "level": "info"})
    return total / len(loader.dataset)


@torch.no_grad()
def _eval_epoch(model, loader, device):
    model.eval()
    total = 0.0
    for imgs, angles in loader:
        imgs, angles = imgs.to(device), angles.to(device)
        total += nn.functional.mse_loss(model(imgs).squeeze(1), angles).item() * len(imgs)
    return total / len(loader.dataset)


# ── ONNX export ───────────────────────────────────────────────────────────────

def _export_onnx(model, device, path):
    model.eval()
    dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
    torch.onnx.export(
        model, dummy, str(path),
        export_params=True, opset_version=13, do_constant_folding=True,
        input_names=["input"], output_names=["steering"],
        dynamic_axes={"input": {0: "batch"}, "steering": {0: "batch"}},
        dynamo=False,
    )


# ── Calibration data ──────────────────────────────────────────────────────────

def _save_calib(records, data_root, out_path):
    resize = T.Resize((IMG_SIZE, IMG_SIZE), antialias=True)
    tt     = T.ToTensor()
    norm   = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    sample = random.sample(records, min(CALIB_N, len(records)))
    arrays = [norm(tt(resize(Image.open(Path(data_root) / r["image_path"]).convert("RGB")))).numpy()
              for r in sample]
    np.save(str(out_path), np.stack(arrays))


# ── Docker HEF compilation ────────────────────────────────────────────────────

def _compile_hef(model_name: str, push):
    compile_sh = ENGINE_DIR / "compile" / "compile.sh"

    push({"type": "log", "text": "Checking Docker...", "level": "info"})
    r = subprocess.run(["docker", "info"], capture_output=True)
    if r.returncode != 0:
        push({"type": "log", "text": "Docker is not running — start Docker Desktop first.", "level": "error"})
        return False

    push({"type": "log", "text": "Building hailo-dfc image (uses cache if already built)...", "level": "info"})
    build_proc = subprocess.Popen(
        ["docker", "build", "--progress=plain", "-t", "hailo-dfc",
         "-f", str(ENGINE_DIR / "compile" / "Dockerfile"),
         str(ENGINE_DIR)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace'
    )
    for line in build_proc.stdout:
        line = line.rstrip()
        if line:
            push({"type": "log", "text": line, "level": "docker"})
    build_proc.wait()
    if build_proc.returncode != 0:
        push({"type": "log", "text": "Docker build failed.", "level": "error"})
        return False

    push({"type": "log", "text": f"Compiling {model_name} to HEF inside Docker...", "level": "info"})
    proc = subprocess.Popen(
        ["docker", "run", "--rm",
         "-v", f"{ROOT_DIR}:/workspace",
         "hailo-dfc", "bash",
         f"/workspace/engine/compile/compile.sh", model_name],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace'
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            push({"type": "log", "text": line, "level": "docker"})
    proc.wait()

    if proc.returncode != 0:
        push({"type": "log", "text": "DFC compilation failed.", "level": "error"})
        return False

    push({"type": "log", "text": f"HEF ready: models/{model_name}.hef", "level": "success"})
    return True


# ── Shared export pipeline ────────────────────────────────────────────────────

def _do_exports(model, model_name, formats, push, models_dir, device,
                records=None, data_root=None):
    """Export a loaded model to the requested formats."""
    need_onnx  = "onnx" in formats
    need_hailo = "hef" in formats or "har" in formats
    onnx_path  = models_dir / f"{model_name}.onnx"

    if need_onnx or need_hailo:
        push({"type": "log", "text": "Exporting ONNX...", "level": "info"})
        _export_onnx(model, device, onnx_path)
        push({"type": "log", "text": f"Saved: models/{model_name}.onnx", "level": "success"})
    if need_onnx:
        push({"type": "file", "name": f"{model_name}.onnx"})

    if need_hailo:
        calib_path = models_dir / "calib_data.npy"
        if not calib_path.exists():
            if records and data_root:
                push({"type": "log", "text": "Saving calibration data...", "level": "info"})
                _save_calib(records, data_root, calib_path)
            else:
                push({"type": "log",
                      "text": "calib_data.npy not found — provide a driving log JSON to generate it.",
                      "level": "error"})
                return

        ok = _compile_hef(model_name, push)
        if ok:
            if "hef" in formats:
                push({"type": "file", "name": f"{model_name}.hef"})
            if "har" in formats:
                compiled = models_dir / f"{model_name}_compiled.har"
                if compiled.exists():
                    push({"type": "file", "name": f"{model_name}_compiled.har"})

        if not need_onnx and onnx_path.exists():
            onnx_path.unlink()
        if "har" not in formats:
            for tmp in [models_dir / f"{model_name}.har",
                        models_dir / f"{model_name}_optimized.har"]:
                if tmp.exists():
                    tmp.unlink()

    push({"type": "log", "text": "All exports complete.", "level": "success"})


# ── Main entry point ──────────────────────────────────────────────────────────

def run(config: dict, push=None, should_stop=None):
    """
    config keys:
      frames_dir  : absolute path to frames folder
      json_path   : absolute path to driving_log.json
      model_name  : output model name (e.g. "my_car")
      epochs      : int  (default 20)
      batch_size  : int  (default 32)
      formats     : list of "pth" | "onnx" | "har" | "hef"
    push(event_dict) streams progress to the web UI.
    should_stop() returns True when early stop has been requested.
    """
    if push is None:
        push = lambda e: print(e.get("text", e))

    frames_dir = Path(config["frames_dir"])
    json_path  = Path(config["json_path"])
    model_name = config.get("model_name", "model").strip() or "model"
    epochs     = int(config.get("epochs", 20))
    batch_size = int(config.get("batch_size", 32))
    formats    = set(config.get("formats", ["onnx"]))
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    push({"type": "log", "text": f"Device: {device}", "level": "info"})
    push({"type": "log", "text": f"Model name: {model_name}", "level": "info"})
    push({"type": "log", "text": f"Formats: {sorted(formats)}", "level": "info"})

    with open(json_path) as f:
        records = json.load(f)

    data_root = json_path.parent
    records = [r for r in records if (data_root / r["image_path"]).exists()]
    push({"type": "log", "text": f"Valid samples: {len(records)}", "level": "info"})

    random.shuffle(records)
    n       = len(records)
    n_train = int(0.80 * n)
    n_val   = int(0.10 * n)
    train_r = records[:n_train]
    val_r   = records[n_train:n_train + n_val]
    push({"type": "log", "text": f"Split  train={len(train_r)}  val={len(val_r)}", "level": "info"})
    push({"type": "split", "train": len(train_r), "val": len(val_r), "total": n})

    train_ds = SteeringDataset(train_r, data_root, augment=True)
    val_ds   = SteeringDataset(val_r,   data_root, augment=False)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model     = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val  = float("inf")
    ckpt_path = models_dir / f"{model_name}.pth"

    push({"type": "log", "text": f"Training for {epochs} epochs...", "level": "info"})

    for epoch in range(1, epochs + 1):
        push({"type": "log", "text": f"Epoch {epoch}/{epochs} — training...", "level": "info"})
        train_loss = _train_epoch(model, train_ld, optimizer, device, push)
        push({"type": "log", "text": f"Epoch {epoch}/{epochs} — validating...", "level": "info"})
        val_loss   = _eval_epoch(model, val_ld, device)
        scheduler.step()
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
        push({
            "type": "epoch",
            "epoch": epoch, "total": epochs,
            "train": round(train_loss, 6),
            "val":   round(val_loss, 6),
            "best":  is_best,
        })
        if should_stop and should_stop():
            push({"type": "log",
                  "text": f"Stopped at epoch {epoch} — exporting best checkpoint...",
                  "level": "warning"})
            break

    push({"type": "log", "text": f"Best val MSE: {best_val:.6f}  RMSE: {best_val**0.5:.4f}", "level": "success"})

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    push({"type": "log", "text": f"Saved: models/{model_name}.pth", "level": "success"})
    push({"type": "file", "name": f"{model_name}.pth"})

    _do_exports(model, model_name, formats, push, models_dir, device, records, data_root)


# ── Convert existing PTH to other formats ────────────────────────────────────

def convert(config: dict, push=None):
    """
    Convert an existing .pth checkpoint to ONNX / HAR / HEF.
    config keys:
      model_name : name of the model (loads models/{model_name}.pth)
      formats    : list of "onnx" | "har" | "hef"
      json_path  : (optional) driving_log.json path — only needed if
                   models/calib_data.npy does not already exist
    """
    if push is None:
        push = lambda e: print(e.get("text", e))

    model_name = config.get("model_name", "model").strip() or "model"
    formats    = set(config.get("formats", ["onnx"]))
    json_path  = config.get("json_path", "").strip()
    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    push({"type": "log", "text": f"Device: {device}", "level": "info"})

    ckpt_path = models_dir / f"{model_name}.pth"
    if not ckpt_path.exists():
        push({"type": "log", "text": f"Not found: models/{model_name}.pth", "level": "error"})
        return

    push({"type": "log", "text": f"Loading {model_name}.pth...", "level": "info"})
    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    push({"type": "log", "text": "Checkpoint loaded.", "level": "success"})
    push({"type": "file", "name": f"{model_name}.pth"})

    records   = None
    data_root = None
    if json_path:
        try:
            with open(json_path) as f:
                records = json.load(f)
            data_root = Path(json_path).parent
            records = [r for r in records if (data_root / r["image_path"]).exists()]
            push({"type": "log", "text": f"Loaded {len(records)} records for calibration.", "level": "info"})
        except Exception as e:
            push({"type": "log", "text": f"Could not load JSON: {e}", "level": "warning"})

    _do_exports(model, model_name, formats, push, models_dir, device, records, data_root)


# ── Standalone usage ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frames-dir", default=str(ROOT_DIR / "frames"))
    p.add_argument("--json",       default=str(ROOT_DIR / "driving_log.json"))
    p.add_argument("--name",       default="model")
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--formats",    nargs="+", default=["onnx"],
                   choices=["pth", "onnx", "har", "hef"])
    args = p.parse_args()

    run({
        "frames_dir": args.frames_dir,
        "json_path":  args.json,
        "model_name": args.name,
        "epochs":     args.epochs,
        "formats":    args.formats,
    })
