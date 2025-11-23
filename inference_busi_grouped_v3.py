import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import time
import random
import numpy as np
import torch
import torch.nn as nn
import gc
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
class CFG:
    ROOT = r"C:\alioth_project\alioth_project\Dataset_BUSI_with_GT"
    OUTDIR = r"C:\alioth_project\alioth_project\runs_busi_stable"
    MVS_WEIGHT = r"C:\anaconda\envs\multiverseg\Lib\site-packages\checkpoints\MultiverSeg_v1_nf256_res128.pt"


    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 256

    BATCH = 2
    AMP = False

    EPOCHS = 30
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 2

    SUPPORT_PER_CLASS = 24
    CLASSES = ['benign', 'malignant', 'normal']
    PROMPT_DROPOUT_RATE = 0.20

    HFLIP_P = 0.5
    VFLIP_P = 0.2
    ROT90_P = 0.2
    BRIGHT_JITTER = 0.08
    CONTRAST_JITTER = 0.10

    SCHED_TMAX = EPOCHS
    MIN_LR = 5e-6
    SAVE_EVERY = 5

os.makedirs(CFG.OUTDIR, exist_ok=True)

# ============================================================
# Preprocessing
# ============================================================

def gamma_adjust(img, gamma=0.9): return np.clip(img ** gamma, 0, 1)
def log_compress(img, scale=5.0): return np.log1p(scale * img) / np.log1p(scale)
def clahe_enhance(img, clip=2.0, tile=(8,8)):
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    x = (img * 255).astype(np.uint8)
    x = cl.apply(x)
    return x.astype(np.float32) / 255.0
def despeckle_median(img, k=3):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    filt = pil.filter(ImageFilter.MedianFilter(size=k))
    return np.array(filt, dtype=np.float32) / 255.0
def jitter(img):
    b = 1.0 + random.uniform(-CFG.BRIGHT_JITTER, CFG.BRIGHT_JITTER)
    c = 1.0 + random.uniform(-CFG.CONTRAST_JITTER, CFG.CONTRAST_JITTER)
    mu = img.mean()
    out = (img - mu) * c + mu
    out = np.clip(out * b, 0, 1)
    return out
def geo_augs(img, msk):
    if random.random() < CFG.HFLIP_P:
        img = np.ascontiguousarray(np.flip(img, axis=1)); msk = np.ascontiguousarray(np.flip(msk, axis=1))
    if random.random() < CFG.VFLIP_P:
        img = np.ascontiguousarray(np.flip(img, axis=0)); msk = np.ascontiguousarray(np.flip(msk, axis=0))
    if random.random() < CFG.ROT90_P:
        k = random.choice([1,2,3])
        img = np.ascontiguousarray(np.rot90(img, k)); msk = np.ascontiguousarray(np.rot90(msk, k))
    return img, msk
def ultrasound_preprocess(img, train=False):
    img = np.clip(img, 0, 1)
    img = gamma_adjust(img, 0.9)
    img = log_compress(img, 5.0)
    img = clahe_enhance(img, 2.0, (8,8))
    img = despeckle_median(img, 3)
    if train: img = jitter(img)
    return np.clip(img, 0, 1)

# ============================================================
# Dataset
# ============================================================

class BUSIDataset(Dataset):
    def __init__(self, samples, indices, train=True):
        """
        samples: list of (img_path, msk_path, label_name)
        """
        self.samples = samples
        self.indices = indices
        self.train = train

    def __len__(self): 
        return len(self.indices)

    def __getitem__(self, i):
        img_path, msk_path, cls_name = self.samples[self.indices[i]]
        label_idx = CFG.CLASSES.index(cls_name)

        img = Image.open(img_path).convert('L').resize((CFG.IMG_SIZE, CFG.IMG_SIZE))
        msk = Image.open(msk_path).convert('L').resize((CFG.IMG_SIZE, CFG.IMG_SIZE))

        img = np.array(img, np.float32) / 255.0
        msk = (np.array(msk, np.float32) > 0.5).astype(np.float32)

        img = ultrasound_preprocess(img, train=self.train)
        if self.train:
            img, msk = geo_augs(img, msk)

        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(msk).unsqueeze(0),
            torch.tensor(label_idx, dtype=torch.long)
        )

# ============================================================
# Build Samples & Split
# ============================================================

def build_samples(root):
    """掃瞄全部 BUSI 檔案，建立統一的 samples list"""
    samples = []
    root = Path(root)
    for cls in CFG.CLASSES:
        img_dir = root/cls/'images'
        msk_dir = root/cls/'mask'
        if not img_dir.exists(): continue

        # sorted 保證 Win/Linux/WSL 一致
        for f in sorted(os.listdir(img_dir)):
            if f.endswith('.png'):
                m = f.replace('.png', '_mask.png')
                if (msk_dir/m).exists():
                    samples.append((str(img_dir/f), str(msk_dir/m), cls))
    return samples


def split_samples(samples):
    labels = [s[2] for s in samples]  # class names
    idxs = list(range(len(samples)))

    tr, te = train_test_split(idxs, test_size=0.3, stratify=labels, random_state=42)
    tr_labels = [labels[i] for i in tr]
    tr_idx, val_idx = train_test_split(tr, test_size=0.1, stratify=tr_labels, random_state=42)
    return tr_idx, val_idx, te, labels


# ============================================================
# Model
# ============================================================

from multiverseg.models.sp_mvs import MultiverSegNet

class WrappedMVS(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = MultiverSegNet(
            in_channels=[5, 2],
            encoder_blocks=[256,256,256,256],
            block_kws=dict(conv_kws=dict(norm='layer')),
            cross_relu=True
        ).to(device)

    def forward(self, q5, s_img, s_msk):
        B = q5.shape[0]
        q5 = q5.unsqueeze(1)

        if s_img.ndim == 4: s_img = s_img.unsqueeze(1)
        if s_msk.ndim == 4: s_msk = s_msk.unsqueeze(1)

        if s_img.shape[0] == 1 and B > 1:
            s_img = s_img.expand(B, -1, -1, -1, -1)
            s_msk = s_msk.expand(B, -1, -1, -1, -1)

        return self.net(q5, s_img, s_msk)

    @torch.no_grad()
    def predict(self, q1, proto_img, proto_msk):
        B,_,H,W = q1.shape
        zeros = torch.zeros((B, 4, H, W), device=q1.device)
        q5 = torch.cat([q1, zeros], dim=1)
        return torch.sigmoid(self.forward(q5, proto_img, proto_msk))

# ============================================================
# Interactive Prompt Channels
# ============================================================

def simulate_interactive_channels(msks, device):
    B, _, H, W = msks.shape
    pos = torch.zeros((B,1,H,W), device=device)
    neg = torch.zeros((B,1,H,W), device=device)
    box = torch.zeros((B,1,H,W), device=device)
    prev = torch.zeros((B,1,H,W), device=device)

    for b in range(B):
        if random.random() < CFG.PROMPT_DROPOUT_RATE: 
            continue

        mask_np = msks[b,0].cpu().numpy()
        ys, xs = np.where(mask_np > 0.5)
        yb, xb = np.where(mask_np < 0.5)

        if len(xs) > 0:
            for _ in range(random.randint(1,3)):
                i = random.randint(0, len(xs)-1)
                py, px = ys[i], xs[i]
                pos[b,0,max(py-1,0):min(py+2,H), max(px-1,0):min(px+2,W)] = 1

        if len(xb) > 0:
            for _ in range(random.randint(0,2)):
                i = random.randint(0, len(xb)-1)
                py, px = yb[i], xb[i]
                neg[b,0,max(py-1,0):min(py+2,H), max(px-1,0):min(px+2,W)] = 1

        if len(xs) > 0:
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            box[b,0,max(0,ymin-3):min(H,ymax+3), max(0,xmin-3):min(W,xmax+3)] = 1

    return pos, neg, box, prev


# ============================================================
# Prototypes (Correct Version)
# ============================================================

@torch.no_grad()
def build_class_prototypes(loader, per_class, device):
    """依照 label 收集 per_class 影像與 mask"""
    buckets_img = {c: [] for c in CFG.CLASSES}
    buckets_msk = {c: [] for c in CFG.CLASSES}

    for imgs, msks, lbls in loader:
        imgs = imgs.to(device)
        msks = msks.to(device)

        for img, msk, lbl in zip(imgs, msks, lbls):
            cls = CFG.CLASSES[int(lbl)]
            if len(buckets_img[cls]) < per_class:
                buckets_img[cls].append(img.unsqueeze(0))
                buckets_msk[cls].append(msk.unsqueeze(0))

        if all(len(buckets_img[c]) >= per_class for c in CFG.CLASSES):
            break

    # average per-class prototypes
    protos = {}
    for c in CFG.CLASSES:
        if len(buckets_img[c]) == 0:
            pi = torch.zeros(1,1,CFG.IMG_SIZE,CFG.IMG_SIZE,device=device)
            pm = torch.zeros_like(pi)
        else:
            pi = torch.cat(buckets_img[c], dim=0).mean(dim=0, keepdim=True)
            pm = torch.cat(buckets_msk[c], dim=0).mean(dim=0, keepdim=True)
        protos[c] = (pi, pm)

    return protos


def merge_prototypes(protos, device):
    """Merged proto_img + proto_msk for MultiverSeg single-proto input"""
    proto_img = torch.cat([protos[c][0] for c in CFG.CLASSES], dim=0).mean(dim=0, keepdim=True)
    proto_msk = torch.cat([protos[c][1] for c in CFG.CLASSES], dim=0).mean(dim=0, keepdim=True)
    return proto_img.to(device), proto_msk.to(device)


# ============================================================
# Metrics
# ============================================================

def dice_metric(y_pred, y_true, eps=1e-8):
    y_pred = (y_pred > 0.5).int().flatten()
    y_true = y_true.int().flatten()
    TP = ((y_pred==1)&(y_true==1)).sum().item()
    FP = ((y_pred==1)&(y_true==0)).sum().item()
    FN = ((y_pred==0)&(y_true==1)).sum().item()
    return 2*TP/(2*TP+FP+FN+eps)


# ============================================================
# Training
# ============================================================

def run():
    print(f"[Info] Device = {CFG.DEVICE}")
    device = torch.device(CFG.DEVICE)

    # Build all samples
    samples = build_samples(CFG.ROOT)
    tr_idx, val_idx, te_idx, labels = split_samples(samples)

    tr_ds = BUSIDataset(samples, tr_idx, train=True)
    val_ds = BUSIDataset(samples, val_idx, train=False)

    tr_loader = DataLoader(tr_ds, batch_size=CFG.BATCH, shuffle=True, num_workers=CFG.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    # Model
    model = WrappedMVS(device).to(device)
    if os.path.exists(CFG.MVS_WEIGHT):
        state = torch.load(CFG.MVS_WEIGHT, map_location=device)
        if 'model' in state: model.net.load_state_dict(state['model'])
        elif 'state_dict' in state: model.load_state_dict(state['state_dict'])
        else: model.load_state_dict(state)
        print("[Info] Pretrained weights loaded.")

    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.SCHED_TMAX, eta_min=CFG.MIN_LR)
    bce = nn.BCEWithLogitsLoss()

    # Build prototypes
    proto_loader = DataLoader(tr_ds, batch_size=2, shuffle=True)
    class_protos = build_class_prototypes(proto_loader, CFG.SUPPORT_PER_CLASS, device)
    proto_img, proto_msk = merge_prototypes(class_protos, device)

    best_dice = -1

    # Training loop
    for epoch in range(1, CFG.EPOCHS+1):
        model.train()
        tloss = 0
        pbar = tqdm(tr_loader, desc=f"Epoch {epoch}")

        for imgs, msks, lbls in pbar:
            imgs = imgs.to(device)
            msks = msks.to(device)

            pos, neg, box, prev = simulate_interactive_channels(msks, device)
            q5 = torch.cat([imgs, pos, neg, box, prev], dim=1)

            opt.zero_grad()
            logits = model.forward(q5, proto_img, proto_msk)

            probs = torch.sigmoid(logits)
            dice_loss = 1 - (2*(probs*msks).sum() / ((probs+msks).sum() + 1e-6))
            loss = 0.5*bce(logits, msks) + 0.5*dice_loss

            loss.backward()
            opt.step()

            tloss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            del q5, pos, neg, box, prev, logits, probs, loss

        tloss /= len(tr_loader)

        # ========== Validation ==========
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, msks, lbls in val_loader:
                imgs = imgs.to(device)
                msks = msks.to(device)
                y = model.predict(imgs, proto_img, proto_msk)
                dices.append(dice_metric(y, msks))
                del imgs, msks, y

        mean_dice = np.mean(dices)
        sched.step()

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(), f"{CFG.OUTDIR}/best.pt")

        print(f"[E{epoch}] TrainLoss={tloss:.4f} | ValDice={mean_dice:.4f}")

        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    run()

