import argparse
import math
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def pixel_unshuffle(x, upscale_factor):
    b, c, h, w = x.size()
    out_h = h // upscale_factor
    out_w = w // upscale_factor
    x = x.contiguous().view(b, c, out_h, upscale_factor, out_w, upscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    return x.view(b, c * upscale_factor * upscale_factor, out_h, out_w)


class FFDNet(nn.Module):
    def __init__(self, in_channels=1, num_features=64, num_conv=15):
        super().__init__()
        self.in_channels = in_channels
        body = []
        body.append(nn.Conv2d(in_channels * 4 + 1, num_features, 3, 1, 1))
        body.append(nn.ReLU(inplace=True))
        for _ in range(num_conv - 2):
            body.append(nn.Conv2d(num_features, num_features, 3, 1, 1, bias=False))
            body.append(nn.BatchNorm2d(num_features))
            body.append(nn.ReLU(inplace=True))
        body.append(nn.Conv2d(num_features, in_channels * 4, 3, 1, 1))
        self.body = nn.Sequential(*body)

    def forward(self, x, sigma):
        # FFDNet works on four downsampled subimages and takes a noise-level map.
        h, w = x.shape[-2:]
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x_unshuffled = pixel_unshuffle(x, 2)
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        sigma_map = sigma.expand(x_unshuffled.size(0), 1, x_unshuffled.size(2), x_unshuffled.size(3))
        out = self.body(torch.cat([x_unshuffled, sigma_map], dim=1))
        out = F.pixel_shuffle(out, 2)
        return out[..., :h, :w]


class CleanPatchDataset(Dataset):
    def __init__(self, root, patch_size=64, channels=1, length=None):
        self.root = Path(root)
        self.patch_size = patch_size
        self.channels = channels
        self.paths = sorted(
            p for p in self.root.rglob("*")
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        if not self.paths:
            raise RuntimeError(f"No images found in {self.root}")
        self.length = length or len(self.paths)

    def __len__(self):
        return self.length

    def _read_image(self, path):
        if self.channels == 1:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            img = img[:, :, None]
        else:
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Failed to read image: {path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _pad_if_needed(self, img):
        h, w = img.shape[:2]
        pad_h = max(0, self.patch_size - h)
        pad_w = max(0, self.patch_size - w)
        if pad_h == 0 and pad_w == 0:
            return img
        return cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

    def __getitem__(self, index):
        path = self.paths[index % len(self.paths)]
        img = self._pad_if_needed(self._read_image(path))
        h, w = img.shape[:2]
        top = random.randint(0, h - self.patch_size)
        left = random.randint(0, w - self.patch_size)
        img = img[top:top + self.patch_size, left:left + self.patch_size, :]

        if random.random() < 0.5:
            img = img[:, ::-1, :]
        if random.random() < 0.5:
            img = img[::-1, :, :]
        k = random.randint(0, 3)
        img = np_rot90(img, k)

        img = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        return img


def np_rot90(img, k):
    return np.ascontiguousarray(np.rot90(img, k))


def save_checkpoint(path, model, optimizer, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal PyTorch FFDNet training for clean-image denoising datasets.")
    parser.add_argument("--train_data", required=True, help="Path to clean training images, e.g. DFWB/train.")
    parser.add_argument("--save_dir", default="models/FFDNet_DFWB_gray", help="Checkpoint directory.")
    parser.add_argument("--channels", type=int, default=1, choices=[1, 3], help="1 for gray, 3 for color.")
    parser.add_argument("--patch_size", type=int, default=64, help="Training patch size. Must be even.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--sigma_min", type=float, default=0, help="Minimum AWGN sigma in [0,255].")
    parser.add_argument("--sigma_max", type=float, default=75, help="Maximum AWGN sigma in [0,255].")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--steps_per_epoch", type=int, default=None, help="Override number of batches per epoch.")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume.")
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.patch_size % 2 != 0:
        raise ValueError("FFDNet patch_size must be even because of PixelUnShuffle.")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset_len = args.steps_per_epoch * args.batch_size if args.steps_per_epoch else None
    dataset = CleanPatchDataset(args.train_data, args.patch_size, args.channels, dataset_len)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    model = FFDNet(in_channels=args.channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    start_epoch = 0
    global_step = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Train images: {len(dataset.paths)}")
    print(f"Patch size: {args.patch_size}, channels: {args.channels}, sigma range: [{args.sigma_min}, {args.sigma_max}]")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for i, clean in enumerate(loader):
            clean = clean.to(device, non_blocking=True)
            sigma = torch.empty(clean.size(0), device=device).uniform_(args.sigma_min, args.sigma_max) / 255.0
            noisy = clean + torch.randn_like(clean) * sigma.view(-1, 1, 1, 1)
            noisy = noisy.clamp(0.0, 1.0)

            restored = model(noisy, sigma)
            loss = criterion(restored, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1
            if i % 20 == 0:
                print(f"epoch {epoch + 1:04d}/{args.epochs:04d} step {i:05d}/{len(loader):05d} loss {loss.item():.6f}")

        avg_loss = epoch_loss / max(1, len(loader))
        print(f"epoch {epoch + 1:04d} average loss {avg_loss:.6f}")
        save_checkpoint(os.path.join(args.save_dir, "latest.pt"), model, optimizer, epoch + 1, global_step)
        if (epoch + 1) % 10 == 0:
            save_checkpoint(os.path.join(args.save_dir, f"epoch_{epoch + 1:04d}.pt"), model, optimizer, epoch + 1, global_step)


if __name__ == "__main__":
    main()
