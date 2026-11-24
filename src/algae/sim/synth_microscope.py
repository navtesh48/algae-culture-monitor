"""Synthetic microscope images + YOLO labels, to prove the pipeline end to end.
Swap in real labeled images (same YOLO layout) when you have them."""
import numpy as np, cv2, pathlib

CLASSES = ['algae_cell', 'rotifer', 'ciliate', 'amoeba', 'bacterial_clump', 'debris']

def make_image(rng, size=640):
    img = rng.normal(235, 4, (size, size, 3)).clip(0, 255).astype(np.uint8)
    labels = []
    for _ in range(int(rng.integers(30, 90))):                       # algae cells
        r = int(rng.integers(6, 11)); x, y = (int(v) for v in rng.integers(r, size - r, 2))
        cv2.circle(img, (x, y), r, (60, 150, 60), -1)
        cv2.circle(img, (x, y), max(2, r // 3), (40, 110, 40), -1)
        labels.append((0, x, y, 2 * r, 2 * r))
    if rng.random() < 0.4:                                            # one contaminant
        cls = int(rng.integers(1, 5))
        x, y = (int(v) for v in rng.integers(60, size - 60, 2))
        axes, color = {1: ((34, 18), (200, 190, 170)), 2: ((24, 10), (190, 170, 150)),
                       3: ((20, 20), (180, 180, 200)), 4: ((14, 14), (120, 120, 120))}[cls]
        ang = int(rng.integers(0, 180))
        cv2.ellipse(img, (x, y), axes, ang, 0, 360, color, -1)
        cv2.ellipse(img, (x, y), axes, ang, 0, 360, (90, 90, 90), 1)
        labels.append((cls, x, y, 2 * max(axes), 2 * max(axes)))
    return cv2.GaussianBlur(img, (3, 3), 0), labels

def build(root='data/yolo', n_train=400, n_val=100, seed=0):
    rng = np.random.default_rng(seed)
    for split, n in (('train', n_train), ('val', n_val)):
        (pathlib.Path(root) / 'images' / split).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(root) / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img, labels = make_image(rng)
            cv2.imwrite(f'{root}/images/{split}/{i:04d}.jpg', img)
            lines = [f'{c} {x/640:.5f} {y/640:.5f} {w/640:.5f} {h/640:.5f}' for c, x, y, w, h in labels]
            pathlib.Path(f'{root}/labels/{split}/{i:04d}.txt').write_text('\n'.join(lines))
    pathlib.Path(root, 'data.yaml').write_text(
        'path: ' + root + '\ntrain: images/train\nval: images/val\nnames:\n'
        + ''.join(f'  {i}: {n}\n' for i, n in enumerate(CLASSES)))

if __name__ == '__main__':
    build()
