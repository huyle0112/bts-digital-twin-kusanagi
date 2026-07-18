"""Compare test CSV poses vs only train poses that have real image files."""
import csv
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "data" / "public_set"

spec = importlib.util.spec_from_file_location(
    "colmap_loader", ROOT / "scene" / "colmap_loader.py"
)
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)


def center_fwd(q, t):
    R = cl.qvec2rotmat(q)
    C = -R.T @ t
    f = R.T @ np.array([0.0, 0.0, 1.0])
    return C, f


def main():
    print("=== Test poses vs TRAIN IMAGE poses only (not full COLMAP) ===\n")
    rows = []
    for scene in sorted(p for p in PUBLIC.iterdir() if p.is_dir()):
        images = cl.read_extrinsics_binary(
            str(scene / "train" / "sparse" / "0" / "images.bin")
        )
        train_files = {p.name for p in (scene / "train" / "images").iterdir()}
        tr_c, tr_f = [], []
        for im in images.values():
            if im.name not in train_files:
                continue
            C, f = center_fwd(im.qvec, im.tvec)
            tr_c.append(C)
            tr_f.append(f)
        tr_c = np.stack(tr_c)
        tr_f = np.stack(tr_f)

        te_c, te_f, names = [], [], []
        with open(scene / "test" / "test_poses.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                q = np.array([float(row[k]) for k in ("qw", "qx", "qy", "qz")])
                t = np.array([float(row[k]) for k in ("tx", "ty", "tz")])
                C, fw = center_fwd(q, t)
                te_c.append(C)
                te_f.append(fw)
                names.append(row["image_name"].strip())
        te_c = np.stack(te_c)
        te_f = np.stack(te_f)

        diag = float(np.linalg.norm(tr_c.max(0) - tr_c.min(0)))
        d2 = ((te_c[:, None, :] - tr_c[None, :, :]) ** 2).sum(2)
        nn = np.sqrt(d2.min(1))
        idx = d2.argmin(1)

        d2t = ((tr_c[:, None, :] - tr_c[None, :, :]) ** 2).sum(2)
        np.fill_diagonal(d2t, np.inf)
        spacing = float(np.median(np.sqrt(d2t.min(1))))

        te_fn = te_f / (np.linalg.norm(te_f, axis=1, keepdims=True) + 1e-12)
        tr_fn = tr_f / (np.linalg.norm(tr_f, axis=1, keepdims=True) + 1e-12)
        ang = np.degrees(np.arccos(np.clip((te_fn * tr_fn[idx]).sum(1), -1, 1)))
        best = np.degrees(np.arccos(np.clip((te_fn @ tr_fn.T).max(1), -1, 1)))

        lo, hi = tr_c.min(0), tr_c.max(0)
        out = np.any((te_c < lo) | (te_c > hi), 1)
        shift = float(np.linalg.norm(te_c.mean(0) - tr_c.mean(0)))

        n_black = sum(1 for im in images.values() if im.name not in train_files)

        print(f"SCENE {scene.name}")
        print(
            f"  train_imgs={len(tr_c)}  test={len(te_c)}  "
            f"colmap_total={len(images)}  colmap_without_train_file={n_black}"
        )
        print(f"  train AABB diagonal={diag:.3f}  train median spacing={spacing:.4f}")
        print(
            f"  test NN to nearest train-img pose: "
            f"min={nn.min():.4f} med={np.median(nn):.4f} "
            f"p90={np.percentile(nn, 90):.4f} max={nn.max():.4f}"
        )
        print(
            f"  NN as % of scene diag: "
            f"med={100 * np.median(nn) / diag:.2f}%  "
            f"p90={100 * np.percentile(nn, 90) / diag:.2f}%  "
            f"max={100 * nn.max() / diag:.2f}%"
        )
        print(
            f"  frac NN > spacing / >2x spacing: "
            f"{(nn > spacing).mean() * 100:.1f}% / "
            f"{(nn > 2 * spacing).mean() * 100:.1f}%"
        )
        print(
            f"  outside train AABB: {out.mean() * 100:.1f}%  "
            f"mean center shift: {shift:.4f}"
        )
        print(
            f"  angle to pos-NN (deg) med/p90/max: "
            f"{np.median(ang):.1f}/{np.percentile(ang, 90):.1f}/{ang.max():.1f}"
        )
        print(
            f"  best angle any train (deg) med/p90/max: "
            f"{np.median(best):.1f}/{np.percentile(best, 90):.1f}/{best.max():.1f}"
        )
        w = np.argsort(-nn)[:5]
        print("  farthest test poses from train images:")
        for i in w:
            print(
                f"    {names[i]}  nn={nn[i]:.4f} "
                f"({100 * nn[i] / diag:.2f}% diag)  "
                f"ang_nn={ang[i]:.1f} best_ang={best[i]:.1f} out={bool(out[i])}"
            )
        print()
        rows.append(
            (
                scene.name,
                len(tr_c),
                len(te_c),
                n_black,
                float(np.median(nn)),
                float(np.percentile(nn, 90)),
                float(nn.max()),
                100 * float(np.median(nn)) / diag,
                100 * float(nn.max()) / diag,
                float(out.mean() * 100),
                float(np.median(best)),
                float(np.percentile(best, 90)),
            )
        )

    print("=" * 90)
    print(
        f"{'scene':10s} {'ntr':>4s} {'nte':>4s} {'nblk':>5s} "
        f"{'nn_med':>7s} {'nn_p90':>7s} {'nn_max':>7s} "
        f"{'med%diag':>8s} {'max%diag':>8s} {'out%':>6s} "
        f"{'bang_med':>8s} {'bang_p90':>8s}"
    )
    for r in rows:
        print(
            f"{r[0]:10s} {r[1]:4d} {r[2]:4d} {r[3]:5d} "
            f"{r[4]:7.4f} {r[5]:7.4f} {r[6]:7.4f} "
            f"{r[7]:7.2f}% {r[8]:7.2f}% {r[9]:5.1f}% "
            f"{r[10]:8.1f} {r[11]:8.1f}"
        )


if __name__ == "__main__":
    main()
