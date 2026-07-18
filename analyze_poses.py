"""Compare public_set test_poses.csv vs train COLMAP poses (black-region diagnosis)."""
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


def cam_center_from_w2c(qvec, tvec):
    """COLMAP: x_cam = R * x_world + t  =>  C = -R^T t."""
    R = cl.qvec2rotmat(qvec)
    C = -R.T @ tvec
    return C, R


def forward_from_R(R):
    """Camera looks along +Z in COLMAP; world forward = R^T @ [0,0,1]."""
    return R.T @ np.array([0.0, 0.0, 1.0])


def load_train(sparse0: Path):
    images = cl.read_extrinsics_binary(str(sparse0 / "images.bin"))
    cams = cl.read_intrinsics_binary(str(sparse0 / "cameras.bin"))
    centers, forwards, tvecs, names, fovs = [], [], [], [], []
    for im in images.values():
        C, R = cam_center_from_w2c(im.qvec, im.tvec)
        centers.append(C)
        forwards.append(forward_from_R(R))
        tvecs.append(im.tvec)
        names.append(im.name)
        cam = cams[im.camera_id]
        if cam.model in (
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL",
            "RADIAL_FISHEYE",
        ):
            fx = fy = float(cam.params[0])
        else:
            fx, fy = float(cam.params[0]), float(cam.params[1])
        fovs.append((fx, fy, int(cam.width), int(cam.height), cam.model))
    return {
        "centers": np.stack(centers),
        "forwards": np.stack(forwards),
        "tvecs": np.stack(tvecs),
        "names": names,
        "fovs": fovs,
    }


def load_test(csv_path: Path):
    centers, forwards, tvecs, names, fovs = [], [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = np.array(
                [float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"])],
                dtype=np.float64,
            )
            t = np.array(
                [float(row["tx"]), float(row["ty"]), float(row["tz"])],
                dtype=np.float64,
            )
            C, R = cam_center_from_w2c(q, t)
            centers.append(C)
            forwards.append(forward_from_R(R))
            tvecs.append(t)
            names.append(row["image_name"].strip())
            fovs.append(
                (
                    float(row["fx"]),
                    float(row["fy"]),
                    int(float(row["width"])),
                    int(float(row["height"])),
                    "CSV",
                )
            )
    return {
        "centers": np.stack(centers),
        "forwards": np.stack(forwards),
        "tvecs": np.stack(tvecs),
        "names": names,
        "fovs": fovs,
    }


def load_points(sparse0: Path):
    bin_path = sparse0 / "points3D.bin"
    if not bin_path.exists():
        return None
    try:
        xyz, _rgb, _err = cl.read_points3D_binary(str(bin_path))
        return xyz
    except Exception as e:
        print("  [warn] points3D load failed:", e)
        return None


def nn_stats(query, ref):
    d2 = ((query[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    nn = np.sqrt(d2.min(axis=1))
    nn_idx = d2.argmin(axis=1)
    return nn, nn_idx


def angle_deg(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    c = np.clip((a * b).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(c))


def bbox_outside(pts, ref, margin=0.0):
    lo = ref.min(axis=0) - margin
    hi = ref.max(axis=0) + margin
    outside = np.any((pts < lo) | (pts > hi), axis=1)
    return outside


def summarize_scene(scene_dir: Path):
    sparse0 = scene_dir / "train" / "sparse" / "0"
    csv_path = scene_dir / "test" / "test_poses.csv"
    if not sparse0.exists() or not csv_path.exists():
        return None

    train = load_train(sparse0)
    test = load_test(csv_path)
    pts = load_points(sparse0)

    tr_c, te_c = train["centers"], test["centers"]
    tr_f, te_f = train["forwards"], test["forwards"]

    d2 = ((tr_c[:, None, :] - tr_c[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    tr_2nn = np.sqrt(d2.min(axis=1))
    train_med_spacing = float(np.median(tr_2nn))

    te_nn, te_nn_idx = nn_stats(te_c, tr_c)
    ang_to_nn = angle_deg(te_f, tr_f[te_nn_idx])

    te_fn = te_f / (np.linalg.norm(te_f, axis=1, keepdims=True) + 1e-12)
    tr_fn = tr_f / (np.linalg.norm(tr_f, axis=1, keepdims=True) + 1e-12)
    cos = np.clip(te_fn @ tr_fn.T, -1.0, 1.0)
    best_ang = np.degrees(np.arccos(cos.max(axis=1)))

    out_mask = bbox_outside(te_c, tr_c, margin=0.0)
    diag = float(np.linalg.norm(tr_c.max(0) - tr_c.min(0)))
    out10_mask = bbox_outside(te_c, tr_c, margin=0.1 * diag / np.sqrt(3.0))

    center = tr_c.mean(axis=0)
    radius = float(np.linalg.norm(tr_c - center, axis=1).max())
    te_r = np.linalg.norm(te_c - center, axis=1)
    tr_r = np.linalg.norm(tr_c - center, axis=1)

    tr_fx = np.array([f[0] for f in train["fovs"]], dtype=float)
    te_fx = np.array([f[0] for f in test["fovs"]], dtype=float)
    tr_wh = np.array([(f[2], f[3]) for f in train["fovs"]], dtype=float)
    te_wh = np.array([(f[2], f[3]) for f in test["fovs"]], dtype=float)

    ood_pos = te_nn > max(2.0 * train_med_spacing, 1e-6)
    ood_ang = ang_to_nn > 30.0
    ood_any = ood_pos | ood_ang | out_mask

    pcd_stats = {}
    if pts is not None and len(pts) > 0:
        if len(pts) > 200_000:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), 200_000, replace=False)
            pts_s = pts[idx]
        else:
            pts_s = pts
        pcd_c = pts_s.mean(0)
        pcd_lo, pcd_hi = pts_s.min(0), pts_s.max(0)
        pcd_diag = float(np.linalg.norm(pcd_hi - pcd_lo))
        tr_to_pcd = np.linalg.norm(tr_c - pcd_c, axis=1)
        te_to_pcd = np.linalg.norm(te_c - pcd_c, axis=1)

        def look_at_angle(cams, fwds, target):
            v = target[None, :] - cams
            return angle_deg(fwds, v)

        tr_look = look_at_angle(tr_c, tr_f, pcd_c)
        te_look = look_at_angle(te_c, te_f, pcd_c)
        te_out_pcd = float(np.any((te_c < pcd_lo) | (te_c > pcd_hi), axis=1).mean())
        tr_out_pcd = float(np.any((tr_c < pcd_lo) | (tr_c > pcd_hi), axis=1).mean())
        pcd_stats = {
            "n_pts": int(len(pts)),
            "pcd_diag": pcd_diag,
            "train_dist_to_pcd_med": float(np.median(tr_to_pcd)),
            "test_dist_to_pcd_med": float(np.median(te_to_pcd)),
            "train_look_at_pcd_med_deg": float(np.median(tr_look)),
            "test_look_at_pcd_med_deg": float(np.median(te_look)),
            "test_look_at_pcd_p90_deg": float(np.percentile(te_look, 90)),
            "train_look_at_pcd_p90_deg": float(np.percentile(tr_look, 90)),
            "frac_test_cam_outside_pcd_bbox": te_out_pcd,
            "frac_train_cam_outside_pcd_bbox": tr_out_pcd,
        }

    worst_idx = np.argsort(-te_nn)[:10]

    return {
        "scene": scene_dir.name,
        "n_train": len(tr_c),
        "n_test": len(te_c),
        "train_center_min": tr_c.min(0),
        "train_center_max": tr_c.max(0),
        "train_center_mean": tr_c.mean(0),
        "train_center_std": tr_c.std(0),
        "test_center_min": te_c.min(0),
        "test_center_max": te_c.max(0),
        "test_center_mean": te_c.mean(0),
        "test_center_std": te_c.std(0),
        "train_radius": radius,
        "train_r_med": float(np.median(tr_r)),
        "test_r_med": float(np.median(te_r)),
        "test_r_max": float(te_r.max()),
        "frac_test_outside_train_radius": float((te_r > radius).mean()),
        "frac_test_outside_bbox": float(out_mask.mean()),
        "frac_test_outside_bbox_10pct": float(out10_mask.mean()),
        "train_med_spacing": train_med_spacing,
        "test_nn_min": float(te_nn.min()),
        "test_nn_med": float(np.median(te_nn)),
        "test_nn_mean": float(te_nn.mean()),
        "test_nn_p90": float(np.percentile(te_nn, 90)),
        "test_nn_max": float(te_nn.max()),
        "test_nn_over_2x_spacing": float(ood_pos.mean()),
        "ang_to_nn_med": float(np.median(ang_to_nn)),
        "ang_to_nn_p90": float(np.percentile(ang_to_nn, 90)),
        "ang_to_nn_max": float(ang_to_nn.max()),
        "best_ang_med": float(np.median(best_ang)),
        "best_ang_p90": float(np.percentile(best_ang, 90)),
        "best_ang_max": float(best_ang.max()),
        "frac_ang_to_nn_gt30": float(ood_ang.mean()),
        "frac_ood_any": float(ood_any.mean()),
        "train_fx_mean": float(tr_fx.mean()),
        "test_fx_mean": float(te_fx.mean()),
        "train_wh": tr_wh[0].tolist() if len(tr_wh) else None,
        "test_wh": te_wh[0].tolist() if len(te_wh) else None,
        "fx_ratio": float(te_fx.mean() / (tr_fx.mean() + 1e-12)),
        "pcd": pcd_stats,
        "te_nn": te_nn,
        "ang_to_nn": ang_to_nn,
        "best_ang": best_ang,
        "te_r": te_r,
        "names": test["names"],
        "ood_any": ood_any,
        "ood_pos": ood_pos,
        "ood_ang": ood_ang,
        "out_mask": out_mask,
        "worst_idx": worst_idx,
        "te_c": te_c,
        "mean_center_shift": float(np.linalg.norm(te_c.mean(0) - tr_c.mean(0))),
        "train_diag": diag,
    }


def main():
    scenes = sorted([p for p in PUBLIC.iterdir() if p.is_dir()])
    results = []
    for s in scenes:
        print("=" * 80)
        print("SCENE:", s.name)
        r = summarize_scene(s)
        if r is None:
            print("  skip (missing data)")
            continue
        results.append(r)
        print(f"  n_train={r['n_train']}  n_test={r['n_test']}")
        print("  TRAIN cam center range XYZ:")
        print(f"    min  {np.round(r['train_center_min'], 3)}")
        print(f"    max  {np.round(r['train_center_max'], 3)}")
        print(
            f"    mean {np.round(r['train_center_mean'], 3)}  "
            f"std {np.round(r['train_center_std'], 3)}"
        )
        print("  TEST  cam center range XYZ:")
        print(f"    min  {np.round(r['test_center_min'], 3)}")
        print(f"    max  {np.round(r['test_center_max'], 3)}")
        print(
            f"    mean {np.round(r['test_center_mean'], 3)}  "
            f"std {np.round(r['test_center_std'], 3)}"
        )
        print(f"  mean center shift |test-train| = {r['mean_center_shift']:.4f}")
        print(f"  train radius (max dist to mean) = {r['train_radius']:.4f}")
        print(f"  train AABB diagonal = {r['train_diag']:.4f}")
        print(f"  test r med/max = {r['test_r_med']:.4f} / {r['test_r_max']:.4f}")
        print(
            f"  frac test outside train radius = "
            f"{r['frac_test_outside_train_radius']*100:.1f}%"
        )
        print(
            f"  frac test outside train AABB = {r['frac_test_outside_bbox']*100:.1f}%  "
            f"(+10% margin: {r['frac_test_outside_bbox_10pct']*100:.1f}%)"
        )
        print(f"  train median NN spacing = {r['train_med_spacing']:.4f}")
        print(
            "  test NN-dist to train: min/med/mean/p90/max = "
            f"{r['test_nn_min']:.4f} / {r['test_nn_med']:.4f} / "
            f"{r['test_nn_mean']:.4f} / {r['test_nn_p90']:.4f} / {r['test_nn_max']:.4f}"
        )
        print(
            f"  frac test NN > 2x train spacing = "
            f"{r['test_nn_over_2x_spacing']*100:.1f}%"
        )
        print(
            "  angle to position-NN train (deg): med/p90/max = "
            f"{r['ang_to_nn_med']:.1f} / {r['ang_to_nn_p90']:.1f} / "
            f"{r['ang_to_nn_max']:.1f}"
        )
        print(
            "  best angular match to any train (deg): med/p90/max = "
            f"{r['best_ang_med']:.1f} / {r['best_ang_p90']:.1f} / "
            f"{r['best_ang_max']:.1f}"
        )
        print(f"  frac angle-to-NN > 30deg = {r['frac_ang_to_nn_gt30']*100:.1f}%")
        print(f"  frac OOD (pos|ang|bbox) = {r['frac_ood_any']*100:.1f}%")
        print(
            f"  intrinsics: train fx~{r['train_fx_mean']:.1f} wh={r['train_wh']} | "
            f"test fx~{r['test_fx_mean']:.1f} wh={r['test_wh']} | "
            f"fx_ratio={r['fx_ratio']:.3f}"
        )
        if r["pcd"]:
            p = r["pcd"]
            print(f"  point cloud: n={p['n_pts']} diag={p['pcd_diag']:.3f}")
            print(
                f"    dist cam->pcd centroid med: train={p['train_dist_to_pcd_med']:.3f} "
                f"test={p['test_dist_to_pcd_med']:.3f}"
            )
            print(
                f"    look-at-pcd angle med/p90: "
                f"train={p['train_look_at_pcd_med_deg']:.1f}/"
                f"{p['train_look_at_pcd_p90_deg']:.1f}  "
                f"test={p['test_look_at_pcd_med_deg']:.1f}/"
                f"{p['test_look_at_pcd_p90_deg']:.1f}"
            )
            print(
                f"    frac cams outside pcd bbox: "
                f"train={p['frac_train_cam_outside_pcd_bbox']*100:.1f}% "
                f"test={p['frac_test_cam_outside_pcd_bbox']*100:.1f}%"
            )
        print("  WORST 10 test poses by NN distance to train:")
        for i in r["worst_idx"]:
            flags = []
            if r["ood_pos"][i]:
                flags.append("POS")
            if r["ood_ang"][i]:
                flags.append("ANG")
            if r["out_mask"][i]:
                flags.append("BOX")
            fl = ",".join(flags) if flags else "ok"
            print(
                f"    {r['names'][i]}: nn={r['te_nn'][i]:.4f} "
                f"ang_nn={r['ang_to_nn'][i]:.1f} best_ang={r['best_ang'][i]:.1f} "
                f"r={r['te_r'][i]:.3f} C={np.round(r['te_c'][i], 3)} [{fl}]"
            )

    print("\n" + "=" * 80)
    print("SUMMARY TABLE (all public_set scenes)")
    hdr = (
        f"{'scene':12s} {'ntr':>4s} {'nte':>4s} {'nn_med':>8s} {'nn_p90':>8s} "
        f"{'nn_max':>8s} {'>2x%':>6s} {'box%':>6s} {'ang30%':>7s} "
        f"{'ood%':>6s} {'r_out%':>7s} {'fx_r':>6s} {'shift':>7s}"
    )
    print(hdr)
    for r in results:
        print(
            f"{r['scene']:12s} {r['n_train']:4d} {r['n_test']:4d} "
            f"{r['test_nn_med']:8.4f} {r['test_nn_p90']:8.4f} {r['test_nn_max']:8.4f} "
            f"{r['test_nn_over_2x_spacing']*100:5.1f}% "
            f"{r['frac_test_outside_bbox']*100:5.1f}% "
            f"{r['frac_ang_to_nn_gt30']*100:6.1f}% "
            f"{r['frac_ood_any']*100:5.1f}% "
            f"{r['frac_test_outside_train_radius']*100:6.1f}% "
            f"{r['fx_ratio']:6.3f} {r['mean_center_shift']:7.4f}"
        )

    print("\nINTERPRETATION:")
    print("- Camera center C = -R^T * t (COLMAP world-to-camera).")
    print("- nn_* = distance test cam center -> nearest train cam center.")
    print("- >2x% = test farther than 2x median train cam spacing (position OOD).")
    print("- box% = outside axis-aligned bbox of train camera centers.")
    print("- ang30% = viewing dir differs >30deg from position-nearest train cam.")
    print("- ood% = union of pos/ang/bbox flags.")
    print("- r_out% = outside max train radius from train centroid.")
    print("- fx_r = test_fx / train_fx (~1 if same camera/scale).")
    print("- shift = ||mean(test centers) - mean(train centers)||.")
    print(
        "- Black regions in 3DGS usually mean few/no Gaussians in view "
        "(OOD pose, looking away from reconstructed volume, or wrong scale)."
    )


if __name__ == "__main__":
    main()
