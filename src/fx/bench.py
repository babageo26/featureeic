import argparse
from pathlib import Path
import csv
import sys
from pipeline import run_once

def infer_label(path: Path) -> str:
    return path.stem  # filename without extension

def main():
    ap = argparse.ArgumentParser(description="Batch feature matching to produce a wide summary table")
    ap.add_argument("--algo", choices=["SIFT","ORB","BRIEF"], required=True)
    ap.add_argument("--base", required=True, help="Path to base/original image (A)")
    ap.add_argument("--targets", nargs="+", help="Paths to target images (B1 B2 ...)")
    ap.add_argument("--dir", help="Directory containing target images (if --targets not given)")
    ap.add_argument("--exts", default="jpg,jpeg,png,JPG,JPEG,PNG", help="Comma-separated extensions to scan when using --dir")
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--ransac", type=float, default=3.0)
    ap.add_argument("--preset", choices=["AUTO","PP_DEFAULT","PP_BLUR","PP_BRIEF","NONE"], default="AUTO")
    ap.add_argument("--no-preprocess", action="store_true")
    ap.add_argument("--out", default="runs", help="Output base directory (per algo)")
    ap.add_argument("--save-vis", action="store_true", help="Save visualization images")
    args = ap.parse_args()

    # Decide list of targets
    targets = []
    if args.targets:
        targets = [Path(p) for p in args.targets]
    elif args.dir:
        dirp = Path(args.dir)
        if not dirp.is_dir():
            print(f"--dir not found: {dirp}", file=sys.stderr)
            sys.exit(2)
        allow = set(x.strip().lower() for x in args.exts.split(","))
        for p in sorted(dirp.iterdir()):
            if p.is_file() and p.suffix.lower().lstrip(".") in allow:
                targets.append(p)
    else:
        print("Please provide --targets or --dir", file=sys.stderr)
        sys.exit(2)

    # Remove base if accidentally included in targets
    base_path = Path(args.base).resolve()
    targets = [t for t in targets if t.resolve() != base_path]

    # Prepare pp kwargs
    pp_kwargs = {}
    use_preprocess = not args.no_preprocess
    if args.preset == "NONE":
        use_preprocess = False
        pp_kwargs = {}
    elif args.preset == "AUTO":
        if args.algo in ("SIFT","ORB"):
            pp_kwargs = dict(do_clahe=True, do_unsharp=True)
        else:
            pp_kwargs = dict(do_denoise=True)
    elif args.preset == "PP_DEFAULT":
        pp_kwargs = dict(do_clahe=True, do_unsharp=True)
    elif args.preset == "PP_BLUR":
        pp_kwargs = dict(do_clahe=True, do_unsharp=True, strong_unsharp=True)
    elif args.preset == "PP_BRIEF":
        pp_kwargs = dict(do_denoise=True)

    out_dir = Path(args.out) / args.algo.lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run all
    rows = []
    algo_u = args.algo.upper()
    for tgt in targets:
        label = infer_label(tgt)
        result, vis = run_once(
            algo=args.algo,
            imgA=str(base_path),
            imgB=str(tgt),
            pp_kwargs=pp_kwargs,
            use_preprocess=use_preprocess,
            ratio=args.ratio,
            ransac_thresh=args.ransac,
            draw=args.save_vis,
            save=True,
            save_dir=str(out_dir),
            label=label,
        )
        row = {
            "CONDITION": result.get("label",""),
            "ALGO": algo_u,
            f"{algo_u} MATCHES": result.get("n_match", 0),
            f"{algo_u} MATCHES TEST": result.get("n_inlier", 0),
            f"{algo_u} SPEED (ms)": int(float(result.get("t_total_s", 0.0)) * 1000),
        }
        rows.append(row)

    # Save wide table
    wide_csv = out_dir / f"summary_wide_{args.algo.lower()}.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(wide_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # Print table to console
    print("\\n== WIDE SUMMARY ==")
    if rows:
        # header
        print("\\t".join(rows[0].keys()))
        for r in rows:
            print("\\t".join(str(r[k]) for k in rows[0].keys()))
        print(f"\\nSaved CSV: {wide_csv}")
    else:
        print("No rows.")
if __name__ == "__main__":
    main()
