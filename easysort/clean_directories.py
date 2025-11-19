from pathlib import Path
from tqdm import tqdm
import argparse
import os


def list_subdirs(path: Path) -> list[Path]:
    """List immediate subdirectories of a path."""
    if not path.exists() or not path.is_dir():
        return []
    # os.scandir is faster than Path.iterdir for large dirs
    try:
        with os.scandir(path) as it:
            return [Path(entry.path) for entry in it if entry.is_dir(follow_symlinks=False)]
    except PermissionError:
        return []


def delete_empty_dirs(root: Path) -> int:
    """
    Delete empty directories under root (bottom-up).
    Keeps files (e.g., .json) intact; only prunes directories with no contents.
    Returns number of directories removed.
    """
    if not root.exists():
        return 0
    # Collect all directories bottom-up
    all_dirs = [p for p in root.rglob("*") if p.is_dir()]
    removed = 0
    for d in tqdm(sorted(all_dirs, key=lambda p: len(p.parts), reverse=True),
                  desc=f"Pruning empty dirs under {root.name}"):
        try:
            # If directory is empty now, remove it
            if d.exists() and not any(d.iterdir()):
                d.rmdir()
                removed += 1
        except OSError:
            # Not empty or permission issues; skip
            continue
    # Finally try root itself (but do not remove project root)
    return removed


def prune_empty_dirs_fast(root: Path) -> int:
    """
    Fast global prune: collect all directories once, traverse bottom-up and remove empties.
    Significantly faster than pruning each subtree repeatedly.
    """
    if not root.exists():
        return 0
    # Single walk to collect directories
    dir_list: list[Path] = []
    for r, _, _ in os.walk(root):
        dir_list.append(Path(r))
    # Bottom-up by depth
    removed = 0
    for d in tqdm(sorted(dir_list, key=lambda p: len(p.parts), reverse=True), desc=f"Global prune in {root}"):
        try:
            if d.exists() and d.is_dir() and not any(os.scandir(d)):
                d.rmdir()
                removed += 1
        except OSError:
            continue
    return removed


def clean_registry(root_path: str) -> None:
    """
    Traverse a results registry progressively by levels and clean empty dirs
    inside final project folders like:
      root/argo/Device/Year/Month/Day/Hour/Timestamp/<model>/<project>
    """
    root = Path(root_path)
    assert root.exists(), f"Root path not found: {root}"

    argo_root = root / "argo"
    if not argo_root.exists():
        # Fallback to root if "argo" isn't present
        argo_root = root

    # One fast global prune first to drop most empties up front
    tqdm.write("Starting fast global prune of empty directories...")
    removed_global = prune_empty_dirs_fast(argo_root)
    tqdm.write(f"Global prune removed {removed_global} empty directories")

    # Level-by-level traversal with tqdm to show progress and scale
    devices = list_subdirs(argo_root)
    devices = sorted(devices)
    for device in tqdm(devices, desc="Devices"):
        years = sorted(list_subdirs(device))
        for year in tqdm(years, desc=f"Years in {device.name}", leave=False):
            months = sorted(list_subdirs(year))
            for month in tqdm(months, desc=f"Months in {device.name}/{year.name}", leave=False):
                days = sorted(list_subdirs(month))
                for day in tqdm(days, desc=f"Days in {device.name}/{year.name}/{month.name}", leave=False):
                    hours = sorted(list_subdirs(day))
                    for hour in tqdm(hours, desc=f"Hours in .../{day.name}", leave=False):
                        timestamps = sorted(list_subdirs(hour))
                        for ts in tqdm(timestamps, desc=f"Timestamps in .../{hour.name}", leave=False):
                            models = sorted(list_subdirs(ts))
                            for model in tqdm(models, desc=f"Models in .../{ts.name}", leave=False):
                                projects = sorted(list_subdirs(model))
                                for project in tqdm(projects, desc=f"Projects in .../{model.name}", leave=False):
                                    # Inside project directory; keep .json files, delete empty dirs
                                    # Show summary of files present
                                    json_files = list(project.glob("*.json"))
                                    tqdm.write(f"Project: {project} | JSON files: {len(json_files)}")
                                    # No per-project prune (expensive). Rely on global prune(s).
                                    # If desired, uncomment the next two lines for targeted prune:
                                    # removed = delete_empty_dirs(project)
                                    # tqdm.write(f\"Removed empty dirs: {removed} under {project}\")

    # Final global prune to catch directories that became empty after traversal
    tqdm.write("Final fast global prune of empty directories...")
    removed_global_end = prune_empty_dirs_fast(argo_root)
    tqdm.write(f"Final global prune removed {removed_global_end} empty directories")


def write_all_paths(root_path: str, output_path: str = "paths.txt", suffix: str = "") -> None:
    """
    Very fast listing of all files under root (skipping directories),
    optionally filtering by suffix, and writing to output_path.
    """
    root = Path(root_path)
    assert root.exists(), f"Root path not found: {root}"

    total_written = 0
    with open(output_path, "w") as out_f:
        pbar = tqdm(desc="Writing file paths", unit=" files")
        for r, _, files in os.walk(root):
            if suffix:
                files = [f for f in files if f.endswith(suffix)]
            # skip macOS metadata
            files = [f for f in files if not f.startswith("._")]
            if not files:
                continue
            lines = [str(Path(r) / f) + "\n" for f in files]
            out_f.writelines(lines)
            total_written += len(lines)
            pbar.update(len(lines))
        pbar.close()
    tqdm.write(f"Wrote {total_written} file paths to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Registry utilities: clean empty dirs or quickly dump file paths.")
    parser.add_argument("--root", type=str, required=True,
                        help="Root path to the registry (e.g., /mnt/c/Users/lucas/Desktop/results/)")
    parser.add_argument("--write-paths", action="store_true",
                        help="Write all file paths under --root to paths.txt (skips directories) and exit.")
    parser.add_argument("--output", type=str, default="paths.txt",
                        help="Output file for --write-paths (default: paths.txt)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Optional suffix filter for --write-paths (e.g., .json)")
    args = parser.parse_args()
    if args.write_paths:
        write_all_paths(args.root, args.output, args.suffix)
        return
    # clean_registry(args.root)


if __name__ == "__main__":
    main()


