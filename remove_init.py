from pathlib import Path

def remove_init_files(root: Path, target_dirs=None):
    """
    Remove __init__.py files inside specific directories (e.g. build, .git).
    """
    if target_dirs is None:
        target_dirs = {"build", ".git"}

    for path in root.rglob("__init__.py"):
        if any(part in target_dirs for part in path.parts):
            try:
                path.unlink()
                print(f"Removed: {path}")
            except Exception as e:
                print(f"Failed to remove {path}: {e}")


if __name__ == "__main__":
    project_root = Path.cwd()  # run from repo root
    remove_init_files(project_root)