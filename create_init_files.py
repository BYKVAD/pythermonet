from pathlib import Path

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
}

def create_init_files(root: Path) -> None:
    for path in root.rglob("*"):
        if not path.is_dir():
            continue

        if path.name in EXCLUDE_DIRS:
            continue

        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.touch()
            print(f"Created: {init_file}")

if __name__ == "__main__":
    project_root = Path(__file__).parent
    create_init_files(project_root)
