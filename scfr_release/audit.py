"""Offline source audit and CLI inventory. No imports of optional model code."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    inventory = {}
    unresolved = []
    hashes = {}
    for path in sorted(root.rglob("*.py")):
        if any(part in {"build", ".venv", "__pycache__"} or part.endswith(".egg-info") for part in path.relative_to(root).parts):
            continue
        source = path.read_bytes()
        tree = ast.parse(source, filename=str(path.relative_to(root)))
        name = path.relative_to(root).as_posix()
        hashes[name] = hashlib.sha256(source).hexdigest()
        flags = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                names = [x.value for x in node.args if isinstance(x, ast.Constant) and isinstance(x.value, str)]
                if names:
                    flags.append(names)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(("scripts.", "src.", "train.", "utils.")):
                module = root.joinpath(*node.module.split("."))
                if not module.with_suffix(".py").exists() and not (module / "__init__.py").exists():
                    unresolved.append({"path": name, "module": node.module})
        if flags:
            inventory[name] = flags
    print(json.dumps({"python_files_parsed": len(hashes), "unresolved_project_imports": unresolved, "cli_options": inventory, "source_sha256": hashes}, indent=2))
    return int(bool(unresolved))


if __name__ == "__main__":
    raise SystemExit(main())
