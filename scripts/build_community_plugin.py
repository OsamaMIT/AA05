"""Export the racing MPC as a self-contained AVLite community plugin.

Only the transitive Python dependencies of the controller and planners are
included. Imports become package-relative, so no checkout or sys.path changes
are needed when AVLite loads the exported directory under its plugin namespace.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
NAME = "avlite_a2rl_mpc"
RENAMES = {
    "A2RLMPCControlStrategy": "A2RLTubeMPCControlStrategy",
    "A2RLGlobalRacePlanner": "A2RLMPCGlobalRacePlanner",
    "A2RLReferenceLocalPlanner": "A2RLMPCReferenceLocalPlanner",
}


def collect_modules() -> dict[str, str]:
    pending = ["avlite_stack.controller", "avlite_stack.planning"]
    modules = {}
    while pending:
        module = pending.pop()
        if module in modules:
            continue
        path = ROOT / "src/chrono_a2rl" / (module.replace(".", "/") + ".py")
        source = path.read_text(encoding="utf-8")
        modules[module] = source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("chrono_a2rl."):
                    pending.append(node.module.removeprefix("chrono_a2rl."))
            elif isinstance(node, ast.Import):
                if any(n.name.startswith("chrono_a2rl") for n in node.names):
                    raise ValueError(f"Unsupported absolute package import in {module}")
    return modules


def build_plugin(destination: Path) -> Path:
    destination = destination.resolve()
    # Never merge with or delete an existing plugin installation.
    destination.mkdir(parents=True, exist_ok=False)
    modules = collect_modules()
    for module, source in sorted(modules.items()):
        target = destination / "core" / (module.replace(".", "/") + ".py")
        target.parent.mkdir(parents=True, exist_ok=True)
        parent = target.parent
        while parent != destination:
            (parent / "__init__.py").touch()
            parent = parent.parent
        prefix = "." * (module.count(".") + 2) + "core."
        source = re.sub(r"from chrono_a2rl\.", f"from {prefix}", source)
        for old, new in RENAMES.items():
            source = source.replace(old, new)
        if module == "common.config":
            source = source.replace(
                "Path(__file__).resolve().parents[3]",
                'Path(__file__).resolve().parents[2] / "assets"',
            )
        if module == "avlite_stack.settings":
            source = source.replace(
                '    filepath: str = "configs/plugin_avlite_a2rl.yaml"\n', ""
            ).replace(
                "configs/experiments/rl_planner_yas_marina.yaml",
                "configs/experiments/community_mpc_yas_marina.yaml",
            )
        target.write_text(source, encoding="utf-8")

    assets = destination / "assets"
    # Component configuration references remain relative to this portable root.
    for folder in ("configs", "vehicles"):
        for path in (ROOT / folder).rglob("*.yaml"):
            target = assets / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for path in (ROOT / "tracks/yas_marina").rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".yaml", ".md"} and "raw" not in path.parts:
            target = assets / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)

    (destination / "__init__.py").write_text(
        '"""A2RL Tube MPC and raceline planning for AVLite."""\n'
        "from .settings import PluginSettings\n"
        "from .core.avlite_stack.controller import A2RLTubeMPCControlStrategy\n"
        "from .core.avlite_stack.planning import (\n"
        "    A2RLMPCGlobalRacePlanner, A2RLMPCReferenceLocalPlanner,\n)\n",
        encoding="utf-8",
    )
    (destination / "settings.py").write_text(
        # AVLite also executes settings.py outside its package namespace.
        "from avlite.plugins.avlite_a2rl_mpc.core.avlite_stack.settings import A2RLStackSettings\n\n"
        "PluginSettings = A2RLStackSettings\n", encoding="utf-8",
    )
    (destination / "requirements.txt").write_text(
        "avlite>=0.5.3,<0.6\nnumpy>=2.0\nscipy>=1.11\npandas>=2.0\n"
        "pyyaml>=6.0\nosqp>=1.0\n", encoding="utf-8",
    )
    shutil.copyfile(ROOT / "docs/COMMUNITY_MPC_PLUGIN.md", destination / "README.md")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist" / NAME)
    args = parser.parse_args()
    plugin = build_plugin(args.output)
    archive = shutil.make_archive(str(plugin), "zip", plugin.parent, plugin.name)
    print(f"Plugin: {plugin}\nArchive: {archive}")


if __name__ == "__main__":
    main()
