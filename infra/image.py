"""Shared Isaac Sim + OceanSim image for sim jobs on Modal (owner: P2).

Needs a Modal secret called "ngc" with:
    REGISTRY_USERNAME=$oauthtoken
    REGISTRY_PASSWORD=<your NGC API key from ngc.nvidia.com>
"""
import os
import subprocess

import modal

# 5.x segfaults on Modal driver 610 (see PLAN.md §3). Team standard is 6.0.1.
ISAAC_TAG = os.environ.get("ISAAC_TAG", "nvcr.io/nvidia/isaac-sim:6.0.1")
OCEANSIM_REF = os.environ.get("OCEANSIM_REF", "main")
OCEANSIM_DIR = "/isaac-sim/extsUser/OceanSim"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Shader / kit caches persist here so only the first Isaac start is slow.
cache_volume = modal.Volume.from_name("isaac-cache", create_if_missing=True)
CACHE_MOUNT = "/cache"
CACHE_DIRS = [
    "/root/.cache/ov",
    "/root/.cache/nvidia/GLCache",
    "/root/.nv/ComputeCache",
    "/root/.local/share/ov/data",
]

isaac_image = (
    modal.Image.from_registry(
        ISAAC_TAG,
        secret=modal.Secret.from_name("ngc"),
        add_python="3.11",
    )
    .dockerfile_commands(["USER root"])
    .apt_install("vulkan-tools", "pciutils", "ffmpeg", "git")
    # OceanSim extension (targets Isaac 5.0; we run it on 6.0.1). Commit hash recorded for pinning.
    .run_commands(
        f"git clone https://github.com/umfieldrobotics/OceanSim.git {OCEANSIM_DIR}"
        f" && cd {OCEANSIM_DIR} && git checkout {OCEANSIM_REF}"
        f" && git rev-parse HEAD > {OCEANSIM_DIR}.commit && cat {OCEANSIM_DIR}.commit"
    )
    .env(
        {
            "ACCEPT_EULA": "Y",
            "PRIVACY_CONSENT": "Y",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            # Ask the runtime for graphics (Vulkan) libs, not just compute.
            "NVIDIA_DRIVER_CAPABILITIES": "all",
            "XDG_RUNTIME_DIR": "/tmp/xdg",
        }
    )
    .entrypoint([])
    # Modal 1.x doesn't auto-ship local modules; job files do `from image import ...`.
    .add_local_python_source("image")
    .add_local_dir(os.path.join(REPO_ROOT, "sim"), remote_path="/root/repo/sim")
    # Mission loop imports the real detector / planner / imagination when they exist.
    .add_local_dir(os.path.join(REPO_ROOT, "perception"), remote_path="/root/repo/perception", ignore=["weights", "*.pt", "__pycache__"])
    .add_local_dir(os.path.join(REPO_ROOT, "agent"), remote_path="/root/repo/agent", ignore=["__pycache__"])
)


def link_caches() -> None:
    """Point Isaac's cache dirs at the Volume. Call at the top of every sim job."""
    os.makedirs("/tmp/xdg", exist_ok=True)
    for path in CACHE_DIRS:
        target = os.path.join(CACHE_MOUNT, path.strip("/").replace("/", "_"))
        os.makedirs(target, exist_ok=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.islink(path):
            continue
        if os.path.exists(path):
            subprocess.run(["rm", "-rf", path], check=True)
        os.symlink(target, path)


def run_isaac(script: str, *args: str, timeout: int = 1800) -> int:
    """Run a script under Isaac's Python, streaming its output to the Modal logs."""
    cmd = ["/isaac-sim/python.sh", script, *args]
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd="/root/repo", timeout=timeout).returncode
