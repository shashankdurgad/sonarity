"""Step 1 of the OceanSim proof: can an L40S container on Modal see Vulkan?

Result on 7 Nov: passes, but Isaac 5.x then crashes at render; use 6.0.1 (see step 2).

Run: modal run infra/step1_vulkan.py
"""
import json
import os
import subprocess

import modal

image = (
    modal.Image.from_registry("nvcr.io/nvidia/isaac-sim:5.0.0", add_python="3.11")
    .dockerfile_commands("USER root")
    .entrypoint([])
    .apt_install("vulkan-tools")
    .env({"ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"})
)
app = modal.App("uxo-step1-vulkan", image=image)

ICD = "/etc/vulkan/icd.d/nvidia_icd.json"


def sh(cmd: str, limit: int = 3000) -> int:
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
    print((r.stdout + r.stderr)[:limit], flush=True)
    print(f"[exit {r.returncode}]", flush=True)
    return r.returncode


@app.function(gpu=os.environ.get("TEST_GPU", "L40S"), timeout=300, scaledown_window=60)
def check() -> None:
    sh("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
    print("\nNVIDIA_DRIVER_CAPABILITIES =", os.environ.get("NVIDIA_DRIVER_CAPABILITIES"), flush=True)
    sh("uname -a; id; cat /etc/os-release | head -2; dmesg 2>/dev/null | head -3")
    sh("ls -la /etc/vulkan/icd.d /usr/share/vulkan/icd.d 2>&1; cat " + ICD + " 2>&1")
    sh("ldconfig -p | grep -E 'libGLX_nvidia|libEGL_nvidia|libnvidia-glcore|libnvidia-glvkspirv|libvulkan'")
    sh("ls -la /dev/nvidia* /dev/dri 2>&1")
    sh("find / -xdev \\( -name 'libGLX_nvidia*' -o -name 'libnvidia-glcore*' \\) 2>/dev/null | head")

    print("\n=== vulkaninfo with the image's own ICD ===", flush=True)
    rc = sh("vulkaninfo --summary")
    if rc == 0 and subprocess.run("vulkaninfo --summary | grep -qi 'NVIDIA'", shell=True).returncode == 0:
        print("\nRESULT: PASS, Vulkan sees the NVIDIA GPU with the image's ICD", flush=True)
        return

    print("\n=== retry with a written NVIDIA ICD ===", flush=True)
    os.makedirs(os.path.dirname(ICD), exist_ok=True)
    with open(ICD, "w") as f:
        json.dump({"file_format_version": "1.0.0",
                   "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3"}}, f)
    rc = sh("VK_ICD_FILENAMES=" + ICD + " vulkaninfo --summary")
    ok = rc == 0 and subprocess.run(
        "VK_ICD_FILENAMES=" + ICD + " vulkaninfo --summary | grep -qi 'NVIDIA'", shell=True
    ).returncode == 0
    print("\nRESULT:", "PASS, Vulkan works once the ICD is written" if ok else "FAIL, Vulkan does not see the GPU", flush=True)


@app.local_entrypoint()
def main() -> None:
    check.remote()
