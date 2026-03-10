"""
System Check Module - Detects GPU, VRAM, RAM and determines operating mode.
Operating modes:
  - "high"   (>=8 GB VRAM): models stay loaded in GPU memory
  - "medium" (5-8 GB VRAM): selective caching
  - "low"    (<5 GB VRAM):  strict load->use->unload per operation
"""

import platform
import psutil


def get_system_info() -> dict:
    """Gather comprehensive system information."""
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "gpu_available": False,
        "gpu_name": "N/A",
        "vram_total_gb": 0.0,
        "vram_free_gb": 0.0,
        "cuda_version": "N/A",
        "torch_version": "N/A",
        "operating_mode": "cpu",
    }

    # Check PyTorch and CUDA
    try:
        import torch
        info["torch_version"] = torch.__version__

        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda or "N/A"

            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            vram_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (1024**3)

            info["vram_total_gb"] = round(vram_total, 2)
            info["vram_free_gb"] = round(vram_free, 2)

            # Determine operating mode based on VRAM
            if vram_total >= 8:
                info["operating_mode"] = "high"
            elif vram_total >= 5:
                info["operating_mode"] = "medium"
            else:
                info["operating_mode"] = "low"
        else:
            info["operating_mode"] = "cpu"
    except ImportError:
        info["operating_mode"] = "cpu"

    return info


def get_operating_mode() -> str:
    """Get just the operating mode string."""
    return get_system_info()["operating_mode"]


def get_recommended_model_size(info: dict = None) -> str:
    """Recommend model size based on available VRAM."""
    if info is None:
        info = get_system_info()

    if info["vram_total_gb"] >= 12:
        return "1.7B"
    elif info["vram_total_gb"] >= 4:
        return "0.6B"
    else:
        return "0.6B"  # CPU fallback will be slow but possible


def print_system_report(info: dict = None):
    """Print a formatted system report."""
    if info is None:
        info = get_system_info()

    recommended = get_recommended_model_size(info)

    print("=" * 50)
    print("  SYSTEM INFORMATION")
    print("=" * 50)
    print(f"  OS:             {info['os']} {info['os_version'][:30]}")
    print(f"  CPU:            {info['cpu'][:50]}")
    print(f"  RAM:            {info['ram_available_gb']:.1f} / {info['ram_total_gb']:.1f} GB")
    print(f"  PyTorch:        {info['torch_version']}")
    print("-" * 50)

    if info["gpu_available"]:
        print(f"  GPU:            {info['gpu_name']}")
        print(f"  CUDA:           {info['cuda_version']}")
        print(f"  VRAM:           {info['vram_free_gb']:.1f} / {info['vram_total_gb']:.1f} GB")
    else:
        print("  GPU:            Not available (CPU mode)")

    print("-" * 50)
    mode_labels = {
        "high": "HIGH - Models stay loaded (fast)",
        "medium": "MEDIUM - Selective caching",
        "low": "LOW - Load/unload per operation (slow but safe)",
        "cpu": "CPU ONLY - Very slow, GPU recommended",
    }
    print(f"  Mode:           {mode_labels.get(info['operating_mode'], info['operating_mode'])}")
    print(f"  Recommended:    {recommended} parameter model")
    print("=" * 50)


if __name__ == "__main__":
    info = get_system_info()
    print_system_report(info)
