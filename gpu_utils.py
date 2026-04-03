import numpy as np


def get_array_backend():
    """
    Return (xp, using_gpu):
    - xp = cupy when available with a working CUDA runtime
    - xp = numpy otherwise
    """
    try:
        import cupy as cp
        _ = cp.cuda.runtime.getDeviceCount()
        return cp, True
    except Exception:
        return np, False


def to_numpy(arr):
    if isinstance(arr, np.ndarray):
        return arr
    try:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return cp.asnumpy(arr)
    except Exception:
        pass
    return np.asarray(arr)


def configure_tensorflow_gpu(verbose: bool = True):
    """
    Enable TensorFlow GPU memory growth when GPUs are available.
    Returns:
      {"tf_available": bool, "gpu_count": int, "gpus": list[str]}
    """
    info = {"tf_available": False, "gpu_count": 0, "gpus": []}
    try:
        import tensorflow as tf
        info["tf_available"] = True
        gpus = tf.config.list_physical_devices("GPU")
        info["gpu_count"] = len(gpus)
        info["gpus"] = [g.name for g in gpus]
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass

    if verbose:
        if not info["tf_available"]:
            print("[GPU] TensorFlow not available.")
        elif info["gpu_count"] == 0:
            print("[GPU] TensorFlow loaded, but no GPU device was detected.")
        else:
            print(f"[GPU] TensorFlow using {info['gpu_count']} GPU(s): {info['gpus']}")
    return info


def detect_npu_runtime(verbose: bool = True):
    """
    Detect NPU-capable runtime providers via ONNX Runtime.
    Returns:
      {
        "ort_available": bool,
        "providers": list[str],
        "npu_provider": str | None
      }
    """
    info = {"ort_available": False, "providers": [], "npu_provider": None}
    preferred = [
        "QNNExecutionProvider",
        "OpenVINOExecutionProvider",
        "CoreMLExecutionProvider",
        "NNAPIExecutionProvider",
        "DmlExecutionProvider",
    ]

    try:
        import onnxruntime as ort

        info["ort_available"] = True
        providers = ort.get_available_providers()
        info["providers"] = providers

        for provider in preferred:
            if provider in providers:
                info["npu_provider"] = provider
                break
    except Exception:
        pass

    if verbose:
        if not info["ort_available"]:
            print("[NPU] ONNX Runtime not available.")
        elif info["npu_provider"] is None:
            print(f"[NPU] No NPU provider detected. Providers={info['providers']}")
        else:
            print(f"[NPU] Using provider: {info['npu_provider']}")
    return info


def require_accelerators(require_gpu: bool = True, require_npu: bool = True):
    """
    Hard-fail if required accelerators are not detected.
    GPU check: TensorFlow GPU or CuPy CUDA backend.
    NPU check: ONNX Runtime NPU-capable provider.
    """
    tf_info = configure_tensorflow_gpu(verbose=True)
    _, using_cupy_gpu = get_array_backend()
    npu_info = detect_npu_runtime(verbose=True)

    gpu_ok = bool(tf_info.get("gpu_count", 0) > 0 or using_cupy_gpu)
    npu_ok = bool(npu_info.get("npu_provider"))

    if require_gpu and not gpu_ok:
        raise RuntimeError(
            "GPU is required but not detected by TensorFlow or CuPy. "
            "Install CUDA-enabled TensorFlow and/or CuPy with a matching CUDA toolkit."
        )
    if require_npu and not npu_ok:
        raise RuntimeError(
            "NPU is required but no ONNX Runtime NPU provider was detected. "
            "Install/enable an ONNX Runtime build/provider for your NPU if you need NPU execution."
        )
    return {"gpu_ok": gpu_ok, "npu_ok": npu_ok, "tf": tf_info, "npu": npu_info}


def log_runtime_gpu_status(context: str):
    xp, using_gpu = get_array_backend()
    if using_gpu:
        try:
            device_name = xp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
        except Exception:
            device_name = "unknown"
        print(f"[GPU] {context}: CuPy backend active on device: {device_name}")
    else:
        print(f"[GPU] {context}: NumPy backend active (CPU fallback).")
