import os
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("TF_DISABLE_MLIR_BRIDGE", "1")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")

try:
    import tensorflow as tf
except Exception as exc:
    print(f"ERRO ao importar TensorFlow: {exc}")
    sys.exit(1)


def run_cmd(command):
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        return result.returncode, result.stdout.strip(), result.stderr.strip()
    except Exception as exc:
        return 1, "", str(exc)


def print_section(title):
    print(f"\n=== {title} ===", flush=True)


def verify():
    print_section("Versoes")
    print(f"Python: {sys.version.split()[0]}")
    print(f"TensorFlow: {tf.__version__}")
    try:
        build = tf.sysconfig.get_build_info()
        print(f"TF CUDA build: {build.get('cuda_version', 'desconhecido')}")
        print(f"TF cuDNN build: {build.get('cudnn_version', 'desconhecido')}")
    except Exception:
        pass
    print(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', '(vazio)')}")

    print_section("Driver NVIDIA")
    if shutil.which("nvidia-smi"):
        code, out, err = run_cmd(["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"])
        if code == 0:
            print(out)
        else:
            print(f"Falha ao executar nvidia-smi (rc={code}): {err or '(sem stderr)'}")
            code2, out2, err2 = run_cmd(["nvidia-smi"])
            if code2 == 0:
                print("Saida de fallback do nvidia-smi:")
                print(out2)
            else:
                print(f"Fallback nvidia-smi tambem falhou (rc={code2}): {err2 or '(sem stderr)'}")
    else:
        print("nvidia-smi nao encontrado no PATH.")

    print_section("CUDA Toolkit")
    if shutil.which("nvcc"):
        code, out, err = run_cmd(["nvcc", "--version"])
        if code == 0:
            lines = [ln for ln in out.splitlines() if ln.strip()]
            print(lines[-1] if lines else out)
        else:
            print(f"Falha ao executar nvcc: {err}")
    else:
        print("nvcc nao encontrado no PATH (opcional para treino com wheel and-cuda).")

    print_section("Dispositivos TensorFlow")
    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("ERRO: TensorFlow nao detectou GPU.")
        sys.exit(2)

    for gpu in gpus:
        print(gpu)

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    print_section("Teste MatMul em GPU")
    try:
        with tf.device("/GPU:0"):
            a = tf.random.normal((4096, 4096))
            b = tf.random.normal((4096, 4096))
            start = time.time()
            c = tf.matmul(a, b)
            _ = c.numpy()
            elapsed = time.time() - start
        print(f"OK: MatMul executado em {elapsed:.3f}s")
    except Exception as exc:
        print(f"ERRO no MatMul GPU: {exc}")
        sys.exit(3)

    print_section("Teste Conv2D em GPU")
    try:
        with tf.device("/GPU:0"):
            x = tf.random.normal((8, 224, 224, 3))
            y = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
            _ = y.numpy()
        print("OK: Conv2D executou com sucesso")
    except Exception as exc:
        print(f"ERRO no Conv2D GPU: {exc}")
        sys.exit(4)

    print_section("Resultado")
    print("GPU funcional para treino TensorFlow neste ambiente WSL.")


if __name__ == "__main__":
    verify()
