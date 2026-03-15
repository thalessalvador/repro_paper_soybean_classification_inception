import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera graficos (matplotlib) para um run de treino.")
    parser.add_argument("--run-dir", required=True, type=str, help="Diretorio do run.")
    parser.add_argument("--out-dir", default=None, type=str, help="Diretorio de saida dos graficos.")
    return parser.parse_args()


def load_history(path: Path) -> dict:
    data = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v) if v else np.nan)
    return data


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split(",")])
    return np.array(rows, dtype=int)


def load_class_names(split_manifest: Path) -> list[str]:
    idx_to_name = {}
    with split_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_name[int(row["class_idx"])] = row["class_name"]
    return [idx_to_name[i] for i in sorted(idx_to_name.keys())]


def plot_accuracy(history: dict, test_acc: float, out_path: Path) -> None:
    epochs = np.array(history["epoch"], dtype=int)
    train_acc = np.array(history["accuracy"], dtype=float)
    val_acc = np.array(history["val_accuracy"], dtype=float)

    plt.figure(figsize=(11, 6), dpi=130)
    plt.plot(epochs, train_acc, label="Treino", linewidth=2)
    plt.plot(epochs, val_acc, label="Validacao", linewidth=2)
    plt.axhline(test_acc, linestyle="--", linewidth=2, label=f"Teste final ({test_acc:.4f})")
    plt.title("Evolucao da Acuracia")
    plt.xlabel("Epoca")
    plt.ylabel("Acuracia")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss(history: dict, test_loss: float, out_path: Path) -> None:
    epochs = np.array(history["epoch"], dtype=int)
    train_loss = np.array(history["loss"], dtype=float)
    val_loss = np.array(history["val_loss"], dtype=float)

    plt.figure(figsize=(11, 6), dpi=130)
    plt.plot(epochs, train_loss, label="Treino", linewidth=2)
    plt.plot(epochs, val_loss, label="Validacao", linewidth=2)
    plt.axhline(test_loss, linestyle="--", linewidth=2, label=f"Teste final ({test_loss:.4f})")
    plt.title("Evolucao da Loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion(cm: np.ndarray, class_names: list[str], out_path: Path) -> None:
    plt.figure(figsize=(9, 7), dpi=150)
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Matriz de Confusao")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    plt.xlabel("Classe predita")
    plt.ylabel("Classe real")

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, f"{cm[i, j]}", ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(run_dir / "history.csv")
    metrics = load_metrics(run_dir / "metrics.json")
    cm = load_confusion(run_dir / "confusion_matrix.csv")
    class_names = load_class_names(run_dir / "split_manifest.csv")

    test_metrics = metrics.get("keras_evaluate", {})
    test_acc = float(test_metrics.get("accuracy", np.nan))
    test_loss = float(test_metrics.get("loss", np.nan))

    plot_accuracy(history, test_acc, out_dir / "plot_accuracy.png")
    plot_loss(history, test_loss, out_dir / "plot_loss.png")
    plot_confusion(cm, class_names, out_dir / "plot_confusion_matrix.png")

    print(f"Graficos gerados em: {out_dir}")
    print("- plot_accuracy.png")
    print("- plot_loss.png")
    print("- plot_confusion_matrix.png")


if __name__ == "__main__":
    main()
