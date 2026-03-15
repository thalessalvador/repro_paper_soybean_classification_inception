import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando do gerador de graficos PNG.

    Parametros de entrada:
        Nenhum parametro posicional direto. A funcao le os argumentos referentes
        ao diretorio do experimento e ao diretorio opcional de saida das imagens.

    Parametros de saida:
        argparse.Namespace: objeto com a configuracao necessaria para a execucao
        do script.
    """
    parser = argparse.ArgumentParser(description="Gera graficos (matplotlib) para um run de treino.")
    parser.add_argument("--run-dir", required=True, type=str, help="Diretorio do run.")
    parser.add_argument("--out-dir", default=None, type=str, help="Diretorio de saida dos graficos.")
    return parser.parse_args()


def load_history(path: Path) -> dict:
    """Carrega o historico de treinamento salvo em CSV.

    Parametros de entrada:
        path (Path): caminho do arquivo ``history.csv`` do experimento.

    Parametros de saida:
        dict: dicionario em que cada chave representa uma metrica e cada valor
        contem a lista de valores por epoca.
    """
    data = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(float(v) if v else np.nan)
    return data


def load_metrics(path: Path) -> dict:
    """Carrega o arquivo JSON de metricas do experimento.

    Parametros de entrada:
        path (Path): caminho do arquivo ``metrics.json``.

    Parametros de saida:
        dict: dicionario com as metricas carregadas do arquivo.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion(path: Path) -> np.ndarray:
    """Carrega a matriz de confusao de um CSV simples para um array NumPy.

    Parametros de entrada:
        path (Path): caminho do arquivo ``confusion_matrix.csv``.

    Parametros de saida:
        np.ndarray: matriz de confusao com tipo inteiro.
    """
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([int(x) for x in line.split(",")])
    return np.array(rows, dtype=int)


def load_class_names(split_manifest: Path) -> list[str]:
    """Recupera os nomes de classe ordenados a partir do manifesto do split.

    Parametros de entrada:
        split_manifest (Path): caminho do arquivo ``split_manifest.csv`` que
        mapeia indices numericos para nomes textuais de classe.

    Parametros de saida:
        list[str]: lista ordenada pelos indices das classes.
    """
    idx_to_name = {}
    with split_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx_to_name[int(row["class_idx"])] = row["class_name"]
    return [idx_to_name[i] for i in sorted(idx_to_name.keys())]


def plot_accuracy(history: dict, test_acc: float, out_path: Path) -> None:
    """Gera e salva o grafico PNG da evolucao de acuracia.

    Parametros de entrada:
        history (dict): historico de treinamento contendo as series por epoca.
        test_acc (float): valor de acuracia final medido no conjunto de teste.
        out_path (Path): caminho do arquivo PNG de saida.

    Parametros de saida:
        None: a funcao salva o grafico em disco e nao retorna valor.
    """
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
    """Gera e salva o grafico PNG da evolucao da funcao de perda.

    Parametros de entrada:
        history (dict): historico de treinamento contendo as series por epoca.
        test_loss (float): valor final de loss medido no conjunto de teste.
        out_path (Path): caminho do arquivo PNG de saida.

    Parametros de saida:
        None: a funcao salva o grafico em disco e nao retorna valor.
    """
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
    """Gera e salva o grafico PNG da matriz de confusao.

    Parametros de entrada:
        cm (np.ndarray): matriz de confusao com contagens por classe.
        class_names (list[str]): nomes das classes para rotular os eixos.
        out_path (Path): caminho do arquivo PNG de saida.

    Parametros de saida:
        None: a funcao salva o grafico em disco e nao retorna valor.
    """
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
    """Coordena a leitura dos artefatos do run e a geracao dos graficos PNG.

    Parametros de entrada:
        Nenhum parametro direto. A funcao utiliza os argumentos obtidos em
        ``parse_args()`` para localizar os arquivos do experimento e decidir o
        diretorio de saida das imagens.

    Parametros de saida:
        None: a funcao gera os arquivos PNG e imprime um resumo no terminal,
        sem retornar valor.
    """
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
