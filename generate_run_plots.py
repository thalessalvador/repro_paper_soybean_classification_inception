import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera graficos SVG de um run de treino.")
    parser.add_argument("--run-dir", type=str, required=True, help="Diretorio do run (history/metrics/confusion).")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Diretorio de saida dos graficos. Se omitido, usa o proprio run-dir.",
    )
    return parser.parse_args()


def load_history(path: Path) -> Dict[str, List[float]]:
    data: Dict[str, List[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, [])
                if v is None or v == "":
                    data[k].append(float("nan"))
                else:
                    data[k].append(float(v))
    return data


def load_metrics(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_confusion(path: Path) -> List[List[int]]:
    matrix: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            matrix.append([int(x) for x in line.split(",")])
    return matrix


def load_class_names(split_manifest: Path) -> List[str]:
    mapping: Dict[int, str] = {}
    with split_manifest.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["class_idx"])
            mapping[idx] = row["class_name"]
    return [mapping[i] for i in sorted(mapping.keys())]


def map_x(i: int, n: int, x0: float, x1: float) -> float:
    if n <= 1:
        return (x0 + x1) / 2.0
    return x0 + (x1 - x0) * (i / (n - 1))


def map_y(v: float, vmin: float, vmax: float, y0: float, y1: float) -> float:
    if vmax <= vmin:
        return (y0 + y1) / 2.0
    ratio = (v - vmin) / (vmax - vmin)
    return y1 - ratio * (y1 - y0)


def polyline(points: List[Tuple[float, float]], color: str, width: int = 2) -> str:
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" />'


def line(x1: float, y1: float, x2: float, y2: float, color: str, width: int = 1, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr} />'
    )


def text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle", color: str = "#1f2937") -> str:
    safe = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-family="Arial, sans-serif">{safe}</text>'
    )


def make_metric_svg(
    title: str,
    y_label: str,
    train_vals: List[float],
    val_vals: List[float],
    test_value: float,
    out_path: Path,
) -> None:
    w, h = 1100, 650
    ml, mr, mt, mb = 90, 40, 70, 90
    x0, x1 = ml, w - mr
    y0, y1 = mt, h - mb

    vals = [v for v in train_vals + val_vals if v == v]
    if test_value == test_value:
        vals.append(test_value)
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0
    pad = (vmax - vmin) * 0.08 if vmax > vmin else 0.1
    vmin -= pad
    vmax += pad

    n = len(train_vals)
    train_points = [(map_x(i, n, x0, x1), map_y(v, vmin, vmax, y0, y1)) for i, v in enumerate(train_vals)]
    val_points = [(map_x(i, n, x0, x1), map_y(v, vmin, vmax, y0, y1)) for i, v in enumerate(val_vals)]
    test_y = map_y(test_value, vmin, vmax, y0, y1)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff" />')

    for j in range(6):
        yy = y0 + (y1 - y0) * (j / 5)
        val = vmax - (vmax - vmin) * (j / 5)
        parts.append(line(x0, yy, x1, yy, "#e5e7eb", 1))
        parts.append(text(x0 - 10, yy + 4, f"{val:.3f}", size=11, anchor="end", color="#4b5563"))

    epoch_ticks = min(10, max(2, n))
    for j in range(epoch_ticks):
        idx = round((n - 1) * j / (epoch_ticks - 1)) if epoch_ticks > 1 else 0
        xx = map_x(idx, n, x0, x1)
        parts.append(line(xx, y0, xx, y1, "#f3f4f6", 1))
        parts.append(text(xx, y1 + 22, str(idx + 1), size=11, color="#4b5563"))

    parts.append(line(x0, y1, x1, y1, "#111827", 2))
    parts.append(line(x0, y0, x0, y1, "#111827", 2))

    parts.append(polyline(train_points, "#2563eb", 2))
    parts.append(polyline(val_points, "#f59e0b", 2))
    parts.append(line(x0, test_y, x1, test_y, "#16a34a", 2, "8 6"))

    parts.append(text((x0 + x1) / 2, 36, title, size=20, color="#111827"))
    parts.append(text((x0 + x1) / 2, h - 30, "Epoca", size=14, color="#111827"))
    parts.append(text(24, (y0 + y1) / 2, y_label, size=14, anchor="middle", color="#111827"))

    lx, ly = x0 + 10, y0 + 10
    parts.append(line(lx, ly, lx + 26, ly, "#2563eb", 3))
    parts.append(text(lx + 35, ly + 4, "Treino", size=12, anchor="start"))
    parts.append(line(lx + 120, ly, lx + 146, ly, "#f59e0b", 3))
    parts.append(text(lx + 155, ly + 4, "Validacao", size=12, anchor="start"))
    parts.append(line(lx + 295, ly, lx + 321, ly, "#16a34a", 3, "8 6"))
    parts.append(text(lx + 330, ly + 4, f"Teste final ({test_value:.4f})", size=12, anchor="start"))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def make_confusion_svg(matrix: List[List[int]], class_names: List[str], out_path: Path) -> None:
    n = len(matrix)
    cell = 90
    ml, mt = 280, 110
    w = ml + n * cell + 80
    h = mt + n * cell + 130

    max_v = max(max(row) for row in matrix) if matrix else 1

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff" />')
    parts.append(text(w / 2, 40, "Matriz de Confusao", size=22, color="#111827"))
    parts.append(text(w / 2, 65, "Linhas: classe real | Colunas: classe predita", size=13, color="#4b5563"))

    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            intensity = int(235 - 180 * (v / max_v if max_v else 0))
            fill = f"rgb({intensity},{intensity},{255})"
            x = ml + j * cell
            y = mt + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#d1d5db" />')
            parts.append(text(x + cell / 2, y + cell / 2 + 5, str(v), size=18, color="#111827"))

    for i, name in enumerate(class_names):
        y = mt + i * cell + cell / 2 + 5
        x = ml - 12
        parts.append(text(x, y, name, size=12, anchor="end", color="#111827"))

    for j, name in enumerate(class_names):
        x = ml + j * cell + cell / 2
        y = mt - 14
        parts.append(text(x, y, name, size=12, anchor="middle", color="#111827"))

    parts.append(text(32, mt + (n * cell) / 2, "Classe real", size=14, anchor="start", color="#111827"))
    parts.append(text(ml + (n * cell) / 2, h - 40, "Classe predita", size=14, color="#111827"))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


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
    test_acc = float(test_metrics.get("accuracy", float("nan")))
    test_loss = float(test_metrics.get("loss", float("nan")))

    make_metric_svg(
        title="Evolucao da Acuracia",
        y_label="Acuracia",
        train_vals=history["accuracy"],
        val_vals=history["val_accuracy"],
        test_value=test_acc,
        out_path=out_dir / "plot_accuracy.svg",
    )
    make_metric_svg(
        title="Evolucao da Loss",
        y_label="Loss",
        train_vals=history["loss"],
        val_vals=history["val_loss"],
        test_value=test_loss,
        out_path=out_dir / "plot_loss.svg",
    )
    make_confusion_svg(cm, class_names, out_dir / "plot_confusion_matrix.svg")

    print(f"Graficos salvos em: {out_dir}")
    print("- plot_accuracy.svg")
    print("- plot_loss.svg")
    print("- plot_confusion_matrix.svg")


if __name__ == "__main__":
    main()
