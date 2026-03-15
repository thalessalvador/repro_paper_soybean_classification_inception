import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List


SUMMARY_FIELDS = [
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "weighted_precision",
    "weighted_recall",
    "weighted_f1",
]


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de agregacao de validacao cruzada.

    Args:
        Nenhum argumento posicional direto. A funcao le argumentos de CLI.

    Returns:
        argparse.Namespace: configuracao da agregacao.
    """
    parser = argparse.ArgumentParser(description="Agrega os resultados de folds do experimento Inception.")
    parser.add_argument(
        "--cv-run-dir",
        type=str,
        required=True,
        help="Diretorio raiz do experimento de validacao cruzada contendo fold_0 ... fold_4.",
    )
    return parser.parse_args()


def load_fold_metrics(cv_run_dir: Path) -> List[Dict]:
    """Carrega o arquivo metrics.json de cada fold encontrado.

    Args:
        cv_run_dir: diretorio raiz do experimento CV.

    Returns:
        List[Dict]: metricas de cada fold carregadas em memoria.

    Raises:
        FileNotFoundError: se nenhum metrics.json for encontrado.
    """
    metrics_list: List[Dict] = []
    for fold_dir in sorted(cv_run_dir.glob("fold_*")):
        metrics_path = fold_dir / "metrics.json"
        if metrics_path.exists():
            metrics_list.append(json.loads(metrics_path.read_text(encoding="utf-8")))

    if not metrics_list:
        raise FileNotFoundError(f"Nenhum metrics.json encontrado em {cv_run_dir}")
    return metrics_list


def build_summary(metrics_list: List[Dict]) -> Dict:
    """Calcula media, desvio padrao e formato de tabela para cada metrica.

    Args:
        metrics_list: metricas individuais dos folds.

    Returns:
        Dict: resumo agregado pronto para serializacao.
    """
    summary: Dict[str, object] = {
        "num_folds": len(metrics_list),
        "folds": [m.get("fold") for m in metrics_list],
    }

    for field in SUMMARY_FIELDS:
        values = [float(m[field]) for m in metrics_list]
        field_mean = mean(values)
        field_std = pstdev(values) if len(values) > 1 else 0.0
        summary[f"{field}_mean"] = field_mean
        summary[f"{field}_std"] = field_std
        summary[f"{field}_formatted"] = f"{field_mean:.3f} ({field_std:.3f})"

    return summary


def save_summary(summary: Dict, cv_run_dir: Path) -> None:
    """Salva o resumo agregado em JSON e CSV.

    Args:
        summary: dicionario de metricas agregadas.
        cv_run_dir: diretorio raiz do experimento CV.

    Returns:
        None.
    """
    json_path = cv_run_dir / "cv_summary.json"
    csv_path = cv_run_dir / "cv_summary.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "formatted"])
        for field in SUMMARY_FIELDS:
            writer.writerow(
                [
                    field,
                    summary[f"{field}_mean"],
                    summary[f"{field}_std"],
                    summary[f"{field}_formatted"],
                ]
            )


def main() -> None:
    """Executa a agregacao final das metricas de validacao cruzada.

    Args:
        Nenhum argumento direto. A funcao usa parse_args().

    Returns:
        None.
    """
    args = parse_args()
    cv_run_dir = Path(args.cv_run_dir).resolve()
    metrics_list = load_fold_metrics(cv_run_dir)
    summary = build_summary(metrics_list)
    save_summary(summary, cv_run_dir)
    print(f"Resumo agregado salvo em: {cv_run_dir}")


if __name__ == "__main__":
    main()