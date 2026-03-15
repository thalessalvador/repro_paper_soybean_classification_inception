import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


def parse_args() -> Tuple[argparse.Namespace, List[str]]:
    """Interpreta argumentos do runner e preserva argumentos extras para o treino.

    Args:
        Nenhum argumento posicional direto. A funcao le argumentos de CLI.

    Returns:
        Tuple[argparse.Namespace, List[str]]:
            - configuracao do runner.
            - argumentos adicionais que serao repassados ao script principal.
    """
    parser = argparse.ArgumentParser(description="Executa os folds de validacao cruzada do experimento Inception.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Diretorio do dataset a ser passado para cada fold.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments_inception",
        help="Diretorio raiz onde o cv_run_TIMESTAMP sera criado.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Quantidade total de folds a executar.",
    )
    args, extras = parser.parse_known_args()
    return args, extras


def build_cv_run_dir(out_dir: Path) -> Path:
    """Cria o diretorio raiz do experimento completo de validacao cruzada.

    Args:
        out_dir: diretorio base escolhido pelo usuario.

    Returns:
        Path: diretorio final do tipo cv_run_TIMESTAMP.
    """
    run_dir = (out_dir / f"cv_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _strip_runner_managed_args(extras: List[str]) -> List[str]:
    """Remove argumentos extras que sao controlados diretamente pelo runner.

    Parametros de entrada:
        extras (List[str]): lista de argumentos adicionais recebidos da linha de
            comando e repassados ao script principal.

    Parametros de saida:
        List[str]: nova lista sem argumentos que o runner ja define
        explicitamente, evitando duplicidade e ambiguidade.
    """
    managed_flags = {"--tb-root-dir", "--out-dir", "--dataset-dir", "--fold-index", "--num-folds", "--experiment-mode"}
    filtered: List[str] = []
    i = 0
    while i < len(extras):
        token = extras[i]
        if token in managed_flags:
            i += 2
            continue
        filtered.append(token)
        i += 1
    return filtered


def run_fold_commands(args: argparse.Namespace, extras: List[str], cv_run_dir: Path) -> None:
    """Executa cada fold chamando o script principal em subprocessos.

    Args:
        args: configuracao do runner.
        extras: argumentos extras repassados ao script principal.
        cv_run_dir: diretorio raiz do experimento CV.

    Returns:
        None.

    Raises:
        subprocess.CalledProcessError: se qualquer fold falhar.
    """
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "train_modified_inception_repro.py"
    tb_root = cv_run_dir / "tf_tb_logs"
    forwarded_extras = _strip_runner_managed_args(extras)

    for fold_index in range(args.num_folds):
        command = [
            sys.executable,
            str(train_script),
            "--experiment-mode",
            "cv",
            "--fold-index",
            str(fold_index),
            "--num-folds",
            str(args.num_folds),
            "--dataset-dir",
            args.dataset_dir,
            "--out-dir",
            str(cv_run_dir),
            "--tb-root-dir",
            str(tb_root),
        ]
        command.extend(forwarded_extras)
        print(f"Executando fold {fold_index}: {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)


def run_aggregation(cv_run_dir: Path) -> None:
    """Executa o agregador ao final dos folds.

    Args:
        cv_run_dir: diretorio raiz do experimento CV.

    Returns:
        None.
    """
    script_dir = Path(__file__).resolve().parent
    aggregate_script = script_dir / "aggregate_inception_cv.py"
    command = [sys.executable, str(aggregate_script), "--cv-run-dir", str(cv_run_dir)]
    print(f"Agregando resultados: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    """Executa a validacao cruzada completa e agrega os resultados.

    Args:
        Nenhum argumento direto. A funcao usa parse_args().

    Returns:
        None.
    """
    args, extras = parse_args()
    cv_run_dir = build_cv_run_dir(Path(args.out_dir))
    run_fold_commands(args, extras, cv_run_dir)
    run_aggregation(cv_run_dir)
    print(f"Validacao cruzada concluida em: {cv_run_dir}")


if __name__ == "__main__":
    main()