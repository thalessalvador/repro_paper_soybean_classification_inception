import argparse
from io import StringIO

import tensorflow as tf

from train_modified_inception_repro import build_model


def parse_args() -> argparse.Namespace:
    """Interpreta os argumentos de linha de comando do utilitario de visualizacao.

    Parametros de entrada:
        Nenhum parametro posicional direto. A funcao le os argumentos informados
        na linha de comando, incluindo tamanho da imagem, quantidade de classes,
        unidades da camada densa e caminho opcional para salvar o grafo.

    Parametros de saida:
        argparse.Namespace: objeto contendo os argumentos normalizados para a
        execucao do script.
    """
    parser = argparse.ArgumentParser(description="Mostra a arquitetura da rede no terminal.")
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--dense-units", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument(
        "--save-plot",
        type=str,
        default="",
        help="Caminho opcional para salvar imagem do grafo via keras.utils.plot_model.",
    )
    return parser.parse_args()


def print_ascii_overview(model: tf.keras.Model) -> None:
    """Exibe no terminal um resumo textual curto da arquitetura da rede.

    Parametros de entrada:
        model (tf.keras.Model): modelo Keras ja construido cuja arquitetura sera
            descrita de forma resumida no terminal.

    Parametros de saida:
        None: a funcao apenas escreve informacoes no terminal e nao retorna valor.
    """
    print("\n=== Arquitetura (visao rapida) ===")
    print("Input")
    print("  -> InceptionV3 backbone (include_top=False)")
    print("  -> AveragePooling2D(7x7)")
    print("  -> Flatten")
    print("  -> Dense(relu)")
    print("  -> Dropout(0.5)")
    print("  -> Dense(softmax)")
    print(f"\nNome do modelo: {model.name}")


def print_summary(model: tf.keras.Model) -> None:
    """Captura e imprime no terminal a saida completa de ``model.summary()``.

    Parametros de entrada:
        model (tf.keras.Model): modelo Keras que sera detalhado com a listagem
            completa de camadas, formatos de tensor e quantidade de parametros.

    Parametros de saida:
        None: a funcao apenas envia a representacao textual do modelo para o
        terminal e nao retorna valor.
    """
    print("\n=== model.summary() ===")
    buf = StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    print(buf.getvalue())


def try_save_plot(model: tf.keras.Model, out_path: str) -> None:
    """Tenta salvar uma imagem do grafo do modelo usando ``keras.utils.plot_model``.

    Parametros de entrada:
        model (tf.keras.Model): modelo Keras que tera o grafo exportado.
        out_path (str): caminho de destino da imagem. Quando vazio, nenhuma
            exportacao e realizada.

    Parametros de saida:
        None: a funcao salva a imagem quando possivel e escreve mensagens de
        status no terminal, sem retornar valor.
    """
    if not out_path:
        return

    try:
        tf.keras.utils.plot_model(
            model,
            to_file=out_path,
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            expand_nested=False,
            dpi=120,
        )
        print(f"[OK] Grafo salvo em: {out_path}")
    except Exception as exc:
        print("[AVISO] Nao foi possivel gerar imagem do grafo com plot_model.")
        print(f"Motivo: {exc}")
        print("Dica: instale 'pydot' e Graphviz (comando 'dot').")


def main() -> None:
    """Coordena a construcao do modelo e a exibicao de suas representacoes.

    Parametros de entrada:
        Nenhum parametro direto. A funcao utiliza os argumentos obtidos por
        ``parse_args()`` para configurar a construcao e a visualizacao do modelo.

    Parametros de saida:
        None: a funcao executa o fluxo principal do script e nao retorna valor.
    """
    args = parse_args()
    model, _ = build_model(
        num_classes=args.num_classes,
        dense_units=args.dense_units,
        image_size=args.image_size,
        base_weights=None,
    )
    print_ascii_overview(model)
    print_summary(model)
    try_save_plot(model, args.save_plot)


if __name__ == "__main__":
    main()
