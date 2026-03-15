import argparse
from io import StringIO

import tensorflow as tf

from train_modified_inception_repro import build_model


def parse_args() -> argparse.Namespace:
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
    print("\n=== model.summary() ===")
    buf = StringIO()
    model.summary(print_fn=lambda x: buf.write(x + "\n"))
    print(buf.getvalue())


def try_save_plot(model: tf.keras.Model, out_path: str) -> None:
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
