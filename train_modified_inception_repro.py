import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Modo seguro para GPUs Blackwell (RTX 50xx) em WSL.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0 --tf_xla_enable_xla_devices=false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_enable_triton_gemm=false")
os.environ.setdefault("TF_DISABLE_MLIR_BRIDGE", "1")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras import mixed_precision


"""Pipeline de reproducao aproximada para classificacao de sementes de soja.

Este modulo implementa o fluxo completo de treino e avaliacao de um modelo
InceptionV3 modificado, inspirado no artigo "Enhancing soybean classification
with modified inception model". O objetivo e fornecer uma base auditavel para
reproducao de resultados, registrando split, historico e metricas finais.
"""


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Table 1 from the paper:
# Enhancing soybean classification with modified inception model (2024)
TARGET_SPLITS = {
    "Broken soybeans": {"train": 800, "val": 101, "test": 101},
    "Immature soybeans": {"train": 901, "val": 112, "test": 112},
    "Intact soybeans": {"train": 961, "val": 120, "test": 120},
    "Skin-damaged soybeans": {"train": 901, "val": 113, "test": 113},
    "Spotted soybeans": {"train": 846, "val": 106, "test": 106},
}

CLASS_BAR_ORDER = [
    ("Broken soybeans", "Broken"),
    ("Immature soybeans", "Immature"),
    ("Intact soybeans", "Intact"),
    ("Skin-damaged soybeans", "Skin-damaged"),
    ("Spotted soybeans", "Spotted"),
]


@dataclass
class SplitData:
    train_paths: List[str]
    train_labels: List[int]
    val_paths: List[str]
    val_labels: List[int]
    test_paths: List[str]
    test_labels: List[int]
    class_names: List[str]


@dataclass
class DatasetItem:
    """Representa um item individual do inventario do dataset.

    Args:
        path: caminho absoluto ou relativo da imagem.
        class_name: nome textual da classe.
        class_idx: indice numerico da classe.

    Returns:
        Nenhum valor. A dataclass apenas encapsula os campos do inventario.
    """

    path: str
    class_name: str
    class_idx: int


@dataclass
class AugmentationConfig:
    mode: str
    flip_left_right: bool
    flip_up_down: bool
    brightness_max_delta: float
    contrast_lower: float
    contrast_upper: float
    saturation_lower: float
    saturation_upper: float
    hue_max_delta: float
    jpeg_quality_min: int
    jpeg_quality_max: int
    sharpness_min: float
    sharpness_max: float


def parse_args() -> argparse.Namespace:
    """Define e interpreta argumentos de linha de comando.

    A funcao concentra todas as opcoes que controlam dataset, hiperparametros,
    estrategia de fine-tuning e diretorio de saida.

    Args:
        Nenhum argumento posicional direto. A funcao le argumentos de CLI.

    Returns:
        argparse.Namespace: objeto com todos os parametros configurados para
        execucao do pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Reproducao aproximada do paper de InceptionV3 modificado para soja."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"..\dataset_kaggle_soja",
        help="Diretorio do dataset (somente leitura).",
    )
    parser.add_argument(
        "--experiment-mode",
        type=str,
        default="paper",
        choices=["paper", "cv"],
        help="Modo experimental: split original do paper ou validacao cruzada.",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=None,
        help="Indice do fold a executar quando --experiment-mode cv.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Quantidade de folds para validacao cruzada estratificada.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed para split deterministico.")
    parser.add_argument("--image-size", type=int, default=299, help="Tamanho da imagem (paper: 299).")
    parser.add_argument("--batch-size", type=int, default=32, help="Assuncao (paper nao informa).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate base (paper: 0.001).")
    parser.add_argument(
        "--freeze-epochs",
        type=int,
        default=20,
        help="Epocas com base congelada (paper: ate iteracao 20).",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=125,
        help="Epocas totais (assuncao alinhada com texto do paper).",
    )
    parser.add_argument(
        "--fine-tune-lr-multiplier",
        type=float,
        default=0.1,
        help="Multiplicador de LR no fine-tuning.",
    )
    parser.add_argument(
        "--dense-units",
        type=int,
        default=256,
        help="Unidades da Dense adicional (assuncao, paper nao informa).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="runs_modified_inception",
        help="Diretorio de saida para artefatos.",
    )
    parser.add_argument(
        "--tb-root-dir",
        type=str,
        default="tf_tb_logs",
        help="Diretorio raiz para logs do TensorBoard.",
    )
    parser.add_argument(
        "--disable-tensorboard",
        action="store_true",
        help="Desativa o callback TensorBoard para evitar problemas de path no Windows.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=10,
        help="Paciencia do EarlyStopping monitorando val_accuracy.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Dispositivo de execucao: auto, gpu ou cpu.",
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Ativa mixed precision (float16) quando houver GPU.",
    )
    parser.add_argument(
        "--jit-compile",
        action="store_true",
        help="Ativa XLA JIT na compilacao do modelo.",
    )
    parser.add_argument(
        "--num-parallel-calls",
        type=int,
        default=-1,
        help="Paralelismo do map no tf.data (-1 usa AUTOTUNE).",
    )
    parser.add_argument(
        "--prefetch-size",
        type=int,
        default=-1,
        help="Prefetch do tf.data (-1 usa AUTOTUNE).",
    )
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Desativa data augmentation no treino (igual ao comportamento anterior).",
    )
    parser.add_argument(
        "--augmentation-mode",
        type=str,
        default="expanded",
        choices=["on_the_fly", "expanded"],
        help=(
            "Modo de augmentation no treino: "
            "'on_the_fly' aplica 1:1; 'expanded' gera multiplas versoes por imagem."
        ),
    )
    parser.add_argument("--aug-brightness", type=float, default=0.15, help="Delta maximo de brilho em [0, 1].")
    parser.add_argument("--aug-contrast-min", type=float, default=0.8, help="Limite inferior de contraste.")
    parser.add_argument("--aug-contrast-max", type=float, default=1.2, help="Limite superior de contraste.")
    parser.add_argument("--aug-saturation-min", type=float, default=0.8, help="Limite inferior de saturacao.")
    parser.add_argument("--aug-saturation-max", type=float, default=1.2, help="Limite superior de saturacao.")
    parser.add_argument("--aug-hue", type=float, default=0.03, help="Delta maximo de matiz em [0, 0.5].")
    parser.add_argument("--aug-jpeg-quality-min", type=int, default=80, help="Qualidade JPEG minima [0, 100].")
    parser.add_argument("--aug-jpeg-quality-max", type=int, default=100, help="Qualidade JPEG maxima [0, 100].")
    parser.add_argument("--aug-sharpness-min", type=float, default=0.0, help="Fator minimo de nitidez adicional.")
    parser.add_argument("--aug-sharpness-max", type=float, default=0.3, help="Fator maximo de nitidez adicional.")
    parser.add_argument(
        "--aug-disable-flip-up-down",
        action="store_true",
        help="Desativa flip vertical no augmentation.",
    )
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    """Configura seeds globais para reproducibilidade parcial.

    Args:
        seed: valor inteiro usado para inicializar geradores pseudoaleatorios
            do NumPy e TensorFlow.

    Returns:
        None.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)


def list_images(path: Path) -> List[Path]:
    """Lista imagens validas de forma recursiva dentro de um diretorio.

    Args:
        path: diretorio raiz de busca das imagens.

    Returns:
        List[Path]: caminhos absolutos/relativos encontrados, ordenados
        alfabeticamente para manter determinismo.
    """
    files = [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    files.sort()
    return files


def build_dataset_inventory(dataset_dir: Path) -> Tuple[List[DatasetItem], List[str]]:
    """Constroi o inventario completo do dataset sem aplicar estrategia de split.

    Args:
        dataset_dir: diretorio raiz contendo uma subpasta por classe.

    Returns:
        Tuple[List[DatasetItem], List[str]]:
            - lista completa de itens do dataset.
            - nomes de classes na ordem dos indices.

    Raises:
        FileNotFoundError: se o diretorio raiz nao existir.
        ValueError: se nenhuma classe com imagens for encontrada.
    """
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_dir}")

    discovered = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    class_names = [name for name in TARGET_SPLITS.keys() if name in discovered]
    class_names.extend([name for name in discovered if name not in class_names])
    if not class_names:
        raise ValueError(f"Nenhuma subpasta de classe encontrada em {dataset_dir}")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    inventory: List[DatasetItem] = []
    for class_name in class_names:
        class_dir = dataset_dir / class_name
        for image_path in list_images(class_dir):
            inventory.append(
                DatasetItem(
                    path=str(image_path),
                    class_name=class_name,
                    class_idx=class_to_idx[class_name],
                )
            )
    return inventory, class_names


def _items_to_split_data(
    train_items: List[DatasetItem],
    val_items: List[DatasetItem],
    test_items: List[DatasetItem],
    class_names: List[str],
) -> SplitData:
    """Converte listas de itens inventariados para a estrutura SplitData.

    Args:
        train_items: itens alocados ao treino.
        val_items: itens alocados a validacao.
        test_items: itens alocados a teste.
        class_names: nomes de classes na ordem dos indices.

    Returns:
        SplitData: estrutura consolidada contendo paths, labels e classes.
    """
    return SplitData(
        train_paths=[item.path for item in train_items],
        train_labels=[item.class_idx for item in train_items],
        val_paths=[item.path for item in val_items],
        val_labels=[item.class_idx for item in val_items],
        test_paths=[item.path for item in test_items],
        test_labels=[item.class_idx for item in test_items],
        class_names=class_names,
    )


def generate_paper_split(dataset_inventory: List[DatasetItem], seed: int) -> SplitData:
    """Reproduz o split deterministico descrito no paper original.

    Args:
        dataset_inventory: inventario completo do dataset.
        seed: seed usada para embaralhamento deterministico.

    Returns:
        SplitData: subconjuntos train/val/test no formato original do paper.

    Raises:
        ValueError: se alguma classe do inventario nao estiver mapeada em TARGET_SPLITS
            ou se nao houver imagens suficientes para reproduzir os totais do paper.
    """
    rng = np.random.default_rng(seed)
    grouped: Dict[str, List[DatasetItem]] = defaultdict(list)
    for item in dataset_inventory:
        grouped[item.class_name].append(item)

    class_names = sorted(grouped.keys(), key=lambda name: grouped[name][0].class_idx)
    train_items: List[DatasetItem] = []
    val_items: List[DatasetItem] = []
    test_items: List[DatasetItem] = []

    for class_name in class_names:
        if class_name not in TARGET_SPLITS:
            raise ValueError(f"Classe sem configuracao paper: {class_name}")
        counts = TARGET_SPLITS[class_name]
        items = list(grouped[class_name])
        expected_total = counts["train"] + counts["val"] + counts["test"]
        if len(items) < expected_total:
            raise ValueError(
                f"Classe {class_name} tem {len(items)} imagens, mas precisa de pelo menos {expected_total}."
            )

        perm = rng.permutation(len(items))
        shuffled = [items[i] for i in perm]
        n_train = counts["train"]
        n_val = counts["val"]
        n_test = counts["test"]

        train_items.extend(shuffled[:n_train])
        val_items.extend(shuffled[n_train : n_train + n_val])
        test_items.extend(shuffled[n_train + n_val : n_train + n_val + n_test])

    return _items_to_split_data(train_items, val_items, test_items, class_names)


def generate_cv_split(
    dataset_inventory: List[DatasetItem],
    fold_index: int,
    num_folds: int,
    seed: int,
    val_ratio: float = 1.0 / 9.0,
) -> SplitData:
    """Gera um fold estratificado de validacao cruzada.

    Regras:
    - teste = fold selecionado
    - train/val = itens restantes
    - val = subconjunto estratificado retirado apenas de train_val

    Args:
        dataset_inventory: inventario completo do dataset.
        fold_index: indice do fold de teste a executar.
        num_folds: quantidade total de folds.
        seed: seed para determinismo.
        val_ratio: proporcao de validacao retirada do conjunto train_val.

    Returns:
        SplitData: estrutura com train/val/test do fold solicitado.

    Raises:
        ValueError: se o fold solicitado for invalido.
    """
    if fold_index < 0 or fold_index >= num_folds:
        raise ValueError(f"fold_index invalido: {fold_index}. Esperado entre 0 e {num_folds - 1}.")

    grouped: Dict[int, List[DatasetItem]] = defaultdict(list)
    for item in dataset_inventory:
        grouped[item.class_idx].append(item)

    class_names = [grouped[idx][0].class_name for idx in sorted(grouped.keys())]
    train_items: List[DatasetItem] = []
    val_items: List[DatasetItem] = []
    test_items: List[DatasetItem] = []

    for class_idx in sorted(grouped.keys()):
        class_items = list(grouped[class_idx])
        rng = np.random.default_rng(seed + class_idx)
        perm = rng.permutation(len(class_items))
        shuffled = [class_items[i] for i in perm]
        buckets = [list(bucket) for bucket in np.array_split(np.array(shuffled, dtype=object), num_folds)]

        class_test = [item for item in buckets[fold_index]]
        class_train_val: List[DatasetItem] = []
        for idx, bucket in enumerate(buckets):
            if idx != fold_index:
                class_train_val.extend(bucket)

        if len(class_train_val) > 1:
            val_rng = np.random.default_rng(seed + (fold_index + 1) * 1000 + class_idx)
            val_perm = val_rng.permutation(len(class_train_val))
            shuffled_train_val = [class_train_val[i] for i in val_perm]
            n_val = int(round(len(shuffled_train_val) * val_ratio))
            n_val = min(max(n_val, 1), len(shuffled_train_val) - 1)
        else:
            shuffled_train_val = class_train_val
            n_val = 0

        class_val = shuffled_train_val[:n_val]
        class_train = shuffled_train_val[n_val:]

        train_items.extend(class_train)
        val_items.extend(class_val)
        test_items.extend(class_test)

    return _items_to_split_data(train_items, val_items, test_items, class_names)


def create_split(dataset_dir: Path, seed: int) -> SplitData:
    """Mantem compatibilidade com o comportamento historico do script.

    Args:
        dataset_dir: diretorio raiz contendo uma subpasta por classe.
        seed: seed do embaralhamento deterministico.

    Returns:
        SplitData: split equivalente ao modo `paper`.
    """
    inventory, _class_names = build_dataset_inventory(dataset_dir)
    return generate_paper_split(inventory, seed)


def _load_image(path: tf.Tensor, label: tf.Tensor, image_size: int, num_classes: int):
    """Carrega e redimensiona imagem, retornando label one-hot.

    Args:
        path: tensor string com caminho do arquivo de imagem.
        label: tensor inteiro com indice da classe.
        image_size: tamanho alvo (altura e largura) da imagem.
        num_classes: quantidade total de classes para one-hot encoding.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - imagem em float32 no range [0, 255].
            - label one-hot com dimensao num_classes.
    """
    bytes_ = tf.io.read_file(path)
    image = tf.io.decode_image(bytes_, channels=3, expand_animations=False)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, depth=num_classes)
    return image, label


def _random_sharpen(image: tf.Tensor, min_factor: float, max_factor: float) -> tf.Tensor:
    """Aplica nitidez aleatoria por unsharp mask simplificado."""
    factor = tf.random.uniform([], minval=min_factor, maxval=max_factor, dtype=tf.float32)
    kernel = tf.constant([[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]], dtype=tf.float32)
    kernel = tf.reshape(kernel, [3, 3, 1, 1])
    channels = tf.shape(image)[-1]
    kernel = tf.tile(kernel, [1, 1, channels, 1])
    img4 = tf.expand_dims(image, axis=0)
    sharpened = tf.nn.depthwise_conv2d(img4, kernel, strides=[1, 1, 1, 1], padding="SAME")
    sharpened = tf.squeeze(sharpened, axis=0)
    out = image * (1.0 - factor) + sharpened * factor
    return tf.clip_by_value(out, 0.0, 1.0)


def _augment_image(image: tf.Tensor, label: tf.Tensor, cfg: AugmentationConfig):
    """Aplica augmentations visuais estocasticos em imagem de treino.

    Args:
        image: imagem float32 no range [0, 255].
        label: label one-hot.
        cfg: configuracao de augmentation.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - imagem augmentada no range [0, 255].
            - label original.
    """
    x = image / 255.0
    if cfg.flip_left_right:
        x = tf.image.random_flip_left_right(x)
    if cfg.flip_up_down:
        x = tf.image.random_flip_up_down(x)
    x = tf.image.random_brightness(x, max_delta=cfg.brightness_max_delta)
    x = tf.image.random_contrast(x, lower=cfg.contrast_lower, upper=cfg.contrast_upper)
    x = tf.image.random_saturation(x, lower=cfg.saturation_lower, upper=cfg.saturation_upper)
    x = tf.image.random_hue(x, max_delta=cfg.hue_max_delta)
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = _random_sharpen(x, min_factor=cfg.sharpness_min, max_factor=cfg.sharpness_max)
    x = tf.image.random_jpeg_quality(
        tf.cast(x * 255.0, tf.uint8),
        min_jpeg_quality=cfg.jpeg_quality_min,
        max_jpeg_quality=cfg.jpeg_quality_max,
    )
    x = tf.cast(x, tf.float32)
    return x, label


def _apply_brightness_01(x01: tf.Tensor, cfg: AugmentationConfig) -> tf.Tensor:
    return tf.clip_by_value(tf.image.random_brightness(x01, max_delta=cfg.brightness_max_delta), 0.0, 1.0)


def _apply_contrast_01(x01: tf.Tensor, cfg: AugmentationConfig) -> tf.Tensor:
    return tf.clip_by_value(tf.image.random_contrast(x01, lower=cfg.contrast_lower, upper=cfg.contrast_upper), 0.0, 1.0)


def _apply_saturation_01(x01: tf.Tensor, cfg: AugmentationConfig) -> tf.Tensor:
    return tf.clip_by_value(
        tf.image.random_saturation(x01, lower=cfg.saturation_lower, upper=cfg.saturation_upper),
        0.0,
        1.0,
    )


def _apply_hue_01(x01: tf.Tensor, cfg: AugmentationConfig) -> tf.Tensor:
    return tf.clip_by_value(tf.image.random_hue(x01, max_delta=cfg.hue_max_delta), 0.0, 1.0)


def _apply_jpeg_01(x01: tf.Tensor, cfg: AugmentationConfig) -> tf.Tensor:
    x_u8 = tf.cast(tf.clip_by_value(x01, 0.0, 1.0) * 255.0, tf.uint8)
    x_u8 = tf.image.random_jpeg_quality(
        x_u8,
        min_jpeg_quality=cfg.jpeg_quality_min,
        max_jpeg_quality=cfg.jpeg_quality_max,
    )
    return tf.cast(x_u8, tf.float32) / 255.0


def _expand_augmentations(image: tf.Tensor, label: tf.Tensor, cfg: AugmentationConfig):
    """Gera multiplas versoes por imagem para aumentar steps por epoca.

    Args:
        image: imagem float32 no range [0, 255].
        label: label one-hot.
        cfg: configuracao de augmentation.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]:
            - tensor [N, H, W, 3] com N variantes da imagem.
            - tensor [N, C] com label replicado para cada variante.
    """
    x01 = tf.clip_by_value(image / 255.0, 0.0, 1.0)
    variants = [x01]

    if cfg.flip_left_right:
        variants.append(tf.image.flip_left_right(x01))
    if cfg.flip_up_down:
        variants.append(tf.image.flip_up_down(x01))

    variants.append(_apply_brightness_01(x01, cfg))
    variants.append(_apply_contrast_01(x01, cfg))
    variants.append(_apply_saturation_01(x01, cfg))
    variants.append(_apply_hue_01(x01, cfg))
    variants.append(_apply_jpeg_01(x01, cfg))
    variants.append(_random_sharpen(x01, min_factor=cfg.sharpness_min, max_factor=cfg.sharpness_max))

    combo = x01
    if cfg.flip_left_right:
        combo = tf.image.flip_left_right(combo)
    if cfg.flip_up_down:
        combo = tf.image.flip_up_down(combo)
    combo = _apply_brightness_01(combo, cfg)
    combo = _apply_contrast_01(combo, cfg)
    combo = _apply_saturation_01(combo, cfg)
    combo = _apply_hue_01(combo, cfg)
    combo = _random_sharpen(combo, min_factor=cfg.sharpness_min, max_factor=cfg.sharpness_max)
    combo = _apply_jpeg_01(combo, cfg)
    variants.append(combo)

    images = tf.stack([tf.cast(v * 255.0, tf.float32) for v in variants], axis=0)
    labels = tf.repeat(tf.expand_dims(label, axis=0), repeats=tf.shape(images)[0], axis=0)
    return images, labels


def _preprocess_for_inception(image: tf.Tensor, label: tf.Tensor):
    """Aplica preprocess_input da InceptionV3 em imagem float32."""
    return preprocess_input(image), label


def make_dataset(
    paths: List[str],
    labels: List[int],
    image_size: int,
    batch_size: int,
    training: bool,
    num_parallel_calls: int,
    prefetch_size: int,
    augmentation_config: Optional[AugmentationConfig],
):
    """Constroi tf.data.Dataset com leitura, preprocessamento e batching.

    Args:
        paths: lista de caminhos de imagem.
        labels: lista de labels inteiros alinhada com paths.
        image_size: tamanho final da imagem para entrada da rede.
        batch_size: tamanho do lote usado no treinamento/avaliacao.
        training: se True, aplica shuffle a cada epoca.
        num_parallel_calls: numero de chamadas paralelas no map.
            Se -1, usa tf.data.AUTOTUNE.
        prefetch_size: tamanho do buffer de prefetch.
            Se -1, usa tf.data.AUTOTUNE.
        augmentation_config: configuracao de augmentation para treino.
            Se None, nenhum augmentation e aplicado.

    Returns:
        tf.data.Dataset: dataset pronto para consumir no Keras fit/evaluate.
    """
    num_classes = len(TARGET_SPLITS)
    autotune = tf.data.AUTOTUNE
    num_parallel = autotune if num_parallel_calls == -1 else num_parallel_calls
    prefetch = autotune if prefetch_size == -1 else prefetch_size

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: _load_image(p, y, image_size, num_classes),
        num_parallel_calls=num_parallel,
    )
    if training and augmentation_config is not None:
        if augmentation_config.mode == "expanded":
            ds = ds.map(
                lambda x, y: _expand_augmentations(x, y, augmentation_config),
                num_parallel_calls=num_parallel,
            )
            ds = ds.unbatch()
        else:
            ds = ds.map(
                lambda x, y: _augment_image(x, y, augmentation_config),
                num_parallel_calls=num_parallel,
            )
    ds = ds.map(_preprocess_for_inception, num_parallel_calls=num_parallel)
    ds = ds.batch(batch_size).prefetch(prefetch)
    return ds


def build_model(
    num_classes: int,
    dense_units: int,
    image_size: int,
    base_weights: str | None = "imagenet",
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    """Monta o modelo InceptionV3 modificado com cabeca customizada.

    Estrutura principal:
    - Base: InceptionV3 (sem top) pretreinada em ImageNet.
    - Cabeca adicional: AveragePooling2D(7x7), Flatten, Dense(ReLU),
      Dropout(0.5), Dense(Softmax).

    Args:
        num_classes: numero de classes de saida.
        dense_units: numero de neuronios na camada Dense adicional.
        image_size: tamanho de entrada esperado pelo backbone.
        base_weights: pesos do backbone InceptionV3.
            Use "imagenet" para treino normal ou None para visualizacao/testes.

    Returns:
        Tuple[tf.keras.Model, tf.keras.Model]:
            - modelo completo pronto para compilacao.
            - backbone InceptionV3 para congelamento/descongelamento.
    """
    base = InceptionV3(include_top=False, weights=base_weights, input_shape=(image_size, image_size, 3))
    base.trainable = False
    x = base.output
    x = layers.AveragePooling2D(pool_size=(7, 7), name="avg_pool_7x7")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_assumed")(x)
    x = layers.Dropout(0.5, name="dropout_0_5")(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32", name="predictions")(x)
    model = tf.keras.Model(inputs=base.input, outputs=outputs, name="modified_inceptionv3_repro")
    return model, base


def compile_model(model: tf.keras.Model, lr: float, jit_compile: bool) -> None:
    """Compila o modelo com configuracao padrao de otimizacao e metricas.

    Args:
        model: instancia de modelo Keras a ser compilada.
        lr: learning rate do otimizador Adam.
        jit_compile: habilita/desabilita compilacao JIT (XLA).

    Returns:
        None.
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
        jit_compile=jit_compile,
        run_eagerly=False,
    )


def configure_runtime(device: str, use_mixed_precision: bool) -> List[tf.config.PhysicalDevice]:
    """Configura device policy e mixed precision.

    Args:
        device: politica de dispositivo ('auto', 'gpu' ou 'cpu').
        use_mixed_precision: se True, ativa mixed_float16 quando houver GPU.

    Returns:
        Lista de GPUs visiveis apos a configuracao.

    Raises:
        RuntimeError: se --device gpu for solicitado sem GPU disponivel.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if device == "cpu":
        tf.config.set_visible_devices([], "GPU")
        mixed_precision.set_global_policy("float32")
        return []

    if device == "gpu" and not gpus:
        raise RuntimeError("Nenhuma GPU detectada, mas --device gpu foi solicitado.")

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if use_mixed_precision and gpus:
        mixed_precision.set_global_policy("mixed_float16")
    else:
        mixed_precision.set_global_policy("float32")

    tf.config.optimizer.set_jit(False)
    tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    return gpus


def build_augmentation_config(args: argparse.Namespace) -> Optional[AugmentationConfig]:
    """Constroi configuracao de data augmentation a partir dos argumentos.

    Args:
        args: argumentos de linha de comando.

    Returns:
        Optional[AugmentationConfig]:
            - configuracao de augmentation quando habilitado.
            - None quando --disable-augmentation for usado.
    """
    if args.disable_augmentation:
        return None

    return AugmentationConfig(
        mode=args.augmentation_mode,
        flip_left_right=True,
        flip_up_down=not args.aug_disable_flip_up_down,
        brightness_max_delta=args.aug_brightness,
        contrast_lower=args.aug_contrast_min,
        contrast_upper=args.aug_contrast_max,
        saturation_lower=args.aug_saturation_min,
        saturation_upper=args.aug_saturation_max,
        hue_max_delta=args.aug_hue,
        jpeg_quality_min=args.aug_jpeg_quality_min,
        jpeg_quality_max=args.aug_jpeg_quality_max,
        sharpness_min=args.aug_sharpness_min,
        sharpness_max=args.aug_sharpness_max,
    )


def save_split_manifest(split: SplitData, run_dir: Path) -> None:
    """Salva o manifesto padrao do split para o modo paper.

    Args:
        split: estrutura contendo caminhos e labels dos subconjuntos.
        run_dir: diretorio da execucao onde o CSV sera gravado.

    Returns:
        None.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    out_csv = run_dir / "split_manifest.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_name", "class_idx", "path"])
        idx_to_class = {i: n for i, n in enumerate(split.class_names)}

        for p, y in zip(split.train_paths, split.train_labels):
            writer.writerow(["train", idx_to_class[y], y, p])
        for p, y in zip(split.val_paths, split.val_labels):
            writer.writerow(["val", idx_to_class[y], y, p])
        for p, y in zip(split.test_paths, split.test_labels):
            writer.writerow(["test", idx_to_class[y], y, p])



def save_fold_manifest(split: SplitData, run_dir: Path, fold_index: int) -> None:
    """Salva o manifesto completo de um fold de validacao cruzada.

    Args:
        split: estrutura contendo caminhos e labels dos subconjuntos.
        run_dir: diretorio da execucao do fold.
        fold_index: indice do fold executado.

    Returns:
        None.
    """
    out_csv = run_dir / "fold_manifest.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "subset", "path", "class_name", "class_idx"])
        idx_to_class = {i: n for i, n in enumerate(split.class_names)}

        for p, y in zip(split.train_paths, split.train_labels):
            writer.writerow([fold_index, "train", p, idx_to_class[y], y])
        for p, y in zip(split.val_paths, split.val_labels):
            writer.writerow([fold_index, "val", p, idx_to_class[y], y])
        for p, y in zip(split.test_paths, split.test_labels):
            writer.writerow([fold_index, "test", p, idx_to_class[y], y])



def save_predictions_csv(
    test_paths: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_probs: np.ndarray,
    class_names: List[str],
    run_dir: Path,
    fold_index: Optional[int],
) -> None:
    """Salva predicoes por amostra do conjunto de teste.

    Args:
        test_paths: caminhos das imagens de teste na mesma ordem das predicoes.
        y_true: indices reais por amostra.
        y_pred: indices previstos por amostra.
        pred_probs: probabilidades previstas por amostra.
        class_names: nomes das classes na ordem dos indices.
        run_dir: diretorio onde o CSV sera salvo.
        fold_index: indice do fold ou None para modo paper.

    Returns:
        None.
    """
    out_csv = run_dir / "predictions.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "fold",
                "path",
                "true_class_idx",
                "true_class_name",
                "pred_class_idx",
                "pred_class_name",
                "correct",
                "top1_confidence",
            ]
        )
        fold_value = "paper" if fold_index is None else fold_index
        for path_value, true_idx, pred_idx, probs in zip(test_paths, y_true.tolist(), y_pred.tolist(), pred_probs.tolist()):
            top1_conf = float(max(probs)) if probs else 0.0
            writer.writerow(
                [
                    fold_value,
                    path_value,
                    true_idx,
                    class_names[true_idx],
                    pred_idx,
                    class_names[pred_idx],
                    bool(true_idx == pred_idx),
                    top1_conf,
                ]
            )



def confusion_and_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> Dict:
    """Calcula matriz de confusao e metricas agregadas sem sklearn.

    Args:
        y_true: vetor com labels reais (inteiros).
        y_pred: vetor com labels previstos (inteiros).
        class_names: nomes das classes na ordem dos indices.

    Returns:
        Dict: dicionario serializavel com metricas detalhadas e agregadas.
    """
    n = len(class_names)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class_metrics = {}
    precision_vals: List[float] = []
    recall_vals: List[float] = []
    f1_vals: List[float] = []
    support_vals: List[int] = []

    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = int(cm.sum() - (tp + fp + fn))
        support = int(cm[i, :].sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        accuracy = float((tp + tn) / cm.sum()) if cm.sum() else 0.0

        per_class_metrics[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "support": support,
        }
        precision_vals.append(precision)
        recall_vals.append(recall)
        f1_vals.append(f1)
        support_vals.append(support)

    total = int(cm.sum())
    tp_total = int(np.trace(cm))
    fp_total = total - tp_total
    fn_total = total - tp_total

    accuracy = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    macro_precision = float(np.mean(precision_vals)) if precision_vals else 0.0
    macro_recall = float(np.mean(recall_vals)) if recall_vals else 0.0
    macro_f1 = float(np.mean(f1_vals)) if f1_vals else 0.0

    micro_precision = float(tp_total / (tp_total + fp_total)) if (tp_total + fp_total) else 0.0
    micro_recall = float(tp_total / (tp_total + fn_total)) if (tp_total + fn_total) else 0.0
    micro_f1 = (
        float(2 * micro_precision * micro_recall / (micro_precision + micro_recall))
        if (micro_precision + micro_recall)
        else 0.0
    )

    weights = np.array(support_vals, dtype=np.float64)
    if weights.sum() == 0:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0
    else:
        weighted_precision = float(np.average(precision_vals, weights=weights))
        weighted_recall = float(np.average(recall_vals, weights=weights))
        weighted_f1 = float(np.average(f1_vals, weights=weights))

    support_per_class = {name: int(per_class_metrics[name]["support"]) for name in class_names}
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": macro_f1,
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": micro_f1,
        },
        "weighted": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1": weighted_f1,
        },
        "per_class": per_class_metrics,
        "per_class_metrics": per_class_metrics,
        "support_per_class": support_per_class,
        "confusion_matrix": cm.tolist(),
    }



def save_history(histories: List[tf.keras.callbacks.History], run_dir: Path) -> None:
    """Consolida e salva historico de treino em CSV unico por epoca.

    Quando ha treino em fases (congelado + fine-tuning), esta funcao concatena
    os historicos para facilitar auditoria e graficos posteriores.

    Args:
        histories: lista de objetos History retornados por model.fit.
        run_dir: diretorio de execucao onde o arquivo history.csv sera salvo.

    Returns:
        None.
    """
    merged: Dict[str, List[float]] = defaultdict(list)
    for hist in histories:
        for k, v in hist.history.items():
            merged[k].extend(v)

    out_csv = run_dir / "history.csv"
    keys = sorted(merged.keys())
    max_len = max((len(v) for v in merged.values()), default=0)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch"] + keys)
        for i in range(max_len):
            row = [i + 1]
            for k in keys:
                vals = merged[k]
                row.append(vals[i] if i < len(vals) else "")
            writer.writerow(row)


def compute_data_totals(
    split: SplitData,
    total_epochs: int,
    augmentation_config: Optional[AugmentationConfig],
) -> Dict[str, int]:
    """Calcula totalizadores de imagens reais e augmentadas no treino.

    Args:
        split: estrutura com os caminhos/labels do split.
        total_epochs: quantidade total de epocas de treinamento.
        augmentation_config: configuracao de augmentation aplicada no treino.

    Returns:
        Dict[str, int]: dicionario com totalizadores consolidados.
    """
    train_real_unique = len(split.train_paths)
    train_real_exposures_total = train_real_unique * total_epochs
    variants_per_image = 1
    if augmentation_config is not None and augmentation_config.mode == "expanded":
        variants_per_image = 8 + int(augmentation_config.flip_left_right) + int(augmentation_config.flip_up_down)
        # 8 variantes fixas: original, brilho, contraste, saturacao, hue, jpeg, sharpen, combo
        # + flips habilitados

    if augmentation_config is None:
        train_augmented_generated_total = 0
    elif augmentation_config.mode == "expanded":
        train_augmented_generated_total = train_real_unique * total_epochs * (variants_per_image - 1)
    else:
        train_augmented_generated_total = train_real_unique * total_epochs
    train_grand_total = train_real_exposures_total + train_augmented_generated_total

    return {
        "train_real_unique": train_real_unique,
        "train_real_exposures_total": train_real_exposures_total,
        "train_augmented_generated_total": train_augmented_generated_total,
        "train_grand_total": train_grand_total,
        "augmentation_mode": "disabled" if augmentation_config is None else augmentation_config.mode,
        "variants_per_real_image": variants_per_image,
    }


def _svg_map_x(i: int, n: int, x0: float, x1: float) -> float:
    if n <= 1:
        return (x0 + x1) / 2.0
    return x0 + (x1 - x0) * (i / (n - 1))


def _svg_map_y(v: float, vmin: float, vmax: float, y0: float, y1: float) -> float:
    if vmax <= vmin:
        return (y0 + y1) / 2.0
    ratio = (v - vmin) / (vmax - vmin)
    return y1 - ratio * (y1 - y0)


def _svg_line(x1: float, y1: float, x2: float, y2: float, color: str, width: int = 1, dash: str = "") -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{color}" stroke-width="{width}"{dash_attr} />'
    )


def _svg_text(x: float, y: float, value: str, size: int = 12, anchor: str = "middle", color: str = "#1f2937") -> str:
    safe = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{color}" font-family="Arial, sans-serif">{safe}</text>'
    )


def _save_metric_svg(
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
    train_points = [(_svg_map_x(i, n, x0, x1), _svg_map_y(v, vmin, vmax, y0, y1)) for i, v in enumerate(train_vals)]
    val_points = [(_svg_map_x(i, n, x0, x1), _svg_map_y(v, vmin, vmax, y0, y1)) for i, v in enumerate(val_vals)]
    test_y = _svg_map_y(test_value, vmin, vmax, y0, y1)
    train_pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in train_points)
    val_pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in val_points)

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff" />')

    for j in range(6):
        yy = y0 + (y1 - y0) * (j / 5)
        val = vmax - (vmax - vmin) * (j / 5)
        parts.append(_svg_line(x0, yy, x1, yy, "#e5e7eb", 1))
        parts.append(_svg_text(x0 - 10, yy + 4, f"{val:.3f}", size=11, anchor="end", color="#4b5563"))

    epoch_ticks = min(10, max(2, n))
    for j in range(epoch_ticks):
        idx = round((n - 1) * j / (epoch_ticks - 1)) if epoch_ticks > 1 else 0
        xx = _svg_map_x(idx, n, x0, x1)
        parts.append(_svg_line(xx, y0, xx, y1, "#f3f4f6", 1))
        parts.append(_svg_text(xx, y1 + 22, str(idx + 1), size=11, color="#4b5563"))

    parts.append(_svg_line(x0, y1, x1, y1, "#111827", 2))
    parts.append(_svg_line(x0, y0, x0, y1, "#111827", 2))
    parts.append(f'<polyline points="{train_pts}" fill="none" stroke="#2563eb" stroke-width="2" />')
    parts.append(f'<polyline points="{val_pts}" fill="none" stroke="#f59e0b" stroke-width="2" />')
    parts.append(_svg_line(x0, test_y, x1, test_y, "#16a34a", 2, "8 6"))

    parts.append(_svg_text((x0 + x1) / 2, 36, title, size=20, color="#111827"))
    parts.append(_svg_text((x0 + x1) / 2, h - 30, "Epoca", size=14, color="#111827"))
    parts.append(_svg_text(24, (y0 + y1) / 2, y_label, size=14, anchor="middle", color="#111827"))

    lx, ly = x0 + 10, y0 + 10
    parts.append(_svg_line(lx, ly, lx + 26, ly, "#2563eb", 3))
    parts.append(_svg_text(lx + 35, ly + 4, "Treino", size=12, anchor="start"))
    parts.append(_svg_line(lx + 120, ly, lx + 146, ly, "#f59e0b", 3))
    parts.append(_svg_text(lx + 155, ly + 4, "Validacao", size=12, anchor="start"))
    parts.append(_svg_line(lx + 295, ly, lx + 321, ly, "#16a34a", 3, "8 6"))
    parts.append(_svg_text(lx + 330, ly + 4, f"Teste final ({test_value:.4f})", size=12, anchor="start"))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _save_confusion_svg(matrix: List[List[int]], class_names: List[str], out_path: Path) -> None:
    n = len(matrix)
    cell = 90
    ml, mt = 280, 110
    w = ml + n * cell + 80
    h = mt + n * cell + 130
    max_v = max(max(row) for row in matrix) if matrix else 1

    parts: List[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">')
    parts.append('<rect width="100%" height="100%" fill="#ffffff" />')
    parts.append(_svg_text(w / 2, 40, "Matriz de Confusao", size=22, color="#111827"))
    parts.append(_svg_text(w / 2, 65, "Linhas: classe real | Colunas: classe predita", size=13, color="#4b5563"))

    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            intensity = int(235 - 180 * (v / max_v if max_v else 0))
            fill = f"rgb({intensity},{intensity},{255})"
            x = ml + j * cell
            y = mt + i * cell
            parts.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="#d1d5db" />')
            parts.append(_svg_text(x + cell / 2, y + cell / 2 + 5, str(v), size=18, color="#111827"))

    for i, name in enumerate(class_names):
        y = mt + i * cell + cell / 2 + 5
        x = ml - 12
        parts.append(_svg_text(x, y, name, size=12, anchor="end", color="#111827"))

    for j, name in enumerate(class_names):
        x = ml + j * cell + cell / 2
        y = mt - 14
        parts.append(_svg_text(x, y, name, size=12, anchor="middle", color="#111827"))

    parts.append(_svg_text(32, mt + (n * cell) / 2, "Classe real", size=14, anchor="start", color="#111827"))
    parts.append(_svg_text(ml + (n * cell) / 2, h - 40, "Classe predita", size=14, color="#111827"))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def _save_png_plots_with_matplotlib(
    run_dir: Path,
    history: Dict[str, List[float]],
    test_acc: float,
    test_loss: float,
    confusion_matrix: List[List[int]],
    class_names: List[str],
    per_class_metrics: Dict[str, Dict[str, float]],
) -> None:
    """Gera graficos PNG com matplotlib dentro do diretorio de run.

    Args:
        run_dir: diretorio de saida do run.
        history: historico consolidado por epoca.
        test_acc: acuracia final no conjunto de teste.
        test_loss: loss final no conjunto de teste.
        confusion_matrix: matriz de confusao em formato lista.
        class_names: nomes das classes na ordem da matriz.

    Returns:
        None.

    Raises:
        RuntimeError: se matplotlib nao estiver instalado.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib nao encontrado. Instale as dependencias com `pip install -r requirements.txt`."
        ) from exc

    epochs = np.array(history["epoch"], dtype=int)
    train_acc = np.array(history["accuracy"], dtype=float)
    val_acc = np.array(history["val_accuracy"], dtype=float)
    train_loss = np.array(history["loss"], dtype=float)
    val_loss = np.array(history["val_loss"], dtype=float)

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
    plt.savefig(run_dir / "plot_accuracy.png")
    plt.close()

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
    plt.savefig(run_dir / "plot_loss.png")
    plt.close()

    cm = np.array(confusion_matrix, dtype=int)
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
    plt.savefig(run_dir / "plot_confusion_matrix.png")
    plt.close()

    ordered_full = [name for name, _short in CLASS_BAR_ORDER if name in per_class_metrics]
    ordered_short = [short for name, short in CLASS_BAR_ORDER if name in per_class_metrics]
    if not ordered_full:
        ordered_full = class_names
        ordered_short = class_names

    precision = [100.0 * float(per_class_metrics[name]["precision"]) for name in ordered_full]
    recall = [100.0 * float(per_class_metrics[name]["recall"]) for name in ordered_full]
    f1 = [100.0 * float(per_class_metrics[name]["f1"]) for name in ordered_full]
    acc = [100.0 * float(per_class_metrics[name]["accuracy"]) for name in ordered_full]

    x = np.arange(len(ordered_full))
    w = 0.2
    plt.figure(figsize=(12, 6), dpi=140)
    plt.bar(x - 1.5 * w, precision, width=w, label="Precision")
    plt.bar(x - 0.5 * w, recall, width=w, label="Recall")
    plt.bar(x + 0.5 * w, f1, width=w, label="F1 Score")
    plt.bar(x + 1.5 * w, acc, width=w, label="Accuracy")
    plt.xticks(x, ordered_short, rotation=20, ha="right")
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.title("Metricas por Classe (Precision, Recall, F1, Accuracy)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "plot_class_metrics_histogram.png")
    plt.savefig(run_dir / "plot_class_metrics_histogram.svg")
    plt.close()


def save_run_plots(
    run_dir: Path,
    history: Dict[str, List[float]],
    test_acc: float,
    test_loss: float,
    confusion_matrix: List[List[int]],
    class_names: List[str],
    per_class_metrics: Dict[str, Dict[str, float]],
) -> None:
    """Gera automaticamente graficos SVG e PNG para o run.

    Args:
        run_dir: diretorio de saida do run.
        history: historico consolidado por epoca.
        test_acc: acuracia final no conjunto de teste.
        test_loss: loss final no conjunto de teste.
        confusion_matrix: matriz de confusao em formato lista.
        class_names: nomes das classes na ordem da matriz.

    Returns:
        None.
    """
    _save_metric_svg(
        title="Evolucao da Acuracia",
        y_label="Acuracia",
        train_vals=history["accuracy"],
        val_vals=history["val_accuracy"],
        test_value=test_acc,
        out_path=run_dir / "plot_accuracy.svg",
    )
    _save_metric_svg(
        title="Evolucao da Loss",
        y_label="Loss",
        train_vals=history["loss"],
        val_vals=history["val_loss"],
        test_value=test_loss,
        out_path=run_dir / "plot_loss.svg",
    )
    _save_confusion_svg(
        matrix=confusion_matrix,
        class_names=class_names,
        out_path=run_dir / "plot_confusion_matrix.svg",
    )
    _save_png_plots_with_matplotlib(
        run_dir=run_dir,
        history=history,
        test_acc=test_acc,
        test_loss=test_loss,
        confusion_matrix=confusion_matrix,
        class_names=class_names,
        per_class_metrics=per_class_metrics,
    )


def _prepare_run_dirs(
    args: argparse.Namespace,
    fold_index: Optional[int],
) -> Tuple[Path, Path]:
    """Define os diretorios de artefatos e TensorBoard para um experimento.

    Args:
        args: configuracao completa da execucao.
        fold_index: indice do fold ou None para modo paper.

    Returns:
        Tuple[Path, Path]:
            - diretorio do run.
            - diretorio de logs do TensorBoard.
    """
    if args.experiment_mode == "paper":
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (Path(args.out_dir) / run_name).resolve()
        tb_dir = (Path(args.tb_root_dir) / run_name).resolve()
    else:
        if fold_index is None:
            raise ValueError("fold_index e obrigatorio para preparar diretorios em modo cv.")
        run_dir = (Path(args.out_dir) / f"fold_{fold_index}").resolve()
        tb_dir = (Path(args.tb_root_dir) / f"fold_{fold_index}").resolve()

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    return run_dir, tb_dir



def _prepare_tensorboard_dir(tb_dir: Path, disable_tensorboard: bool) -> Tuple[bool, bool]:
    """Valida o caminho do TensorBoard e decide se o callback sera habilitado.

    Args:
        tb_dir: diretorio de logs pretendido para TensorBoard.
        disable_tensorboard: flag explicita para desligar TensorBoard.

    Returns:
        Tuple[bool, bool]:
            - se o TensorBoard esta habilitado para o run.
            - se o caminho e ASCII-safe.
    """
    if disable_tensorboard:
        return False, True

    tb_str = str(tb_dir)
    try:
        tb_str.encode("ascii")
    except UnicodeEncodeError:
        print(
            "TensorBoard desativado automaticamente: caminho com caracteres nao-ASCII.",
            flush=True,
        )
        print(f"Caminho detectado: {tb_dir}", flush=True)
        return False, False

    tb_dir.mkdir(parents=True, exist_ok=True)
    return True, True



def run_single_experiment(
    split: SplitData,
    config: argparse.Namespace,
    run_dir: Path,
    tb_dir: Path,
    fold_index: Optional[int] = None,
) -> Dict:
    """Executa um treinamento completo a partir de um split externo.

    Args:
        split: estrutura com train/val/test ja definidos.
        config: argumentos completos de configuracao do experimento.
        run_dir: diretorio onde os artefatos do run serao salvos.
        tb_dir: diretorio base de logs do TensorBoard para este run.
        fold_index: indice do fold no modo cv ou None no modo paper.

    Returns:
        Dict: metricas finais consolidadas do experimento.
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_enabled, _tb_safe = _prepare_tensorboard_dir(tb_dir, config.disable_tensorboard)

    if fold_index is None:
        save_split_manifest(split, run_dir)
    else:
        save_fold_manifest(split, run_dir, fold_index)

    augmentation_config = build_augmentation_config(config)
    train_ds = make_dataset(
        split.train_paths,
        split.train_labels,
        config.image_size,
        config.batch_size,
        training=True,
        num_parallel_calls=config.num_parallel_calls,
        prefetch_size=config.prefetch_size,
        augmentation_config=augmentation_config,
    )
    val_ds = make_dataset(
        split.val_paths,
        split.val_labels,
        config.image_size,
        config.batch_size,
        training=False,
        num_parallel_calls=config.num_parallel_calls,
        prefetch_size=config.prefetch_size,
        augmentation_config=None,
    )
    test_ds = make_dataset(
        split.test_paths,
        split.test_labels,
        config.image_size,
        config.batch_size,
        training=False,
        num_parallel_calls=config.num_parallel_calls,
        prefetch_size=config.prefetch_size,
        augmentation_config=None,
    )

    print("Construindo modelo InceptionV3...", flush=True)
    model, backbone = build_model(
        num_classes=len(split.class_names),
        dense_units=config.dense_units,
        image_size=config.image_size,
    )

    backbone.trainable = False
    print("Compilando modelo...", flush=True)
    compile_model(model, lr=config.lr, jit_compile=config.jit_compile)

    best_ckpt_path = ckpt_dir / "best_model.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_ckpt_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.3,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    if tb_enabled:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=str(tb_dir.as_posix())))
    else:
        print("TensorBoard desativado.")

    print("Iniciando treinamento (Fase 1)...", flush=True)
    hist_1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.freeze_epochs,
        callbacks=callbacks,
    )

    backbone.trainable = True
    fine_tune_lr = config.lr * config.fine_tune_lr_multiplier
    compile_model(model, lr=fine_tune_lr, jit_compile=config.jit_compile)

    print("Iniciando treinamento (Fase 2)...", flush=True)
    hist_2 = model.fit(
        train_ds,
        validation_data=val_ds,
        initial_epoch=config.freeze_epochs,
        epochs=config.total_epochs,
        callbacks=callbacks,
    )

    best_model = tf.keras.models.load_model(best_ckpt_path)
    eval_values = best_model.evaluate(test_ds, return_dict=True, verbose=1)
    pred_probs = best_model.predict(test_ds, verbose=1)
    y_true = np.array(split.test_labels, dtype=np.int64)
    y_pred = np.argmax(pred_probs, axis=1)

    report = confusion_and_report(y_true=y_true, y_pred=y_pred, class_names=split.class_names)
    report["keras_evaluate"] = eval_values
    report["experiment_mode"] = config.experiment_mode
    report["fold"] = fold_index

    train_acc_hist = hist_1.history.get("accuracy", []) + hist_2.history.get("accuracy", [])
    val_acc_hist = hist_1.history.get("val_accuracy", []) + hist_2.history.get("val_accuracy", [])
    train_loss_hist = hist_1.history.get("loss", []) + hist_2.history.get("loss", [])
    val_loss_hist = hist_1.history.get("val_loss", []) + hist_2.history.get("val_loss", [])
    trained_epochs = len(train_acc_hist)
    history_data = {
        "epoch": list(range(1, trained_epochs + 1)),
        "accuracy": [float(x) for x in train_acc_hist],
        "val_accuracy": [float(x) for x in val_acc_hist],
        "loss": [float(x) for x in train_loss_hist],
        "val_loss": [float(x) for x in val_loss_hist],
    }

    totals = compute_data_totals(
        split=split,
        total_epochs=trained_epochs,
        augmentation_config=augmentation_config,
    )
    report["data_totals"] = totals

    cm = np.array(report["confusion_matrix"], dtype=np.int64)
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    save_predictions_csv(
        test_paths=split.test_paths,
        y_true=y_true,
        y_pred=y_pred,
        pred_probs=pred_probs,
        class_names=split.class_names,
        run_dir=run_dir,
        fold_index=fold_index,
    )
    save_history([hist_1, hist_2], run_dir)
    save_run_plots(
        run_dir=run_dir,
        history=history_data,
        test_acc=float(eval_values.get("accuracy", 0.0)),
        test_loss=float(eval_values.get("loss", 0.0)),
        confusion_matrix=report["confusion_matrix"],
        class_names=split.class_names,
        per_class_metrics=report["per_class_metrics"],
    )
    best_model.save(run_dir / "final_model.keras")

    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    run_config = vars(config).copy()
    run_config["fold_index"] = fold_index
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Treinamento finalizado. Artefatos em: {run_dir.resolve()}")
    print(f"Acuracia de teste (Keras): {eval_values.get('accuracy', 0.0):.4f}")
    print(f"Acuracia de teste (manual): {report.get('accuracy', 0.0):.4f}")
    return report



def main() -> None:
    """Executa o experimento em modo paper ou cv para um fold especifico.

    Args:
        Nenhum argumento direto. A funcao usa parse_args().

    Returns:
        None.

    Raises:
        ValueError: se os parametros de fold forem invalidos.
        FileNotFoundError: se o dataset informado nao existir.
        RuntimeError: se GPU for exigida e nao estiver disponivel.
    """
    args = parse_args()

    if args.dataset_dir:
        args.dataset_dir = args.dataset_dir.replace("\\\\", "/")

    print(f"TensorFlow Version: {tf.__version__}")
    gpus = configure_runtime(device=args.device, use_mixed_precision=args.mixed_precision)
    if gpus:
        print(f"SUCESSO: {len(gpus)} GPU(s) detectada(s): {gpus}")
        print(f"Mixed precision policy: {mixed_precision.global_policy()}")
    else:
        print("ATENCAO: Nenhuma GPU detectada. O codigo rodara na CPU.")

    if args.total_epochs < args.freeze_epochs:
        raise ValueError("--total-epochs deve ser >= --freeze-epochs")
    if args.num_folds < 2:
        raise ValueError("--num-folds deve ser >= 2")

    set_seeds(args.seed)
    dataset_dir = Path(args.dataset_dir).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {dataset_dir}")

    inventory, _class_names = build_dataset_inventory(dataset_dir)

    if args.experiment_mode == "paper":
        split = generate_paper_split(inventory, args.seed)
        run_dir, tb_dir = _prepare_run_dirs(args, fold_index=None)
        run_single_experiment(split=split, config=args, run_dir=run_dir, tb_dir=tb_dir, fold_index=None)
        return

    if args.fold_index is None:
        raise ValueError("Modo cv requer --fold-index. Para executar todos os folds, use run_inception_cv.py.")

    split = generate_cv_split(
        dataset_inventory=inventory,
        fold_index=args.fold_index,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    run_dir, tb_dir = _prepare_run_dirs(args, fold_index=args.fold_index)
    run_single_experiment(
        split=split,
        config=args,
        run_dir=run_dir,
        tb_dir=tb_dir,
        fold_index=args.fold_index,
    )


if __name__ == "__main__":
    main()
