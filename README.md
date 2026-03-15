# Classificacao de Graos de Soja com InceptionV3 Modificado

Este projeto reproduz, de forma aproximada, o artigo de classificacao de sementes de soja com `InceptionV3` modificada.

## Modos de execucao

- `Windows local (CPU)`: caminho mais simples e estavel.
- `Windows + Docker Desktop (GPU)`: usa container Linux com suporte NVIDIA. Este e o caminho certo quando a GPU nao funciona bem no TensorFlow local.

## Dataset

O dataset esperado fica em uma pasta vizinha ao projeto:

```text
..\dataset_kaggle_soja
```

Classes esperadas:

- `Broken soybeans`
- `Immature soybeans`
- `Intact soybeans`
- `Skin-damaged soybeans`
- `Spotted soybeans`

## Opcao A: Windows local (CPU)

### 1. Criar a `.venv`

No PowerShell, dentro da pasta do projeto:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

### 2. Instalar dependencias

No Windows local, nao use `tensorflow[and-cuda]`. O `requirements.txt` atual foi mantido para o fluxo com GPU em container.

Instale manualmente:

```powershell
python -m pip install --upgrade pip
pip install "tensorflow==2.16.2" "numpy>=1.22" "matplotlib>=3.8"
python -m pip install --upgrade setuptools==69.5.1 tensorboard
```

### 3. Rodar treino simples

```powershell
cd C:\Projetos\Pos_ia\TCC\impl-art-inception
.\.venv\Scripts\Activate.ps1
python train_modified_inception_repro.py --dataset-dir C:\Projetos\Pos_ia\TCC\dataset_kaggle_soja --disable-tensorboard
```

### 4. Rodar treino + TensorBoard

Use caminhos ASCII para os logs e artefatos. Isso evita falha de encoding por causa de acentos no caminho do projeto.

Janela 1:

```powershell
.\.venv\Scripts\Activate.ps1
python train_modified_inception_repro.py `
  --dataset-dir C:\Projetos\Pos_ia\TCC\dataset_kaggle_soja `
  --out-dir C:\tb_runs\soy_inception `
  --tb-root-dir C:\tb_logs\soy_inception
```

Janela 2:

```powershell
.\.venv\Scripts\Activate.ps1
tensorboard --logdir C:\tb_logs\soy_inception
```

Abrir no navegador:

```text
http://localhost:6006
```

Se a porta estiver ocupada:

```powershell
tensorboard --logdir C:\tb_logs\soy_inception --port 6007
```

## Opcao B: Windows + Docker Desktop (GPU)

Este fluxo depende de:

- Docker Desktop instalado
- backend WSL2 habilitado
- suporte NVIDIA ao Docker funcionando

### 1. Validar Docker

No PowerShell:

```powershell
docker --version
docker run --rm hello-world
```

### 2. Validar GPU no container

```powershell
docker run --rm --gpus all nvidia/cuda:13.1.1-base-ubuntu22.04 nvidia-smi
```

### 3. Subir container TensorFlow NVIDIA

Dentro da pasta do projeto no Windows:

```powershell
docker run --rm -it --gpus all `
  -v "${PWD}:/workspace/project" `
  -v "C:\Projetos\Pos_ia\TCC\dataset_kaggle_soja:/workspace/dataset_kaggle_soja" `
  -w /workspace/project `
  nvcr.io/nvidia/tensorflow:25.01-tf2-py3 `
  bash
```

Se o dataset estiver em outro caminho no Windows, ajuste o segundo `-v`.

### 4. Instalar dependencias no container

Dentro do container:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Rodar treino com GPU

Dentro do container:

```bash
python train_modified_inception_repro.py \
  --dataset-dir /workspace/dataset_kaggle_soja \
  --out-dir /workspace/project/runs_modified_inception \
  --tb-root-dir /workspace/project/tf_tb_logs
```

### 6. TensorBoard no container

Em outro terminal do Windows:

```powershell
docker run --rm -it --gpus all `
  -p 6006:6006 `
  -v "${PWD}:/workspace/project" `
  -w /workspace/project `
  nvcr.io/nvidia/tensorflow:25.01-tf2-py3 `
  bash
```

Dentro do segundo container:

```bash
tensorboard --logdir /workspace/project/tf_tb_logs --host 0.0.0.0 --port 6006
```

## Artefatos de saida

Cada treino gera uma pasta em `runs_modified_inception/<timestamp>/` com:

- `split_manifest.csv`
- `history.csv`
- `checkpoints/best_model.keras`
- `final_model.keras`
- `confusion_matrix.csv`
- `metrics.json`
- `run_config.json`

## Observacoes praticas

- `--out-dir` guarda artefatos do experimento.
- `--tb-root-dir` guarda eventos do TensorBoard.
- Se usar `--disable-tensorboard`, o TensorBoard vai mostrar `No dashboards are active`.
- No Windows local, o fluxo recomendado e CPU.
- Para GPU com RTX 50xx, use Docker com backend WSL2.