# DynamicViT-Simple

A small PyTorch Vision Transformer experiment for CIFAR-10 image classification. It trains a dense **teacher** model and then a **student** model that learns to prune image patches dynamically, using the teacher for distillation.

The model splits each 32 × 32 image into 64 patches. The student has three pruning stages and, with the default keep ratio of `0.7`, targets retaining approximately 70%, 49%, and 34% of the original patches at those stages. At inference time it selects the highest-scoring patches at each stage.

## Requirements

- Python
- PyTorch
- Torchvision
- Pillow

Install the packages in your environment:

```bash
python -m pip install torch torchvision pillow
```

For GPU training, install the PyTorch build compatible with your CUDA setup. The scripts automatically use CUDA when available and otherwise run on the CPU.

## Train

Run from the project directory:

```bash
python train.py
```

The script downloads CIFAR-10 to `data/` if it is not already present, trains the teacher for 15 epochs, then trains the student for 10 epochs. Training uses a batch size of 64. Progress reports include loss, training and test accuracy, patch-retention rates, and epoch time.

The trained model state dictionaries are saved to:

- `checkpoints/teacher.pth`
- `checkpoints/student.pth`

The prediction example below uses `checkpoints/student.pth`. Running training again overwrites the checkpoint files.

## Predict an image

After training, classify an image with:

```bash
python predict.py --image path/to/image.jpg
```

Optionally specify a different student checkpoint:

```bash
python predict.py --image path/to/image.jpg --checkpoint path/to/student.pth
```

The image is converted to RGB, resized to 32 × 32, and normalized using the CIFAR-10 channel statistics. The script prints the predicted class, confidence, number of patches retained, and the resulting patch-count reduction.

The model's class labels are:

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`.

## Project files

- `model.py` — Vision Transformer, attention masking, and patch-pruning prediction modules.
- `train.py` — CIFAR-10 data loading, teacher/student training, evaluation, and checkpoint saving.
- `predict.py` — Single-image inference using the trained student.
- `RAPPORT_ENTRAINEMENT.md` — Rapport détaillé (en français) sur l'entraînement, le Gumbel-Softmax, le masquage d'attention et les fonctions de perte.
