
import argparse
import os

import torch
from PIL import Image
from torchvision import transforms

from model import SmallViT


CLASSES = [
    "avion", "automobile", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion",
]

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image", required=True,
        help="Chemin vers une image à classifier",
    )
    parser.add_argument(
        "--checkpoint", default="checkpoints/student.pth",
        help="Chemin du modèle entraîné",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image introuvable : {args.image}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Modèle introuvable : {args.checkpoint}. Lance d'abord train.py."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = SmallViT(use_pruning=True, keep_ratio=0.7).to(device)

    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, tokens, scores = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        confidence, predicted = probabilities.max(dim=0)

    print("\n=== Résultat de la prédiction ===")
    print("Classe prédite :", CLASSES[predicted.item()])
    print(f"Confiance du modèle : {confidence.item() * 100:.2f}%")
    print(f"Patchs conservés : {tokens.shape[1]} sur 64")
    print(f"Réduction du nombre de patchs : "
          f"{100 * (1 - tokens.shape[1] / 64):.1f}%")


if __name__ == "__main__":
    main()
