import os
import time
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SmallViT


# ---------------- Configuration ----------------
SEED = 42
BATCH_SIZE = 64
TEACHER_EPOCHS = 15
STUDENT_EPOCHS = 10
NUM_WORKERS = 0

KEEP_RATIO = 0.7          # rho : ratios cibles des 3 étages = [rho, rho^2, rho^3]
NUM_STAGES = 3

# Poids de l'Eq. 16 (valeurs de l'article)
LAMBDA_KL = 0.5
LAMBDA_DISTILL = 0.5
LAMBDA_RATIO = 2.0

# Optimisation du Student (adapté à CIFAR-10 ; l'article utilise
# lr_pred = batch/1024 * 1e-3 et lr_backbone = 0.01 * lr_pred sur ImageNet)
LR_TEACHER = 1e-3
LR_PREDICTOR = 1e-3
LR_BACKBONE = 1e-4
FREEZE_BACKBONE_EPOCHS = 5   # backbone gelé pendant les 5 premières époques

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("checkpoints", exist_ok=True)

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device utilisé :", DEVICE)


# ---------------- Données CIFAR-10 ----------------
mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

train_dataset = datasets.CIFAR10("data", train=True, download=True,
                                 transform=train_transform)
test_dataset = datasets.CIFAR10("data", train=False, download=True,
                                transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS,
                          pin_memory=(DEVICE.type == "cuda"))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS,
                         pin_memory=(DEVICE.type == "cuda"))


# ---------------- Évaluation ----------------
@torch.no_grad()
def evaluate(model, loader):
    model.eval()   # le Student élague physiquement les tokens (Eq. 17)
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits, _, _ = model(images)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


# ---------------- Les 4 pertes de l'article ----------------
def dynamicvit_losses(student_logits, student_tokens, decisions,
                      teacher_logits, teacher_tokens, labels):
    """
    decisions : liste de S tenseurs D_hat^s [B, N, 1] (masque cumulatif, 0/1)
    """
    # Eq. 12 : entropie croisée
    loss_cls = F.cross_entropy(student_logits, labels)

    # Eq. 14 : divergence KL entre prédictions Student et Teacher (sans température)
    loss_kl = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )

    # Eq. 13 : distillation des tokens finaux, moyennée uniquement sur
    # les tokens conservés au dernier étage (pondération par D_hat^S).
    w = decisions[-1].detach()                                   # [B, N, 1]
    sq_err = (student_tokens - teacher_tokens.detach()).pow(2).mean(-1, keepdim=True)
    loss_distill = (sq_err * w).sum() / w.sum().clamp(min=1.0)

    # Eq. 15 : (1/BS) sum_b sum_s ( rho^s - (1/N) sum_i D_hat^{b,s}_i )^2
    keep_rates = torch.stack([d.mean(dim=(1, 2)) for d in decisions], dim=1)  # [B, S]
    targets = torch.tensor(
        [KEEP_RATIO ** (s + 1) for s in range(len(decisions))],
        device=keep_rates.device, dtype=keep_rates.dtype,
    )
    loss_ratio = (targets.unsqueeze(0) - keep_rates).pow(2).mean()

    # Eq. 16
    loss = (loss_cls
            + LAMBDA_KL * loss_kl
            + LAMBDA_DISTILL * loss_distill
            + LAMBDA_RATIO * loss_ratio)

    return loss, {
        "cls": loss_cls.item(),
        "kl": loss_kl.item(),
        "distill": loss_distill.item(),
        "ratio": loss_ratio.item(),
    }, keep_rates.mean(0)


# ---------------- Teacher ----------------
def train_teacher(model, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_TEACHER,
                                  weight_decay=0.01)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        start = time.time()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        print(f"[Teacher] Époque {epoch}/{epochs} | loss={total_loss/total:.4f} | "
              f"train_acc={100*correct/total:.2f}% | "
              f"test_acc={evaluate(model, test_loader):.2f}% | "
              f"temps={time.time()-start:.1f}s")
    torch.save(model.state_dict(), "checkpoints/teacher.pth")


# ---------------- Student ----------------
def train_student(student, teacher, epochs):
    predictor_params = list(student.predictors.parameters())
    predictor_ids = {id(p) for p in predictor_params}
    backbone_params = [p for p in student.parameters()
                       if id(p) not in predictor_ids]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE, "name": "backbone"},
        {"params": predictor_params, "lr": LR_PREDICTOR, "name": "predictor"},
    ], weight_decay=0.01)

    teacher.eval()

    for epoch in range(1, epochs + 1):
        # Backbone gelé (lr = 0) pendant les premières époques
        for g in optimizer.param_groups:
            if g["name"] == "backbone":
                g["lr"] = 0.0 if epoch <= FREEZE_BACKBONE_EPOCHS else LR_BACKBONE

        student.train()
        sums = {"total": 0.0, "cls": 0.0, "kl": 0.0, "distill": 0.0, "ratio": 0.0}
        rates_sum = torch.zeros(NUM_STAGES, device=DEVICE)
        correct, total = 0, 0
        start = time.time()

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            with torch.no_grad():
                teacher_logits, teacher_tokens, _ = teacher(images)

            optimizer.zero_grad(set_to_none=True)
            student_logits, student_tokens, aux = student(images)

            loss, parts, rates = dynamicvit_losses(
                student_logits, student_tokens, aux["decisions"],
                teacher_logits, teacher_tokens, labels,
            )
            loss.backward()
            optimizer.step()

            n = labels.size(0)
            total += n
            correct += (student_logits.argmax(1) == labels).sum().item()
            sums["total"] += loss.item() * n
            for k, v in parts.items():
                sums[k] += v * n
            rates_sum += rates.detach() * n

        rates_avg = (rates_sum / total).tolist()
        print(f"[Student] Époque {epoch}/{epochs} | loss={sums['total']/total:.4f} | "
              f"cls={sums['cls']/total:.4f} | KL={sums['kl']/total:.4f} | "
              f"distill={sums['distill']/total:.4f} | ratio={sums['ratio']/total:.4f}")
        print(f"           train_acc={100*correct/total:.2f}% | "
              f"test_acc={evaluate(student, test_loader):.2f}% | "
              f"conservation par étage={[round(r, 3) for r in rates_avg]} | "
              f"temps={time.time()-start:.1f}s")

    torch.save(student.state_dict(), "checkpoints/student.pth")


def main():
    print("\n=== 1. Teacher ===")
    teacher = SmallViT(use_pruning=False).to(DEVICE)
    train_teacher(teacher, TEACHER_EPOCHS)

    print("\n=== 2. Student (DynamicViT) ===")
    student = SmallViT(use_pruning=True, keep_ratio=KEEP_RATIO).to(DEVICE)
    # Initialisation depuis le Teacher pré-entraîné (comme dans l'article).
    # strict=False : les prédicteurs n'existent pas dans le Teacher.
    student.load_state_dict(teacher.state_dict(), strict=False)

    train_student(student, teacher, STUDENT_EPOCHS)

    print("\n=== Résumé final ===")
    print(f"Teacher test accuracy : {evaluate(teacher, test_loader):.2f}%")
    print(f"Student test accuracy : {evaluate(student, test_loader):.2f}%")


if __name__ == "__main__":
    main()