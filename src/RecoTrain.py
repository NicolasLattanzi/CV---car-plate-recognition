import torch
import torch.nn as nn
from torch.utils.data import *
from torchvision import transforms
import data
import network
import globals
from tqdm import tqdm

def main():
    transform = transforms.ToTensor()
    train_dataset = data.LicensePlateDataset(
        img_dir="crop photo/img",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1028,
        shuffle=True,
        num_workers=6
    )

    NUM_CLASSES = len(globals.unique_total) + 1
    NUM_EPOCHS = 15
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    BLANK_IDX = len(globals.unique_total)

    model = network.LicensePlateResNet(num_classes=NUM_CLASSES)
    model=model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = len(train_loader)
        pbar=tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}  [Train]", unit="batch")
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        for batch_idx, (images, targets, target_lengths) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            logits = model(images)
            log_probs = nn.functional.log_softmax(logits, dim=2)
            input_lengths = torch.full(size=(images.size(0),), fill_value=logits.size(0), dtype=torch.long)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Messaggio ogni 20 batch
            if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == num_batches:
                print(f"  Batch {batch_idx+1}/{num_batches} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / num_batches
        print(f"Fine epoca {epoch+1}: Loss media = {avg_loss:.4f}")

    torch.save(model.state_dict(), "license_plate_model.pth")
    print("Modello salvato in license_plate_model.pth")

if __name__ == "__main__":
    main()
