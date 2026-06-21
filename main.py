import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from convolutional_nn import NeuralNetwork, transform_data

# AMD specific
torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"


# Code runs only in the main process
if __name__ == "__main__":
    print(f"Using {device} device")
    dataset = datasets.ImageFolder(root='../PetImages', transform=transform_data)
    dataset_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=6, pin_memory=True)

    # Start compilation
    model = NeuralNetwork().to(device)
    optimized_model = torch.compile(model)
    # End compilation

    # Start training
    # Loss function
    loss = nn.CrossEntropyLoss()
    # Optimizer updates weights based on loss
    optimizer = torch.optim.Adam(optimized_model.parameters(), lr=0.001)
    print("Start training")
    model.train()
    # 5 epochs = cat bias, 10 epochs = 2/3 correct, 20 epochs = 3/3 correct :)
    epochs = 20
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for batch, (x, y) in enumerate(dataset_loader):
            x, y = x.to(device), y.to(device)

            pred = optimized_model(x)
            loss_value = loss(pred, y)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"Batch {batch}/{len(dataset_loader)} - Loss: {loss_value.item():.4f}")

    torch.save(optimized_model._orig_mod.state_dict(), "cat_dog_model.pth")
    print("Model saved to cat_dog_model.pth!")
