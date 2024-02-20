import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ConvNextV2ForImageClassification, ConvNextV2Config
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Define transformations: Resize to 224x224, convert to tensor, and normalize (as ConvNeXt expects)
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize images to 224x224
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  # Adjusted for 1 channel
])


# Load MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Initialize the model
config = ConvNextV2Config(num_channels=1, patch_size=4, image_size=800, num_labels=10)
model = ConvNextV2ForImageClassification(config)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()

model.train()
for epoch in range(10):  # Example: Train for 1 epoch
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs).logits
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

