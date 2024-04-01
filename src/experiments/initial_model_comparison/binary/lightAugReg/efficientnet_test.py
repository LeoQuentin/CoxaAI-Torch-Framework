import torch
from torchvision import transforms
import os
from datetime import timedelta
import sys
import dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm


# huggingface model
from transformers import AutoConfig, AutoModelForImageClassification

# Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

dotenv.load_dotenv()
project_root = os.getenv("PROJECT_ROOT")
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal  # noqa
from src.utilities.H5DataModule import H5DataModule  # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL  # noqa
from src.augmentation.autoaugment import ImageNetPolicy  # noqa

size = (384, 384)


def train_preprocess(image):
    # image is a numpy array in the shape (H, W, C)
    image = np_image_to_PIL(image)  # convert to PIL image

    # Preprocess the image
    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.RandomRotation(10),
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)

    # Remove the batch dimension if it exists
    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


def val_test_preprocess(image):
    # basically same as train_preprocess but without the augmentations
    image = np_image_to_PIL(image)  # convert to PIL image

    transform_pipeline = transforms.Compose(
        [
            transforms.Resize(size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )
    image = transform_pipeline(image)

    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


class EfficientNet_384(BaseNormalAbnormal):
    def __init__(self, config):
        model = AutoModelForImageClassification.from_config(config)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

        super().__init__(model=model)


def evaluate_model(checkpoint_path, model_id, image_size):
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint["state_dict"]

    config = AutoConfig.from_pretrained(model_id)
    config.image_size = image_size
    config.num_channels = 1

    model = EfficientNet_384(config)
    model.load_state_dict(state_dict)
    model.eval()

    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get the test dataloader
    dm.setup(stage="test")
    test_dataloader = dm.test_dataloader()

    # Initialize variables to store predictions and labels
    all_predictions = []
    all_labels = []

    # Disable gradient computation
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels = batch

            # Move inputs and labels to the appropriate device (e.g., GPU)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Run inference on the test batch
            outputs = model(inputs)

            # Get the predicted labels
            _, predicted = torch.max(outputs, 1)

            # Append predictions and labels to the lists
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1


# Directory containing the checkpoint files
checkpoint_dir = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/src/experiments/initial_model_comparison/binary/lightAugReg/modelcheckpoints/google"

# Initialize the data module
dm = H5DataModule(
    os.getenv("DATA_FILE"),
    batch_size=16,
    train_folds=[0],
    val_folds=[0],
    test_folds=[4],
    target_var="target",
    train_transform=train_preprocess,
    val_transform=val_test_preprocess,
    test_transform=val_test_preprocess,
)

# Open the file to write the metrics
with open("efficientnet_metrics.txt", "w") as file:
    # Iterate over the checkpoint files
    for filename in tqdm(os.listdir(checkpoint_dir)):
        if filename.startswith("efficientnet") and filename.endswith(".ckpt"):
            print(f"Evaluating model: {filename}")
            checkpoint_path = os.path.join(checkpoint_dir, filename)
            model_id = filename.split("_")[0]
            image_size = tuple(map(int, filename.split("_")[3].split("x")))
            config_name = f"google/{model_id}"

            accuracy, precision, recall, f1 = evaluate_model(
                checkpoint_path, model_id, image_size
            )

            # Write the metrics to the file
            file.write(f"Model: {model_id}, Image Size: {image_size}\n")
            file.write(f"Accuracy: {accuracy:.4f}\n")
            file.write(f"Precision: {precision:.4f}\n")
            file.write(f"Recall: {recall:.4f}\n")
            file.write(f"F1-score: {f1:.4f}\n")
            file.write("\n")
