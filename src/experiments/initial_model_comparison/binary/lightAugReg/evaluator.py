import torch # noqa
from torchvision import transforms
import os
from datetime import timedelta
import sys
import dotenv

# huggingface model
from transformers import AutoConfig, AutoModelForImageClassification
# Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau

dotenv.load_dotenv()
project_root = os.getenv('PROJECT_ROOT')
if project_root:
    sys.path.append(project_root)
from src.models.BaseNormalAbnormal import BaseNormalAbnormal # noqa
from src.utilities.H5DataModule import H5DataModule # noqa
from src.utilities.np_image_to_PIL import np_image_to_PIL # noqa
from src.augmentation.autoaugment import ImageNetPolicy # noqa

log_dir = os.path.join(project_root, "/mnt/users/leobakh/VET_project/VET-Special-syllabus/src/experiments/initial_model_comparison/binary/lightAugReg/logs") # noqa
checkpoint_dir = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/src/experiments/initial_model_comparison/binary/lightAugReg/modelcheckpoints/google" # noqa


def val_test_preprocess(image, size):
    # basically same as train_preprocess but without the augmentations
    image = np_image_to_PIL(image)  # convert to PIL image

    transform_pipeline = transforms.Compose([
        transforms.Resize(size),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])
    image = transform_pipeline(image)

    if len(image.size()) == 4:
        image = image.squeeze(0)
    return image


class EfficientNet(BaseNormalAbnormal):
    def __init__(self, config, *args, **kwargs):
        model = AutoModelForImageClassification.from_config(config)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
        super().__init__(model=model, *args, **kwargs)


if __name__ == "__main__":
    test_folds = [4]
    logger = CSVLogger(save_dir=log_dir, name="efficientNet_Test_Results")

    for checkpoint_file in os.listdir(checkpoint_dir):
        if checkpoint_file.startswith("efficientnet-"):
            model_name = checkpoint_file.split("_")[0]
            image_size = int(checkpoint_file.split("_")[3])

            config = AutoConfig.from_pretrained(f"google/{model_name}")
            config.image_size = (image_size, image_size)
            config.num_channels = 1

            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            model = EfficientNet.load_from_checkpoint(checkpoint_path, config=config)

            dm = H5DataModule(os.getenv("DATA_FILE"),
                              batch_size=8,
                              train_folds=[],
                              val_folds=[],
                              test_folds=test_folds,
                              target_var='target',
                              train_transform=lambda x: val_test_preprocess(x, (image_size, image_size)),  # noqa
                              val_transform=lambda x: val_test_preprocess(x, (image_size, image_size)),  # noqa
                              test_transform=lambda x: val_test_preprocess(x, (image_size, image_size)))  # noqa

            trainer = Trainer(accelerator="auto", logger=logger)
            test_loss = trainer.test(model, dm)

            print(f"Model: {model_name}, Image Size: {image_size}")
            print(f"Test Results: {test_loss}")
            print()
