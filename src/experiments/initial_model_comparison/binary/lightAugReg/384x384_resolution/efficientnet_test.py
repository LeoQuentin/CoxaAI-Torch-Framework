import torch
from transformers import AutoConfig, AutoModelForImageClassification


class EfficientNet_384(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = AutoModelForImageClassification.from_config(config)
        self.model.classifier = torch.nn.Linear(self.model.classifier.in_features, 2)

    def forward(self, x):
        return self.model(x)


checkpoint_path = "/mnt/users/leobakh/VET_project/VET-Special-syllabus/src/experiments/initial_model_comparison/binary/lightAugReg/modelcheckpoints/google/efficientnet-b0_binary_lightAugReg_384_best_checkpoint_epoch=23_val_loss=0.25.ckpt"  # noqa
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint["state_dict"]

model_id = "google/efficientnet-b0"  # Replace with the appropriate model ID
config = AutoConfig.from_pretrained(model_id)
config.image_size = (384, 384)  # Replace with the appropriate image size
config.num_channels = 1

model = EfficientNet_384(config)
model.load_state_dict(state_dict)

