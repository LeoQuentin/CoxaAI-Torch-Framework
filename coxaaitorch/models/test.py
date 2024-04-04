from coxaaitorch.models import create_model, list_available_models


# List available models
print(list_available_models())

# Create a model instance
if __name__ == "__main__":
    for model in list_available_models():
        print(model)
        if model == "vit-base-patch16-224":
            instance = create_model(
                model, size=(224, 224), classes=2, pretrained=True, channels=3
            )
            print(instance)
