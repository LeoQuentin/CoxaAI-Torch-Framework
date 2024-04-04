# model_utils.py
MODEL_REGISTRY = {}


def registermodel(model_name):
    """
    Decorator function to register a model creation function in the MODEL_REGISTRY.

    Args:
        model_name (str): The name of the model to register.

    Returns:
        function: The decorated model creation function.
    """

    def decorator(model_func):
        MODEL_REGISTRY[model_name] = model_func
        return model_func

    return decorator


def list_available_models():
    """
    Returns a list of available model names in the MODEL_REGISTRY.

    Returns:
        list: A list of available model names.
    """
    return list(MODEL_REGISTRY.keys())


def get_model_creation_func(model_name):
    """
    Retrieves the model creation function from the MODEL_REGISTRY based on the model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        function: The model creation function.

    Raises:
        ValueError: If the model name is not found in the MODEL_REGISTRY.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name]


def create_model(model_name, *args, **kwargs):
    """
    Creates a model instance based on the provided model name and arguments.

    Args:
        model_name (str): The name of the model to create.
        *args: Variable length argument list to pass to the model creation function.
        **kwargs: Arbitrary keyword arguments to pass to the model creation function.

    Returns:
        object: The created model instance.

    Raises:
        ValueError: If the model name is not found in the MODEL_REGISTRY.

    Example:
        # Create an EfficientNet model
        model = create_model(
            "efficientnet_b0",
            size=(224, 224),
            classes=10,
            pretrained=False,
            channels=3,
            config={"dropout_rate": 0.2}
        )
    """
    model_func = get_model_creation_func(model_name)
    return model_func(*args, **kwargs)
