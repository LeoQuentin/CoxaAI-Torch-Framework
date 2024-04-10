import csv
import os


def print_experiment_metrics(file_paths):
    best_model_path = None
    best_val_loss = float("inf")
    best_epoch = None

    for file_path in file_paths:
        with open(file_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["val_loss"]:
                    val_loss = float(row["val_loss"])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = file_path
                        best_model_metrics = row
                        best_epoch = row["epoch"]

    if best_model_path:
        result = f"Best Model Metrics (from {os.path.basename(best_model_path)}):\n"
        result += "=" * 50 + "\n"
        result += f"Epoch: {best_epoch}\n"
        result += "Validation Metrics:\n"
        for metric, value in best_model_metrics.items():
            if metric.startswith("val_") and value:
                result += f"{metric}: {value}\n"
        result += "\n"

        with open(best_model_path, "r") as file:
            reader = csv.DictReader(file)
            test_metrics = None
            for row in reader:
                if row["test_loss"]:
                    test_metrics = row
                    break

        if test_metrics:
            result += "Test Metrics:\n"
            for metric, value in test_metrics.items():
                if metric.startswith("test_") and value:
                    result += f"{metric}: {value}\n"
        else:
            result += "No test metrics found in the best model's CSV file.\n"
        result += "\n\n" + "=" * 50
    else:
        result = "No valid model metrics found."
    return result
