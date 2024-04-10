import csv
import os


def find_best_model_metrics(file_paths):
    result = ""

    for file_path in file_paths:
        best_val_loss = float('inf')
        best_epoch = None
        best_model_metrics = None
        test_metrics = None

        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row['val_loss']:
                    val_loss = float(row['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_metrics = row
                        best_epoch = row['epoch']
                if row['test_loss']:
                    test_metrics = row

        if best_model_metrics:
            result += "=" * 50 + "\n"
            result += f"Best Metrics for {os.path.basename(file_path)}:\n"
            result += "=" * 50 + "\n\n"
            result += f"Epoch: {best_epoch}\n\n"
            result += "Validation Metrics:\n"
            result += "-" * 20 + "\n"
            for metric, value in best_model_metrics.items():
                if metric.startswith('val_') and value:
                    result += f"{metric}: {value}\n"
            result += "\n"

            if test_metrics:
                result += "Test Metrics:\n"
                result += "-" * 20 + "\n"
                for metric, value in test_metrics.items():
                    if metric.startswith('test_') and value:
                        result += f"{metric}: {value}\n"
            else:
                result += "No test metrics found in the CSV file.\n"

            result += "\n"
        else:
            result += f"No valid metrics found in {os.path.basename(file_path)}.\n\n"

    return result
