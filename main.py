import os

from constant import MEAN, STD
from find_best_treshhold import find_and_save_optimal_threshold
from test import evaluate_model_and_save_plot
from train import train_autoencoder
from validate import validate
from make_validation import make_validation


def run_experiment(experiment_name: str,
                   mse_factor=None) -> None:

    assert mse_factor is not None, "mse_factor must be provided"

    print("=" * 10)
    print(f"Имя эксперимента: {experiment_name}")
    print("=" * 10)


    dataset_path = "dataset/train"
    val_dataset_path = "dataset/val"
    model_path = f"models/{experiment_name}.pth"

    mean, std = MEAN, STD


    print(f"Средние значения: {mean}")
    print(f"Стандартные отклонения: {std}")

    make_validation()

    if not os.path.exists(model_path):
        autoencoder, model_path = train_autoencoder(dataset_path,
                                                    val_dataset_path,
                                                    mean,
                                                    std,
                                                    model_path,
                                                    experiment_name,
                                                    mse_factor)

    if not os.path.isdir(experiment_name):
        validate(model_path, experiment_name=experiment_name, mse_factor=mse_factor)



    thresholds = find_and_save_optimal_threshold(
        model_path=model_path,
        val_dataset_path=val_dataset_path,
        proliv_dataset_path="dataset/proliv",
        mean=mean,
        std=std,
        device="cuda",
        save_path=os.path.join(experiment_name, "threshold_metrics_combined_loss.png"),
        metrics_file_path=os.path.join(experiment_name, "metrics.txt"),
        mse_factor=mse_factor,
    )

    evaluate_model_and_save_plot(
        model_path=model_path,
        dataset_path='dataset/test/imgs',
        labels_file='dataset/test/test_annotation.txt',
        plot_save_path=os.path.join(experiment_name, "test_dist.png"),
        thresholds=thresholds,
        mse_factor=mse_factor,
    )

if __name__ == "__main__":
    experiments = [
        # {"experiment_name": "UNET_MSE_FACTOR_00_VGG", "mse_factor": 0.0},
        # {"experiment_name": "UNET_MSE_FACTOR_03_VGG", "mse_factor": 0.3},
        {"experiment_name": "UNET_MSE_FACTOR_010_VGG_NOISE", "mse_factor": 0.10},
        # {"experiment_name": "UNET_MSE_FACTOR_08", "mse_factor": 0.8},
        # {"experiment_name": "UNET_MSE_FACTOR_10", "mse_factor": 0.10},
    ]

    for experiment in experiments:
        run_experiment(**experiment)


