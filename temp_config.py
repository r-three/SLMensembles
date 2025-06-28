synthetic_data = False
dataset_path = "/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized"
synthetic_dataset_path = "/scratch/klambert/dataset/synthetic_dataset"
base_output_dir = "/scratch/klambert/model_log"
log_dir = "/home/klambert/projects/aip-craffel/klambert/SLMensembles/csv_logs"

id_string = "Experiment with hyperparameters to check if distillation works"
description = "Alpha tweaking: alpha = 1"
custom_run_name = "alpha1_hyperparameters"

ensemble_model_names = []
ensemble_members = []  # Full path of ensemble models which we want to load
checkpoint_path = None  # Full path of the model checkpoint from which we want to resume training (else None)
checkpoint_log_path = None  # exact path of the csv file corresponding to the checkpointed model

# Hyperparameters
learning_rate = 5e-5
alpha = 1  # 1 = next_token loss to 0 = kl_loss
total_rounds = 1  # number of ensemble models
steps_per_round = 20000
