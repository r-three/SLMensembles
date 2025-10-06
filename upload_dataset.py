from datasets import load_from_disk
path = "/scratch/klambert/dataset/tulu-3-sft-mixture-pretokenized"
ds = load_from_disk(path)   # ds is a DatasetDict with 'train' and 'test'

# Choose your repo name: "username/dataset_name"
repo_id = "klambert/tulu-3-sft-mixture-preprocessed"

#ds.push_to_hub(repo_id)

print(ds["train"][0])
