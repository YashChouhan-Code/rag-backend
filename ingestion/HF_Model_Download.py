from huggingface_hub import snapshot_download

model_id = "microsoft/table-transformer-structure-recognition" # Replace with the model you want
project_folder = r"ingestion\Table_Trans_Model" # Your specific local directory

# Download all files directly to your project folder
download_path = snapshot_download(
    repo_id=model_id,
    local_dir=project_folder
)

print(f"Model successfully downloaded to: {download_path}")