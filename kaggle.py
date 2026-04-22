import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "shriyashjagtap/fraudulent-e-commerce-transactions")

print("Path to dataset files:", path)
