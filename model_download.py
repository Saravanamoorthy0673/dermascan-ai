import os
import requests

def download_model_if_needed():
    model_path = os.path.join("model", "skin_disease_model.h5")

    # ✅ ensure folder exists
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(model_path):
        print("Downloading model...")

        url = os.environ.get("MODEL_URL")
        response = requests.get(url)

        with open(model_path, "wb") as f:
            f.write(response.content)

        print("Model downloaded successfully!")
    else:
        print("Model already exists.")
