import os
from pathlib import Path

# ✅ Project Name
PROJECT_NAME = "LLM_gpt2_classifier"

# ✅ Folder Structure
files_to_create = [
    f"{PROJECT_NAME}/data",
    f"{PROJECT_NAME}/streamlit_app.py",
    f"{PROJECT_NAME}/requirements.txt",
    f"{PROJECT_NAME}/README.md",
    f"{PROJECT_NAME}/models",
    f"{PROJECT_NAME}/output",
    f"{PROJECT_NAME}/src/model.py",
    f"{PROJECT_NAME}/src/inference.py",
    f"{PROJECT_NAME}/src/utils.py",
    f"{PROJECT_NAME}/research/experiment_1.ipynb",
    f"{PROJECT_NAME}/assets/.gitkeep",  # keeps empty folder in git
]

# ✅ Create Files & Folders
for file_path in files_to_create:
    file_path = Path(file_path)

    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create file if not exists
    if not file_path.exists():
        file_path.touch()
        print(f"Created: {file_path}")

print("\n✅ Project structure created successfully!")