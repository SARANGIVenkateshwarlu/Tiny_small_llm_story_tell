import torch
from pathlib import Path
import tiktoken
import yaml

DEFAULT_CHECKPOINT_CANDIDATES = [
    Path("./models/final_best_model_state_dict.pt"),
    Path("./models/model_state_dict.pt"),
]

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def get_device(config=None):
    config = config or load_config()
    if config["device"]["auto_detect"]:
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        return config["device"]["manual_device"]

def get_checkpoint_path(candidate_paths=None, default_path=None, config=None):
    config = config or load_config()
    candidate_paths = list(candidate_paths or config["paths"]["checkpoint_candidates"])
    candidate_paths = [Path(p) for p in candidate_paths]
    if default_path is not None:
        candidate_paths.insert(0, Path(default_path))

    for path in candidate_paths:
        if path.exists():
            return path
    return None

def load_checkpoint(path=None, candidate_paths=None, default_path=None, config=None):
    config = config or load_config()
    checkpoint_path = path or get_checkpoint_path(candidate_paths=candidate_paths, default_path=default_path, config=config)
    if checkpoint_path is None:
        expected = [str(p) for p in (candidate_paths or config["paths"]["checkpoint_candidates"])]
        raise FileNotFoundError(
            "No checkpoint file found. Expected one of: " + ", ".join(expected)
        )
    return torch.load(checkpoint_path, map_location="cpu"), checkpoint_path

def load_tokenizer(encoding=None, config=None):
    config = config or load_config()
    encoding = encoding or config["tokenizer"]["encoding"]
    return tiktoken.get_encoding(encoding)

def build_model_from_checkpoint(ckpt, device=None, config=None):
    from src.model import GPT, GPTConfig

    config_yaml = config or load_config()
    device = device or get_device(config_yaml)
    config_data = ckpt["config"]
    config_obj = GPTConfig(**config_data)
    model = GPT(config_obj).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model

def load_model(path=None, candidate_paths=None, default_path=None, device=None, config=None):
    config_yaml = config or load_config()
    ckpt, checkpoint_path = load_checkpoint(path=path, candidate_paths=candidate_paths, default_path=default_path, config=config_yaml)
    model = build_model_from_checkpoint(ckpt, device=device, config=config_yaml)
    return model, checkpoint_path, ckpt
