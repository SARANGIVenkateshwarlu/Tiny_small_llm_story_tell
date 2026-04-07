from src.utils import get_device, load_model, load_tokenizer

model, checkpoint_path, ckpt = load_model()

device = get_device()

tokenizer = load_tokenizer()

