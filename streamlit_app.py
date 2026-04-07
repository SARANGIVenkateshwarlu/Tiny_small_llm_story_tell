import streamlit as st
import torch
import gc

from src.utils import load_model, load_tokenizer, load_config

# Load configuration
config = load_config()

# === STREAMLIT UI ===
st.set_page_config(page_title="Small LLM Inference", page_icon="🧠", layout="wide")

st.title("🧠 Small LLM Text Generator")
st.markdown(
    """
    Welcome to your custom small GPT-style language model inference app!  
    This tool uses the model trained in `llm_fine_tune.ipynb` to generate creative text based on your prompts.  
    Enter a starting sentence, adjust the generation settings, and click **Generate** to see the magic happen.
    """
)

# Sidebar for controls
st.sidebar.header("⚙️ Generation Settings")
st.sidebar.markdown("Adjust these parameters to control the text generation:")

try:
    model, checkpoint_path, ckpt = load_model()
    st.sidebar.success(f"✅ Model Loaded: `{checkpoint_path.name}`")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

# Main content
prompt = st.text_area(
    "📝 Enter your prompt",
    value="Once upon a time",
    height=120,
    help="Start with a sentence to guide the story generation."
)

# Sidebar inputs with config defaults
with st.sidebar:
    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=10,
        max_value=500,
        value=config["generation"]["max_new_tokens"],
        step=10,
        help="Maximum number of tokens to generate (longer = more text)."
    )
    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=config["generation"]["temperature"],
        step=0.1,
        help="Higher values make output more random; lower values make it more focused."
    )
    top_k = st.slider(
        "Top-k",
        min_value=0,
        max_value=200,
        value=config["generation"]["top_k"],
        step=5,
        help="Limits sampling to top-k tokens (0 = no limit)."
    )

# Generate button
if st.button("🚀 Generate Text", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating.")
    else:
        tokenizer = load_tokenizer()
        with st.spinner("🤖 Generating your story..."):
            tokens = tokenizer.encode_ordinary(prompt)
            if len(tokens) == 0:
                generated_text = ""
            else:
                device = next(model.parameters()).device
                input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k if top_k > 0 else None)
                generated_text = tokenizer.decode(output_ids[0].tolist())
        st.success("✨ Generation Complete!")
        st.subheader("📖 Generated Text")
        st.write(generated_text)

st.markdown("---")
st.markdown(
    """
    **About this app:**  
    - Built with Streamlit and PyTorch.  
    - Model: Custom GPT architecture trained on TinyStories dataset.  
    - For best results, use prompts similar to children's stories.  
    - Source: [llm_fine_tune.ipynb](llm_fine_tune.ipynb)
    """
)

# Clean up memory
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
