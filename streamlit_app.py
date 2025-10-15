# --- CACHING: Model Loading & Quantization (Runs only once) ---
# st.cache_resource is essential for large, immutable objects like models
@st.cache_resource
def initialize_quantized_model(model_name: str):
    """Loads and quantizes the Blenderbot model on first run."""
    try:
        # Streamlit defaults to CPU, no need to specify device unless using GPU
        device = torch.device("cpu")
        
        # 1. Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # 2. Dynamic Quantization (INT8) for CPU performance
        # Define the custom quantization configuration map to resolve the Embedding error.
        # Embedding layers require 'float_qparams_weight_only_qconfig'.
        qconfig_dict = {
            nn.Embedding: float_qparams_weight_only_qconfig,
            nn.Linear: default_dynamic_qconfig
        }
        
        # This converts floating-point weights to 8-bit integers using the custom configuration.
        quantized_model = torch.quantization.quantize_dynamic(
            model, qconfig_dict, dtype=torch.qint8
        )
        st.success("Model loaded and quantized successfully!", icon="✅")
        return tokenizer, quantized_model, device
    except Exception as e:
        st.error(f"🛑 Error loading model: {e}")
        st.stop() # Stops the app if model loading fails