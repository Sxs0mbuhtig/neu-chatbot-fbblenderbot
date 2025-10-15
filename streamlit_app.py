# app.py - Optimized for Streamlit Community Cloud
import streamlit as st
import torch
import torch.nn as nn # <-- ADDED: Needed for nn.Linear and nn.Embedding
from torch.ao.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig # <-- ADDED: Needed for quantization fix
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy
import sentencepiece

# --- Configuration ---
MODEL_NAME = "facebook/blenderbot-90M" # Use a smaller model for faster loading and less memory strain
MAX_HISTORY_LENGTH = 10 

st.set_page_config(
    page_title="Streamlit Blenderbot Chatbot",
    layout="wide"
)

# --- CACHING: Model Loading & Quantization (Runs only once) ---
@st.cache_resource
def initialize_quantized_model(model_name: str):
    """Loads and quantizes the Blenderbot model on first run."""
    try:
        # Streamlit defaults to CPU
        device = torch.device("cpu")
        
        # 1. Load Tokenizer and Model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # 2. Dynamic Quantization (INT8) for CPU performance
        
        # Define the custom qconfig map to resolve the Embedding error:
        # "Embedding quantization is only supported with float_qparams_weight_only_qconfig"
        qconfig_dict = {
            nn.Embedding: float_qparams_weight_only_qconfig,
            nn.Linear: default_dynamic_qconfig
        }
        
        # Apply quantization using the corrected config
        quantized_model = torch.quantization.quantize_dynamic(
            model, qconfig_dict, dtype=torch.qint8 # <-- FIXED: Passing the required qconfig_dict
        )
        st.success("Model loaded and quantized successfully!", icon="✅")
        return tokenizer, quantized_model, device
    except Exception as e:
        # Using st.exception gives a better traceback in the Streamlit UI
        st.error(f"🛑 Error loading model: {e}")
        st.exception(e)
        st.stop() # Stops the app if model loading fails

# Global variables for the model
# NOTE: Removed the redundant 'get_model()' call which was causing conflicts.
tokenizer, model, device = initialize_quantized_model(MODEL_NAME)


# --- Initialization: Streamlit Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = [] 

if "history_list" not in st.session_state:
    st.session_state.history_list = [] 


# --- Core Chatbot Generation Function ---
def generate_response(input_text):
    """Handles history, generation, and updates session state."""
    
    # 1. Format Conversation History for the Model
    st.session_state.history_list.append(input_text)
    
    limited_conversation = st.session_state.history_list[-MAX_HISTORY_LENGTH:]

    # 2. Model Generation
    try:
        with torch.no_grad():
            inputs = tokenizer.prepare_seq2seq_batch(
                limited_conversation, 
                return_tensors="pt",
                truncation=True
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_length=50,
                pad_token_id=tokenizer.pad_token_id, 
                num_beams=4,
                early_stopping=True
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
            # 3. Update Model History and Return
            st.session_state.history_list.append(response)
            return response
            
    except Exception as e:
        st.session_state.history_list.pop() # Remove failed user input
        st.error(f"⚠️ Generation Error: {e.__class__.__name__}. Check the console logs.")
        return f"Sorry, I ran into a system error. ({e.__class__.__name__})"

# --- Streamlit Interface Setup ---
st.title("Optimized Blenderbot Chatbot (Streamlit CPU)")
st.caption("Running the 90M model with Dynamic Quantization and Beam Search.")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Say hello!"):
    # 1. Add user message to display state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            full_response = generate_response(prompt)
            st.markdown(full_response)
    
    # 3. Add assistant response to display state
    st.session_state.messages.append({"role": "assistant", "content": full_response})