# app.py - Optimized for Streamlit Community Cloud
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy
import sentencepiece

# --- Configuration ---
MODEL_NAME = "facebook/blenderbot-1B-distill"
MAX_HISTORY_LENGTH = 10 

st.set_page_config(
    page_title="Streamlit Blenderbot Chatbot",
    layout="wide"
)

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
        # This converts floating-point weights to 8-bit integers.
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8
        )
        st.success("Model loaded and quantized successfully!", icon="✅")
        return tokenizer, quantized_model, device
    except Exception as e:
        st.error(f"🛑 Error loading model: {e}")
        st.stop() # Stops the app if model loading fails

# Global variables for the model
tokenizer, model, device = initialize_quantized_model(MODEL_NAME)


# --- Initialization: Streamlit Session State ---
# This ensures chat history and message keys persist across interactions
if "messages" not in st.session_state:
    st.session_state.messages = [] # Format: [{"role": "user", "content": "..."}]

if "history_list" not in st.session_state:
    st.session_state.history_list = [] # Format: Flattened list of prior messages for the model


# --- Core Chatbot Generation Function ---
def generate_response(input_text):
    """Handles history, generation, and updates session state."""
    
    # 1. Format Conversation History for the Model
    # Append the new user input to the model's history list
    st.session_state.history_list.append(input_text)
    
    # Limit conversation length to prevent hitting token limits
    limited_conversation = st.session_state.history_list[-MAX_HISTORY_LENGTH:]

    # 2. Model Generation
    try:
        with torch.no_grad():
            # Use prepare_seq2seq_batch for proper dialogue formatting
            inputs = tokenizer.prepare_seq2seq_batch(
                limited_conversation, 
                return_tensors="pt",
                truncation=True
            ).to(device)

            # Use simple pad_token_id definition (Fix for a common error)
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
st.title("Optimized Blenderbot 1B Chatbot (Streamlit CPU)")
st.caption("Running the larger 1B model with Dynamic Quantization and Beam Search.")

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