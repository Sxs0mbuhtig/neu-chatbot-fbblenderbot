# app.py - Optimized for Hugging Face Free CPU Space (2 vCPU, 16 GB RAM)
import sys
import torch
import os
import platform
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --- Configuration (UPGRADED MODEL) ---
# Now using the larger 1.6 GB 1B-distill model for better quality.
MODEL_NAME = "facebook/blenderbot-1B-distill"
MAX_HISTORY_LENGTH = 10 

# --- CPU Thread Optimization (Optional/Platform Default) ---
# Commented out this block to let the Hugging Face environment manage 
# the 2 vCPU threads optimally, which is often the best approach.
# if platform.system() == "Windows":
#     torch.set_num_threads(4) 
# else:
#     os.environ["OMP_NUM_THREADS"] = "4"
#     os.environ["MKL_NUM_THREADS"] = "4"
#     torch.set_num_threads(4)
# print(f"Set CPU processing threads to 4.")


# --- Initialization & Quantization (Runs only once on startup) ---
def initialize_quantized_model(model_name: str):
    print(f"Loading and Quantizing model: **{model_name}**...")
    try:
        device = torch.device("cpu")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Note: Model loading will take longer, but should fit in 16GB RAM.
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # Dynamic quantization (INT8) is critical for running 1B model fast on CPU
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Embedding}, dtype=torch.qint8
        )
        print("Model loaded and quantized successfully.")
        return tokenizer, quantized_model, device
    except Exception as e:
        print(f"🛑 Error loading model: {e}")
        sys.exit(1)

# Global variables for the model
tokenizer, model, device = initialize_quantized_model(MODEL_NAME)

if model is None:
    # Exit cleanly if initialization failed
    sys.exit(1)


def chatbot_interface(input_text, history):
    """
    The main function called by Gradio's chat interface.
    Handles history tokenization and generation.
    """
    if not input_text:
        return "", history

    # 1. Format Conversation History for the Model
    # Flatten history and add current input (Blenderbot's required input format)
    flat_history = [item for sublist in history for item in sublist if item is not None]
    conversation_list = flat_history + [input_text]
    limited_conversation = conversation_list[-MAX_HISTORY_LENGTH:]

    # 2. Model Generation
    try:
        with torch.no_grad():
            # Use prepare_seq2seq_batch for proper dialogue formatting
            inputs = tokenizer.prepare_seq2seq_batch(
                limited_conversation, 
                return_tensors="pt",
                truncation=True
            ).to(device)

            # (UPGRADED GENERATION) Using Beam Search (num_beams=4) for better quality
            outputs = model.generate(
                **inputs,
                max_length=50,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=4,  # Increased beams for quality
                early_stopping=True
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            
        # 3. Update Gradio History
        history.append([input_text, response])
        
        # Return empty string for input box (clears it) and the updated history
        return "", history 
            
    except Exception as e:
        print(f"⚠️ Generation Error: {e}")
        # Show a user-friendly error in a pop-up
        gr.Error(f"Sorry, I ran into an error ({e.__class__.__name__}). Please try again.")
        
        error_message = f"**Error:** Could not generate a response."
        history.append([input_text, error_message])
        return "", history


if __name__ == "__main__":
    print("\n--- Starting Gradio Interface ---")
    
    # Define the custom chatbot component first
    custom_chatbot = gr.Chatbot(
        type='messages',
        value=[] 
    )
    
    # Now define the ChatInterface, passing the custom chatbot component to it
    iface = gr.ChatInterface(
        fn=chatbot_interface,
        title="Optimized Blenderbot 1B Chatbot (CPU)",
        description="Running the larger 1B model with Dynamic Quantization and Beam Search.",
        chatbot=custom_chatbot
    )
    
    # Launch without parameters; Hugging Face Spaces handles hosting.
    print("Version 2.0: Syntax Confirmed Fixed.") # <--- ADD THIS LINE
    iface.queue().launch()