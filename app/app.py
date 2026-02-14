"""
🩸 MedGemma Leukemia Detection - Gradio App
Fine-tuned model: good2idnan/medgemma-1.5-4b-it-leukemia-lora (V2)
Accuracy: 78.15% | Leukemia Recall: 83.10%
"""

import os
import torch
import gradio as gr
from PIL import Image
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from huggingface_hub import login

# Load environment variables
load_dotenv()

# Model IDs - V2 Model (78.15% accuracy)
BASE_MODEL_ID = "google/medgemma-1.5-4b-it"
LORA_ADAPTER_ID = "good2idnan/medgemma-1.5-4b-it-leukemia-lora"

# Global variables for model
model = None
processor = None


def load_model():
    """Load the fine-tuned MedGemma model"""
    global model, processor
    
    # Login to HuggingFace
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("✅ Logged in to HuggingFace")
    else:
        print("⚠️ HF_TOKEN not found. Set it in .env file")
        return False
    
    print(f"🔄 Loading model: {LORA_ADAPTER_ID}")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📍 Using device: {device}")
    
    if device == "cpu":
        print("⚠️ WARNING: Running on CPU will be slow (~30s per image)")
    
    # Load processor from base model
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, token=hf_token)
    
    # Load base model
    base_model = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        token=hf_token
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID, token=hf_token)
    model.eval()
    
    if device == "cpu":
        model = model.to(device)
    
    print("✅ Model loaded successfully!")
    return True


def predict(image):
    """Predict leukemia from blood cell image"""
    global model, processor
    
    if model is None or processor is None:
        return "❌ Model not loaded. Please check HF_TOKEN in .env file.", None
    
    if image is None:
        return "Please upload a blood cell microscopy image.", None
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image).convert("RGB")
    else:
        image = image.convert("RGB")
    
    # Create message with EXACT prompt used during V2 training
    prompt = (
        "Analyze this blood cell microscopy image and classify it.\n"
        "Is the cell NORMAL or LEUKEMIA (blast)?\n"
        "Answer with exactly one of: Normal, Leukemia.\n"
    )
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    # Apply chat template
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Process inputs
    inputs = processor(
        images=image,
        text=text,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
        )
    
    # Decode response
    response = processor.decode(outputs[0], skip_special_tokens=True)
    
    # Extract classification - get last line
    last_line = response.strip().split('\n')[-1].strip().lower()
    
    # Determine result and confidence display
    if "normal" in last_line and "leukemia" not in last_line:
        result = "✅ **NORMAL**"
        details = "Healthy blood cells detected"
        confidence_html = """
        <div style="background: linear-gradient(90deg, #22c55e 70%, #e5e5e5 70%); 
                    height: 20px; border-radius: 10px; margin: 10px 0;"></div>
        <p style="text-align: center; color: #22c55e; font-weight: bold;">Confidence: High</p>
        """
    elif "leukemia" in last_line:
        result = "⚠️ **LEUKEMIA DETECTED**"
        details = "Acute Lymphoblastic Leukemia (ALL) blast cells detected"
        confidence_html = """
        <div style="background: linear-gradient(90deg, #ef4444 83%, #e5e5e5 83%); 
                    height: 20px; border-radius: 10px; margin: 10px 0;"></div>
        <p style="text-align: center; color: #ef4444; font-weight: bold;">Model Recall: 83.1%</p>
        """
    else:
        result = f"🔍 **UNCERTAIN**"
        details = f"Model response: {last_line}"
        confidence_html = ""
    
    output_text = f"""
## {result}

{details}

{confidence_html}

---
**⚠️ Disclaimer:** This is an AI screening tool. Results should be confirmed by a qualified hematologist.
"""
    
    return output_text


# Create Gradio interface
def create_app():
    """Create and return the Gradio app"""
    
    with gr.Blocks(title="MedGemma Leukemia Detection", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🩸 LeukemiaScope - AI Blood Cell Analysis
        
        Upload a blood cell microscopy image to detect Acute Lymphoblastic Leukemia (ALL).
        
        | Metric | Value |
        |--------|-------|
        | **Model** | MedGemma 1.5 4B (Fine-tuned) |
        | **Accuracy** | 78.15% |
        | **Leukemia Recall** | 83.10% |
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Upload Blood Cell Image",
                    type="pil",
                    height=350
                )
                predict_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
                
                gr.Markdown("""
                ### Sample Images
                Try with images from the [C-NMC Dataset](https://www.kaggle.com/datasets/andrewmvd/leukemia-classification)
                """)
            
            with gr.Column(scale=1):
                output = gr.Markdown(
                    label="Prediction Result",
                    value="*Upload an image and click Analyze*"
                )
        
        predict_btn.click(
            fn=predict,
            inputs=image_input,
            outputs=output
        )
        
        gr.Markdown("""
        ---
        ## About This Tool
        
        **LeukemiaScope** is an AI-powered screening tool that analyzes microscopic blood cell images 
        to detect signs of Acute Lymphoblastic Leukemia (ALL).
        
        ### How It Works
        1. Upload a blood cell microscopy image (single cell, centered)
        2. Click "Analyze Image"
        3. View the AI prediction
        
        ### Clinical Context
        - Early detection of leukemia improves 5-year survival from 20% to 85%
        - This tool is designed to assist healthcare workers in resource-limited settings
        - All positive results should be confirmed with laboratory tests
        
        ---
        **MedGemma Impact Challenge 2026** | Built with Google MedGemma
        """)
    
    return demo


if __name__ == "__main__":
    print("=" * 50)
    print("🩸 LeukemiaScope - MedGemma Leukemia Detection")
    print("=" * 50)
    
    # Load model
    if load_model():
        # Create and launch app
        app = create_app()
        app.launch(share=True)  # share=True creates public link
    else:
        print("❌ Failed to load model. Check your HF_TOKEN.")
