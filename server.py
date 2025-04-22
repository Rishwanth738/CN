import os
import torch
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from flask_socketio import SocketIO

# Configure absolute paths
BASE_DIR = os.path.abspath(r'C:\Users\rishw\Downloads\CN_TRIAL\CN_TRIAL')
TEMPLATE_PATH = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_PATH)
socketio = SocketIO(app)

# --- Verify Template Path ---
print(f"✅ Verified template path: {app.template_folder}")
print(f"📁 Templates available: {os.listdir(app.template_folder)}")

# --- AI Model Configuration ---
base_model_path = 'aboonaji/llama2finetune-v2'
lora_checkpoint = os.path.join(BASE_DIR, r'fine_tuned_model\fine_tuned_model')  # Fixed path
cache_dir = os.path.join(BASE_DIR, 'model_cache')
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create cache directory
os.makedirs(cache_dir, exist_ok=True)

# --- Load Model ---
print("\n⚙️ Initializing AI Model...")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load base model
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=bnb_config,
    cache_dir=cache_dir,
    device_map="auto"
)

# Check if adapter_config.json exists
adapter_config_path = os.path.join(lora_checkpoint, 'adapter_config.json')
if not os.path.exists(adapter_config_path):
    raise FileNotFoundError(f"⚠️ Missing 'adapter_config.json' at '{adapter_config_path}'")

# Load LoRA adapter
print("Merging LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_checkpoint)
model = model.merge_and_unload()

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    cache_dir=cache_dir
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('homepage.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/counselor')
def counselor():
    return render_template('counselor.html')

@app.route('/ai_therapist')
def ai_therapist():
    return render_template('ai_therapist.html')

@app.route('/talk_to_ai', methods=['POST'])
@app.route('/talk_to_ai', methods=['POST'])
def talk_to_ai():
    user_input = request.json.get('message', '').strip()

    if not user_input:
        return jsonify({"error": "Empty message received"}), 400

    prompt = f"You are an AI therapist. Provide concise and supportive responses.\nUser: {user_input}\nAI:"
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)

    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=120,
        temperature=0.5,
        top_p=0.8,
        no_repeat_ngram_size=2,
        repetition_penalty=1.3,
        do_sample=True,
        early_stopping=True
    )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    ai_response = generated_text.split("AI:", 1)[-1].strip()

    return jsonify({"response": ai_response})

if __name__ == '__main__':
    print("\n🚀 Starting server...")
    print(f"Working directory: {os.getcwd()}")
    print(f"Using device: {device}")

    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)

