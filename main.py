from flask import Flask, request, jsonify
import torch
from transformers import pipeline

app = Flask(__name__)
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16, device_map="auto")

@app.route('/generate-text', methods=['POST'])
def generate_text():
    data = request.get_json()
    
    messages = data.get('messages')
    
    if not messages:
        return jsonify({"error": "No messages provided"}), 400
    
    try:
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        generated_text = outputs[0]["generated_text"]
        
        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)


# curl -X POST 127.0.0.1:5000/generate-text \
# -H "Content-Type: application/json" \
# -d "{\"messages\": [{\"role\": \"system\", \"content\": \"You are a friendly chatbot who always responds in the style of a pirate\"}, {\"role\": \"user\", \"content\": \"How many helicopters can a human eat in one sitting?\"}]}"

