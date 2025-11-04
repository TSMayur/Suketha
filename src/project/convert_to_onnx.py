from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from pathlib import Path

# Convert sentence-transformers model to ONNX
model_name = "sentence-transformers/all-mpnet-base-v2"
output_dir = Path("./models/all-mpnet-base-v2-onnx")

print("Converting to ONNX...")
model = ORTModelForFeatureExtraction.from_pretrained(
    model_name,
    export=True  # This triggers ONNX conversion
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save ONNX model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ ONNX model saved to {output_dir}")
