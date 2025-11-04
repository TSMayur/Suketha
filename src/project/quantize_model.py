from optimum.onnxruntime import ORTQuantizer, ORTModelForFeatureExtraction
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from pathlib import Path

onnx_model_path = Path("./models/all-mpnet-base-v2-onnx")
quantized_model_path = Path("./models/all-mpnet-base-v2-onnx-int8")

print("Loading ONNX model...")
model = ORTModelForFeatureExtraction.from_pretrained(onnx_model_path)

print("Applying INT8 dynamic quantization...")
quantizer = ORTQuantizer.from_pretrained(model)

# Dynamic quantization config (no calibration data needed)
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# Quantize
quantizer.quantize(
    save_dir=quantized_model_path,
    quantization_config=qconfig
)

print(f"✅ Quantized model saved to {quantized_model_path}")
