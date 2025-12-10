import torch
from transformers import CLIPProcessor

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)

# Create a dummy image tensor (0-1 range)
image_tensor = torch.rand(1, 3, 224, 224)

# Process it
inputs = processor(images=image_tensor, return_tensors="pt")
pixel_values = inputs["pixel_values"]

print(f"Input mean: {image_tensor.mean().item()}")
print(f"Output mean: {pixel_values.mean().item()}")
print(f"Output min: {pixel_values.min().item()}")
print(f"Output max: {pixel_values.max().item()}")

# Check if it rescaled
# CLIP expects normalized values around 0 (mean 0, std 1 approx)
# If it divided by 255, values will be very small (before normalization)
# Normalization: (x - mean) / std.
# If x is tiny, (tiny - mean) / std approx -mean/std.
# So output will be constant-ish.
