import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# Load the PyTorch model with weights
model_path = '../Pytorch_Weights/DeepLabV3Plus_resnet101.pth'

# Set model params
ENCODER = 'resnet101'
ENCODER_WEIGHTS = 'imagenet'
ACTIVATION = 'softmax'
# Cityscapes dataset has 66 classes
NUM_CLASSES = 66

# Create an instance of the model
model = smp.DeepLabV3Plus(
    encoder_name=ENCODER,
    encoder_weights=ENCODER_WEIGHTS,
    classes=NUM_CLASSES,
    activation=ACTIVATION)

# Load the model
model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model on the CPU

# Unwrap the model from DataParallel if it was wrapped
if isinstance(model, torch.nn.DataParallel):
    model = model.module
print("Dataparallel unravelled")

model.eval()

# Example input dimensions
input_channels = 3  # Change according to your input image channels
input_height = 543  # Change according to your input image height
input_width = 543  # Change according to your input image width

# Pad the input image dimensions to be divisible by 16
target_height = ((input_height - 1) // 16 + 1) * 16
target_width = ((input_width - 1) // 16 + 1) * 16
pad_top = (target_height - input_height) // 2
pad_bottom = target_height - input_height - pad_top
pad_left = (target_width - input_width) // 2
pad_right = target_width - input_width - pad_left

print("padding done")
# Dummy input for tracing the model
dummy_input = torch.randn(1, input_channels, input_height, input_width)

# Apply padding to the input
padded_input = F.pad(dummy_input, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

# Export the model to ONNX format
onnx_path = 'ONNX_Weights/DeepLabV3Plus_resnet101.onnx'
print("Exporting the model to ONNX...")
torch.onnx.export(model, padded_input, onnx_path, verbose=True, opset_version=11, input_names=["input"], output_names=["output"])
print(f"Model exported to: {onnx_path}")