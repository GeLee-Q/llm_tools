import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWithHooks:
    def __init__(self, model):
        self.model = model
        self.layer_outputs = {}

        # Register hooks for each layer
        for name, module in self.model.named_modules():
            if "layer" in name or "block" in name or "transformer" in name:
                module.register_forward_hook(self.get_layer_hook(name))

    def get_layer_hook(self, layer_name):
        def hook(module, input, output):
            if input and len(input) > 0 and isinstance(input[0], torch.Tensor) and input[0].numel() > 0:
                input_data = input[0]
                input_mean = input_data.mean().item()
                input_sum = input_data.sum().item()
                input_shape = input_data.shape
                print(f"Layer {layer_name} input - mean: {input_mean}, sum: {input_sum}, shape: {input_shape}")
            else:
                print(f"Layer {layer_name} input is None, empty or not a tensor.")
            
            # Check if output is not None and has elements
            if output is not None and isinstance(output[0], torch.Tensor) and output[0].numel() > 0:
                output_data = output[0]
                output_mean = output_data.mean().item()
                output_sum = output_data.sum().item()
                output_shape = output_data.shape
                print(f"Layer {layer_name} output - mean: {output_mean}, sum: {output_sum}, shape: {output_shape}")
            else:
                print(f"Layer {layer_name} output is None or not a tensor.")
        return hook

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/workspace/internlm-7b-hf", trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "/workspace/internlm-7b-hf/",
    torch_dtype=torch.float16,
    trust_remote_code=True
).cuda()

# Wrap the model with hooks
model_with_hooks = ModelWithHooks(model)

# Tokenize input
input_text = "This is a test sentence."
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')

# Forward pass
outputs = model_with_hooks.forward(**inputs)
