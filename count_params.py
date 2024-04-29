from model import *

def count_parameters_in_GB(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_in_GB = total_params * 4 / (1024 ** 3)  # Convert bytes to GB
    return total_params_in_GB

device = "cuda"
REN = RainEstimationNetwork().to(device)
RCN = RainEstimationNetwork().to(device)
DN = RainEstimationNetwork().to(device)
enc = Encoder().to(device)
dec = Decoder().to(device)

print(f"Memory used by REN in GB: {count_parameters_in_GB(REN)}")
print(f"Memory used by RCN in GB: {count_parameters_in_GB(RCN)}")
print(f"Memory used by DN in GB: {count_parameters_in_GB(DN)}")
print(f"Memory used by Encoder in GB: {count_parameters_in_GB(enc)}")
print(f"Memory used by Decoder in GB: {count_parameters_in_GB(dec)}")

