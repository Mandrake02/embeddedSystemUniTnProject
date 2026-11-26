import torch

def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")

if (__name__ == "__main__"):
  device = get_device()
  print(f"Detected device: {device}")