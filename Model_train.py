import torch
import torchvision
import cv2
import matplotlib
import matplotlib.pyplot as plt

def check_pytorch():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

def check_opencv():
    print("OpenCV version:", cv2.__version__)

def check_matplotlib():
    print("Matplotlib version:", matplotlib.__version__)

def test_imports():
    try:
        import torch
        import torchvision
        import cv2
        import matplotlib
        import matplotlib.pyplot as plt
        print("All libraries imported successfully.")
    except ImportError as e:
        print(f"Import error: {e}")

if __name__ == "__main__":
    print("Testing PyTorch installation...")
    check_pytorch()
    
    print("\nTesting OpenCV installation...")
    check_opencv()
    
    print("\nTesting Matplotlib installation...")
    check_matplotlib()
    
    print("\nTesting all library imports...")
    test_imports()
