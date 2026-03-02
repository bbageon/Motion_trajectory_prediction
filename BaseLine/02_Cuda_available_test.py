# cuda_test.py
import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("CUDA test FAILED (GPU를 못 찾음)")
        return

    device = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0))

    a = torch.arange(1024, dtype=torch.float32, device=device)
    b = torch.arange(1024, dtype=torch.float32, device=device) * 2
    c = a + b

    expected = torch.arange(1024, dtype=torch.float32, device=device) * 3
    ok = torch.allclose(c, expected)

    print("CUDA test PASSED" if ok else "CUDA test FAILED")

if __name__ == "__main__":
    main()
