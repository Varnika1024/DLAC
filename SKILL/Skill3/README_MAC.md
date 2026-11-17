# VisionPort — Skill 3 (Mac / PyCharm)

This folder demonstrates one complete inference request using one traffic image.
It prints five stages in the PyCharm Run window:

1. API input
2. Preprocessing
3. Model execution
4. Postprocessing
5. API response

## Run on your MacBook Air M2

1. Open the `Skill3` folder in PyCharm.
2. Open PyCharm Terminal.
3. Create a virtual environment:

   ```bash
   python3 -m venv .venv
   ```

4. Activate it:

   ```bash
   source .venv/bin/activate
   ```

5. Install the packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the server:

   ```bash
   uvicorn app:app --reload
   ```

7. Open this address in your browser:

   http://127.0.0.1:8000

8. Choose your traffic image and click **Run vehicle detection**.

9. Return to PyCharm. Read the five trace stages printed in the Run/Terminal window.

The first run downloads the pretrained `yolov8n.pt` model automatically. Keep an internet connection for this first download.

## Useful links

- App: http://127.0.0.1:8000
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Important

- One image is enough to demonstrate one inference request.
- One image is not enough to train a reliable model or measure general accuracy.
- This starter uses PyTorch with Apple MPS when available, otherwise CPU.
- Triton, TensorRT and NVIDIA CUDA are not part of this Mac implementation.
- The model detects COCO vehicle classes: bicycle, car, motorcycle, bus and truck.
- It is a general pretrained model, not a model specifically trained for every Indian vehicle type.
