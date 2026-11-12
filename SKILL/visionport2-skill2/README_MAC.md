# VisionPort 2 – CO1 (Mac setup)

This is a separate copy of the VisionPort starter project. It adds a live horizontal bar graph showing the **actual vehicle counts** returned by YOLO for each uploaded image.

## MacBook Air M2 / 8 GB RAM

Use Python 3.10 or 3.11 if possible.

1. Open this `visionport2-co1` folder in PyCharm.
2. Create/select a project virtual environment named `.venv`.
3. Open PyCharm Terminal and run:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app:app --reload
```

4. Open http://127.0.0.1:8000
5. Choose a traffic JPG/PNG image and click **Detect vehicles**.

The first run downloads `yolov8n.pt` if it is not already cached. Keep the server terminal open while using the website. Stop it with Control+C.

## What the graph means

The graph displays counts for bicycle, car, motorcycle, bus and truck from the current image. It is generated from actual inference output. It is **not** an accuracy graph, training-loss graph, or proof of model performance on a dataset.

## Hardware notes

The app selects Apple MPS when PyTorch reports it available; otherwise it uses CPU. NVIDIA CUDA, TensorRT and NVIDIA Triton are not part of this local Mac starter project. They require a compatible NVIDIA environment.

## Troubleshooting

- If `.venv` does not exist: `python3 -m venv .venv`, then activate it.
- If imports are missing: activate `.venv` and run `pip install -r requirements.txt`.
- If port 8000 is busy: `uvicorn app:app --reload --port 8001`, then open http://127.0.0.1:8001.
- If the page opens but detection fails, read the error shown on the page and in the PyCharm Terminal.
