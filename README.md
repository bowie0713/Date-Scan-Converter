# Date Scan Converter

Date Scan Converter reads a handwritten date from an uploaded image and returns a normalized `YYYY-MM-DD` value. The project combines synthetic handwriting data generation, a CRNN OCR model trained with CTC loss, and a small FastAPI + HTML interface for local inference.

## Repository Layout

```text
.
├── app/
│   ├── front_end.html              # Browser UI for image upload and prediction
│   └── server.py                   # FastAPI inference API
├── docs/
│   └── MODEL_CARD.md               # Model details, metrics, and limitations
├── notebooks/
│   └── new_model_training_date_converter.ipynb
├── scripts/
│   └── generate_synthetic_dates.py # Synthetic handwritten date image generator
├── requirements.txt
└── README.md
```

Large generated assets are intentionally ignored by Git. Keep checkpoints, generated images, extracted datasets, and font caches in local folders such as `models/`, `data/`, `output/`, `fonts/`, or `checkpoints/`.

## Workflow

1. Generate synthetic date images with varied formats, handwriting fonts, paper colors, ink colors, rotation, noise, blur, brightness, and contrast.
2. Train the CRNN model in the notebook on the generated image/label pairs.
3. Save the best checkpoint locally as a `.pt` file.
4. Start the FastAPI server with that checkpoint.
5. Open the HTML UI, upload a date image, and receive both the raw OCR output and normalized date.

## Generate Synthetic Data

```bash
python scripts/generate_synthetic_dates.py --n 500 --output output --fonts fonts --seed 42
```

The generator writes:

```text
output/
├── images/
│   ├── 00000.png
│   └── ...
└── labels.tsv
```

## Train The Model

Use `notebooks/new_model_training_date_converter.ipynb` as the current training notebook. It expects the generated images and labels to be available in the paths configured inside the notebook. The notebook trains a word-level CRNN with CTC decoding over date tokens such as month names, separators, days, and years.

The latest notebook output reports the following synthetic test-set metrics:

| Metric | Value |
| --- | ---: |
| Test loss | `0.003233087094966322` |
| Exact match accuracy | `0.996` |
| Character error rate | `0.0004765384248816596` |

These results are from the synthetic held-out split shown in the notebook. Real-world camera or scan performance should be validated separately with a labeled set of real handwritten dates.

## Run Local Inference

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the trained checkpoint in `models/best_crnn_30k_75ep_dylan.pt`, or point the server to another local checkpoint:

```bash
MODEL_CHECKPOINT=/path/to/best_crnn_30k_75ep_dylan.pt uvicorn app.server:app --reload
```

Then open `app/front_end.html` in a browser. The UI sends uploads to:

```text
http://127.0.0.1:8000/predict
```

## API

Health check:

```bash
curl http://127.0.0.1:8000/
```

Predict a date from an image:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@sample_date.jpg"
```

Example response:

```json
{
  "raw": "13-04-1995",
  "date": "1995-04-13",
  "normalized": true
}
```

## Notes For GitHub Hygiene

- Do not commit `.pt`, `.pth`, `.ckpt`, generated images, zip files, or local dataset folders.
- If a model checkpoint needs to be shared, use GitHub Releases, cloud storage, or Git LFS instead of committing it directly.
- Keep notebook outputs that document useful results, but clear large cell outputs before committing if the notebook grows substantially.
