# Model Card

## Model

The current model is a word-level CRNN for handwritten date OCR. It combines convolutional image features, a bidirectional LSTM sequence model, and CTC decoding. The vocabulary is made from date-specific tokens: month names and abbreviations, numeric days and months, separators, four-digit years, and two-digit years.

## Intended Use

The model is intended to read a single handwritten date from a cropped or clear image and normalize it to `YYYY-MM-DD` when the predicted text matches a supported date format.

## Training Data

Training data is generated synthetically with `scripts/generate_synthetic_dates.py`. The generator renders date strings using handwriting fonts, then applies scan-like augmentation such as rotation, noise, blur, brightness changes, contrast changes, and sharpening.

## Reported Results

The training notebook reports these results on the synthetic test split:

| Metric | Value |
| --- | ---: |
| Test loss | `0.003233087094966322` |
| Exact match accuracy | `0.996` |
| Character error rate | `0.0004765384248816596` |

These metrics are useful for comparing experiments on the synthetic data pipeline. They should not be treated as real-world accuracy until validated on labeled images from actual scans or camera captures.

## Limitations

- Performance depends on the gap between synthetic handwriting and real user images.
- The UI works best with a single clear date in the image.
- Ambiguous numeric dates such as `04/05/2026` may be interpreted according to the supported parsing order in `app/server.py`.
- The checkpoint file is not stored in Git because it is a large binary artifact.

## Recommended Next Evaluation

Create a small real-image validation set with labels, then report exact-match accuracy, normalized-date accuracy, character error rate, and failure examples by date format.
