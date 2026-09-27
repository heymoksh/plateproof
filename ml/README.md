# Optional: experimental AI-image classifier

The main application does **not** need this. It is here so the project can
include a trained model later, with honest evaluation.

## What it is

An EfficientNet classifier fine-tuned to separate real photos from
AI-generated images. When a trained model is present, the app shows its
output as an **advisory** result. It never changes the risk score.

## Why it is advisory only

The suggested training data, CIFAKE, contains 32x32-pixel CIFAR-10 photos
and Stable Diffusion 1.4 images. Refund photos are large phone
photos of food and packaging, and today's generators differ from SD 1.4.
A good CIFAKE test score therefore says little about food-delivery refund photos. Before
letting a classifier influence decisions you would need a labelled set of
real refund-style photos and images from current generators, and you would
evaluate on it.

## Train it

1. Install PyTorch (CPU is fine for a small run; a GPU, e.g. Google Colab, is faster):

   ```
   pip install -r requirements-ml.txt --index-url https://download.pytorch.org/whl/cpu
   ```

2. Download CIFAKE from Kaggle
   (https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
   and extract it so that you have `ml/data/CIFAKE/train/FAKE`, `.../train/REAL`,
   `.../test/FAKE`, `.../test/REAL`.

3. Train (a quick run first, then a full one):

   ```
   python ml/train_detector.py --data ml/data/CIFAKE --epochs 2 --limit-per-class 2000
   python ml/train_detector.py --data ml/data/CIFAKE --epochs 5
   ```

This writes `models/detector.pt` and `models/detector.json`. The JSON file
records the class order, preprocessing and the held-out test metrics. Restart
the app and the classifier appears under "Experimental classifier".

## Notes

- The class order is saved and looked up by name, so the "AI-generated"
  probability cannot be read from the wrong output.
- Validation data is split from `train/`; `test/` is used once, at the end.
