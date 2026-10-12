# APAN5560 Assignments 1 and 2

Linyi Tian | lt3039

This project uses FastAPI for bigram text generation, spaCy word
embeddings, and CNN image classification.

Repository: https://github.com/Evelynntian/APAN5560_Assignment1_LinyiTian

## Run locally

Use Python 3.12 with uv installed. From the project folder, run:

```bash
uv sync --frozen
uv run fastapi dev app/main.py
```

Open http://127.0.0.1:8000/docs

## Run with Docker

Start Docker Desktop. From the project folder, run:

```bash
docker build -t sps-genai-assignment2 .
docker run --rm -p 127.0.0.1:8001:80 sps-genai-assignment2
```

Open http://127.0.0.1:8001/docs

The build needs internet access to download dependencies.
Linux uses CPU versions of PyTorch and torchvision.

The trained CNN weights are included in `models/cnn.pt`.
The API loads these weights for prediction, so no training or
CIFAR-10 download is needed to use the classification endpoint.

## Endpoints

- `GET /`: returns a greeting.
- `POST /generate`: generates text using bigram probabilities.
- `GET /embedding?word=apple`: returns the word and its full
  300-dimensional embedding using spaCy's `en_core_web_lg` model.
- `POST /classify`: accepts an uploaded image and returns its
  predicted CIFAR-10 class ID and class name.

### Text generation

Example JSON body for `/generate`:

```json
{
  "start_word": "the",
  "length": 20
}
```

Text generation may stop early if no next word is available.
Words without a vector in the spaCy model may return a zero vector.

### Image classification

In the API documentation:

1. Expand `POST /classify`.
2. Click **Try it out**.
3. Choose `sample_images/test_0.png`.
4. Click **Execute**.

You can also run this command from the project folder while
the Docker server is running:

```bash
curl -X POST \
  "http://127.0.0.1:8001/classify" \
  -F "file=@sample_images/test_0.png"
```

The included sample is a cat from the CIFAR-10 test set.
The tested response was:

```json
{
  "class_id": 3,
  "class_name": "cat"
}
```

Class IDs follow this order:

| ID | Class |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

Uploaded images are converted to RGB, resized to 64 × 64,
and converted to tensors with pixel values from 0 to 1.
Training uses the same resizing and tensor conversion.

The model always predicts one of these ten classes.
Images outside these categories may receive an incorrect label.
Unreadable image files receive HTTP 400.

## CNN architecture

The model follows the Assignment 2 specification.

Shapes below exclude the batch dimension and use
channels × height × width.

| Layer | Settings | Output shape |
|---|---|---|
| Input | RGB image | 3 × 64 × 64 |
| Convolution 1 + ReLU | 16 filters, kernel 3, stride 1, padding 1 | 16 × 64 × 64 |
| Max pooling 1 | Kernel 2, stride 2 | 16 × 32 × 32 |
| Convolution 2 + ReLU | 32 filters, kernel 3, stride 1, padding 1 | 32 × 32 × 32 |
| Max pooling 2 | Kernel 2, stride 2 | 32 × 16 × 16 |
| Flatten | 32 × 16 × 16 | 8192 |
| Fully connected + ReLU | 8192 → 100 | 100 |
| Fully connected | 100 → 10 | 10 |

The final layer returns logits. Training uses cross-entropy loss,
and prediction selects the class with the largest logit.

## Training and results

The model was trained on the 50,000 CIFAR-10 training images
and evaluated on the 10,000 test images after training.

| Setting | Value |
|---|---|
| Epochs | 10 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 0.0005 |
| Random seed | 42 |
| Final training accuracy | 72.65% |
| Test accuracy | 65.40% |

Weights are stored in `models/cnn.pt`.
Training history and test metrics are stored in
`models/training_results.json`.

To train again, run:

```bash
uv run python -m app.train_cnn
```

This downloads CIFAR-10 if needed and replaces the saved weights
and training results. Retraining is optional for running the API.
Results may vary across devices and runs.

The training script uses MPS when available, then CUDA,
and otherwise CPU. API inference runs on CPU.

## Project files

- `app/main.py`: FastAPI endpoints.
- `app/bigram_model.py`: bigram text generation.
- `app/embedding_model.py`: spaCy word embeddings.
- `app/cnn_model.py`: assignment CNN architecture.
- `app/cnn_data.py`: CIFAR-10 loaders, preprocessing, and class names.
- `app/cnn_predict.py`: loads saved weights and predicts image classes.
- `app/train_cnn.py`: training configuration and results export.
- `helper_lib/`: shared data loading, model creation, training,
  evaluation, and model saving functions.
- `models/`: trained weights and recorded results.
- `sample_images/test_0.png`: sample image for testing the API.
- `Dockerfile`: container setup.
- `pyproject.toml` and `uv.lock`: dependencies and locked versions.

The current helper model factory supports `get_model("CNN")`.
The training script calls the helper functions.
Downloaded CIFAR-10 data and the local virtual environment are
excluded from Git and the Docker build.

## Checks completed

- Local `/embedding` returned HTTP 200 and a 300-dimensional vector.
- Local `/generate` returned HTTP 200 and generated text.
- Docker `/embedding` returned HTTP 200 during Assignment 1 testing.
- The Assignment 2 Docker image built and started successfully.
- Local and Docker `/classify` returned HTTP 200 with class `cat`
  for the included sample image.
- The helper functions imported successfully.
- The CNN loaded the saved weights and returned output shape
  `(1, 10)` for an input shape of `(1, 3, 64, 64)`.
- The updated training script imported successfully.