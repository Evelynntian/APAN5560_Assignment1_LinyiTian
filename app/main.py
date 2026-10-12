from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.bigram_model import BigramModel
from app.cnn_predict import predict_image
from app.embedding_model import calculate_embedding


app = FastAPI(title="APAN5560 Assignments 1 and 2 API")

corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantes, who is falsely imprisoned "
    "and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1, max_length=100)
    length: int = Field(default=20, ge=1, le=200)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    start_word = request.start_word.strip()
    if not start_word or len(start_word.split()) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please enter one starting word.",
        )

    generated_text = bigram_model.generate_text(
        start_word, request.length
    )
    return {"generated_text": generated_text}


@app.get("/embedding")
def get_embedding(
    word: str = Query(..., min_length=1, max_length=100),
):
    word = word.strip()
    if not word or len(word.split()) != 1:
        raise HTTPException(
            status_code=400,
            detail="Please enter one word.",
        )

    embedding = calculate_embedding(word)
    return {
        "word": word,
        "dimension": len(embedding),
        "embedding": embedding,
    }


@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    try:
        with Image.open(file.file) as image:
            image.load()
            rgb_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image, such as PNG or JPEG.",
        )
    finally:
        file.file.close()

    return predict_image(rgb_image)