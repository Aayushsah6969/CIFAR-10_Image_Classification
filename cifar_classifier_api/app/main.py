from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

from .utils import preprocess_image
from .predictor import predict

from fastapi.staticfiles import StaticFiles

app = FastAPI()
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": None}
    )


@app.post("/predict")
async def classify_image(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)

    prediction = predict(processed)

    return {"prediction": prediction}