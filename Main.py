from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def ml_api():
    return {"Tesing": " First App"}