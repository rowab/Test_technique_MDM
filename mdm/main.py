import pickle
import logging
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List

app = FastAPI(title="MDM Classifier API", description="API for MDM classification using ML")


class Predict(BaseModel):
    data: List[conlist(float, min_items=4, max_items=4)]


@app.post("/api", tags=["prediction"])

async def prediction(predict: Predict):
    
    data=predict.dict()
    classifier  = pickle.load(open('classifier.pkl', "rb"))
    data_in = [
        [
            data["weight"],
            data["height"],
            data["width"],
            data["depth"],
        ]
    ]
    prediction = classifier.predict([data_in])
    print(prediction)
    return prediction

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=53405, log_level="info", reload=True)
    