from fastapi import FastAPI
from pydantic import BaseModel
from functions_for_api import classify_fastapi


class InputData(BaseModel):
    Age:int
    Sex:str
    ChestPainType:str
    RestingBP:int
    Cholesterol:int
    FastingBS:int
    RestingECG:str
    MaxHR:int
    ExerciseAngina:str
    Oldpeak:float
    ST_Slope:str


app = FastAPI()


# Define the root endpoint to return the app description
@app.get("/")
async def root():
	return {"message": "use /predict/ with curl to get output from the model"}


@app.post("/predict/")
async def predict(data:InputData):
    data.dict()
    # heart_disease = classify_fastapi([data.Age, data.Sex, data.ChestPainType, data.RestingBP, data.Cholesterol,
    #                                   data.FastingBS, data.RestingECG, data.MaxHR, data.ExerciseAngina, data.Oldpeak,
    #                                   data.ST_Slope])

    heart_disease = classify_fastapi(data.dict())
    
    return {"heart_disease": heart_disease}


## Example of CURL
# curl -X 'POST' \
#   'http://localhost:8000/predict/' \
#   -H 'Content-Type: application/json' \
#   -d '{
#         "Age":"40",
#         "Sex":"M",
#         "ChestPainType":"ATA",
#         "RestingBP":"140",
#         "Cholesterol":"289",
#         "FastingBS":"0",
#         "RestingECG":"Normal",
#         "MaxHR":"172",
#         "ExerciseAngina":"N",
#         "Oldpeak":"0.0",
#         "ST_Slope":"Up"
#     }'