import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle



app= FastAPI()

class ScoringItem(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    total_runs_x: int
    runs_left: int
    balls_left: int
    wickets: int
    crr: float
    rr: float

with open('pipe_ipl1.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')

async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    result = model.predict_proba(df)
    loss = result[0][0]
    win = result[0][1]
    return {"batting_team":int(loss*100),"bowling_team":int(win*100)}

