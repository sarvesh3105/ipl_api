import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json


app= FastAPI()

class ScoringItem(BaseModel):
    batting_team: str
    bowling_team: str
    city: str
    total_runs_x: int
    Runs_Scored: int
    overs : int
    wickets: int

with open('pipe_ipl1.pkl','rb') as f:
    model = pickle.load(f)

@app.post('/')

async def scoring_endpoint(item:ScoringItem):
    input_data = item.json()
    input_dictionary = json.loads(input_data)
    batting=input_dictionary["batting_team"]
    bowling=input_dictionary["bowling_team"]
    selected_city=input_dictionary["city"]
    target=input_dictionary["total_runs_x"]
    score=input_dictionary["Runs_Scored"]
    overs=input_dictionary["overs"]
    wickets=input_dictionary["wickets"]

    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    crr = score / overs
    rr = (runs_left * 6) / balls_left

    df=pd.DataFrame({'batting_team': [batting], 'bowling_team': [bowling], 'city': [selected_city],
                             'total_runs_x': [target], 'runs_left': [runs_left], 'balls_left': [balls_left],
                             'wickets': [wickets_left], 'crr': [crr], 'rr': [rr]})
    result = model.predict_proba(df)
    loss = result[0][0]
    win = result[0][1]
    return {"batting_team":int(win*100),"bowling_team":int(loss*100)}

