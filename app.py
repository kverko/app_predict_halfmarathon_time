from datetime import timedelta
import os

from dotenv import load_dotenv
import instructor
from langfuse.openai import OpenAI
from langfuse.decorators import observe
import pandas as pd
from pycaret.regression import load_model, predict_model
from pydantic import BaseModel
import streamlit as st

load_dotenv()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
instructor_openai_client = instructor.from_openai(openai_client)


class Runner(BaseModel):
    age: int = -1
    gender: str = ""
    run_time_5km: int = -1
    time_units: str = ""


@observe
def retrieve_structure(text, response_model):
    prompt = f"""
        Extract following information from the provided text:
        1. Age of the person, given in full years
        2. Gender of the person: can be one of: "male", "female"
        3. Time required to run 5km given in seconds
        4. Units of time required to run 5km: can be one of: 
        "hours", "minutes", "seconds"
        Here is the input text:
        '{text}'
        """

    res = instructor_openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_model=response_model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    return res.model_dump()


st.set_page_config(page_title='Predict halfmarathon time', layout='centered')
st.title("Predict halfmarathon time")

st.session_state.setdefault("info", "")
st.session_state.setdefault("age", -1)
st.session_state.setdefault("gender", "")
st.session_state.setdefault("run_time_5km", -1)
st.session_state.setdefault("time_units", "")

info = st.text_area(
    label="I need some info to make the prediction",
    placeholder="please tell me your age, gender "
                "and how long it takes you to run 5km distance",
    value=st.session_state["info"]
)
missing_data = ""
model = load_model("polmaraton_slawekr_model_lr")
if st.button("send") and info:
    resp = None
    with st.spinner("Processing..."):
        resp = retrieve_structure(info, Runner)
    missing_data = ""
    if resp["age"] == -1:
        missing_data += "Age not found. Please provide the age. "
    if resp["gender"] == "":
        missing_data += "Gender not found. Please provide the gender. "
    if resp["run_time_5km"] == -1:
        missing_data += "Time needed to run 5km distance not found. Please provide one. "
    
    if missing_data:
        st.write(missing_data)
    else:
        if resp["time_units"] == "hours":
            resp["run_time_5km"] *= 3600
        elif resp["time_units"] == "minutes":
            resp["run_time_5km"] *= 60
        data = {
            'Płeć': [resp["gender"]],
            'Wiek': [resp["age"]],
            '5 km Czas': [resp["run_time_5km"]]
        }

        for k in [
                'Miejsce', 'Płeć Miejsce', '5 km Miejsce Open', '5 km Tempo', 
                '10 km Czas', '10 km Miejsce Open', '10 km Tempo', '15 km Czas', 
                '15 km Miejsce Open', '15 km Tempo', '20 km Czas', 
                '20 km Miejsce Open', '20 km Tempo', 'Tempo']:
            data[k] = [None]
        data_df = pd.DataFrame(data)
        prediction = predict_model(model, data=data_df)
        predicted_time_s = prediction['prediction_label'][0]
        pred_time_hhmmss = str(timedelta(seconds=predicted_time_s)).split(':')

        p_hh = pred_time_hhmmss[0]
        p_mm = pred_time_hhmmss[1]
        p_ss = pred_time_hhmmss[2]

        st.write(f"Your predicted halfmarathon time is: **{p_hh}:{p_mm}:{p_ss}.**")


    
    
