from dotenv import dotenv_values
from openai import OpenAI
import instructor
from pydantic import BaseModel
import streamlit as st
env = dotenv_values(".env")

openai_client = OpenAI(api_key=env["OPENAI_API_KEY"])

instructor_openai_client = instructor.from_openai(openai_client)


class Runner(BaseModel):
    age: int
    gender: str
    run_time_5km: int
    time_units: str


def retrieve_structure(text, response_model):
    st.write(text)
    prompt = f"""
        Extract following information from the provided text:
        1. Age of the person, given in full years or None
        2. Gender of the person: can be one of: "male", "female", None
        3. Time required to run 5km given in seconds or None
        4. Units of time required to run 5km: can be one of: 
        "hours", "minutes", "seconds" or None
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

info = st.text_area(
    label="I need some info to make the prediction",
    placeholder="please tell me your age, gender "
                "and how long it takes you to run 5km distance",
    value=st.session_state["info"]
)

if st.button("send"):
    resp = retrieve_structure(info, Runner)
    st.write(resp)
