import streamlit as st


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
    st.write(info)
