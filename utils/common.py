
import streamlit as st

st.set_page_config(
    page_title="AI Cotton Disease Detection",
    page_icon="🌿",
    layout="wide"
)

st.title("🌿 AI-Based Cotton Disease Detection")
st.subheader("Smart detection and advisory system for cotton farmers")

st.write(
    "Welcome! Upload a cotton leaf image to identify possible diseases, "
    "learn about symptoms, and explore model performance."
)

st.divider()
st.subheader("Explore the Application")

col1, col2 = st.columns(2)

with col1:
    if st.button("📚 Disease Information", use_container_width=True):
        st.session_state["previous_page"] = "app.py"
        st.session_state["current_page"] = "pages/1_Disease_Information.py"
        st.switch_page("pages/1_Disease_Information.py")

    if st.button("🔍 Disease Prediction", use_container_width=True):
        st.session_state["previous_page"] = "app.py"
        st.session_state["current_page"] = "pages/3_Disease_Prediction.py"
        st.switch_page("pages/3_Disease_Prediction.py")

with col2:
    if st.button("💡 Smart Advisory", use_container_width=True):
        st.session_state["previous_page"] = "app.py"
        st.session_state["current_page"] = "pages/2_Smart_Advisory.py"
        st.switch_page("pages/2_Smart_Advisory.py")

    if st.button("📊 Disease Analytics", use_container_width=True):
        st.session_state["previous_page"] = "app.py"
        st.session_state["current_page"] = "pages/4_Disease_Analytics.py"
        st.switch_page("pages/4_Disease_Analytics.py")

st.divider()

st.info(
    "To predict a disease, open Disease Prediction and upload a "
    "clear cotton leaf image."
)

st.caption("AI Cotton Disease Detection Project")