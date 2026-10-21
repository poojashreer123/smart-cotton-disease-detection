import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smart Cotton Plant Disease Detection And Advisory System",
    page_icon="🌱",
    layout="wide"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

.main-title {
    font-size: 38px;
    font-weight: bold;
    text-align: center;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
}

.dashboard-card {
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #dddddd;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# PATHS
# ==========================================

# MOBILE NET V2 FINE-TUNED MODEL
MODEL_PATH = "model/mobilenetv2_finetuned_best.keras"

RESULTS_DIR = "results"

HISTORY_FILE = os.path.join(
    RESULTS_DIR,
    "prediction_history.csv"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ==========================================
# MODEL SETTINGS
# ==========================================

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 60.0

CLASS_NAMES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation"
]


# ==========================================
# ADVISORY INFORMATION
# ==========================================

ADVISORIES = {

    "Bacterial Blight": [
        "Monitor affected plants regularly.",
        "Maintain good field sanitation.",
        "Avoid unnecessary overhead irrigation and prolonged leaf wetness.",
        "Follow recommended disease-management practices for cotton."
    ],

    "Curl Virus": [
        "Monitor plants regularly for disease symptoms.",
        "Remove severely affected plants where recommended.",
        "Monitor and manage insect vectors according to agricultural guidance.",
        "Maintain proper field sanitation."
    ],

    "Healthy Leaf": [
        "Continue regular monitoring of the crop.",
        "Maintain proper irrigation and nutrition.",
        "Continue normal crop-management practices.",
        "Inspect plants regularly for early signs of disease or pests."
    ],

    "Herbicide Growth Damage": [
        "Review recent herbicide application and dosage.",
        "Avoid unintended herbicide drift onto cotton plants.",
        "Follow recommended herbicide application practices.",
        "Consult an agricultural expert if severe damage is observed."
    ],

    "Leaf Hopper Jassids": [
        "Monitor the crop regularly for jassid activity.",
        "Use appropriate integrated pest-management practices.",
        "Monitor heavily affected plants carefully.",
        "Use recommended pest-control measures according to local agricultural guidance."
    ],

    "Leaf Redding": [
        "Check plant nutrition and soil conditions.",
        "Maintain appropriate irrigation.",
        "Monitor plants for additional disease or pest symptoms.",
        "Consult an agricultural expert if the condition persists or spreads."
    ],

    "Leaf Variegation": [
        "Monitor affected plants regularly.",
        "Check for associated pest, disease or nutritional problems.",
        "Maintain proper crop nutrition and irrigation.",
        "Consult an agricultural expert if symptoms increase."
    ]
}


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_model()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title(
        "🌱 Project Information"
    )

    st.markdown("---")

    st.subheader(
        "📌 Project"
    )

    st.write(
        "AI-Based Smart Cotton Plant "
        "Disease Detection and Advisory System"
    )

    st.markdown("---")

    st.subheader(
        "🧠 AI Model"
    )

    st.write("MobileNetV2")
    st.write("Transfer Learning")
    st.write("Fine-Tuning")

    st.markdown("---")

    st.subheader(
        "📊 Dataset"
    )

    st.write("SAR-CLD-2024")
    st.write("7 Cotton Leaf Conditions")

    st.markdown("---")

    st.subheader(
        "🎯 Model Performance"
    )

    st.metric(
        "Test Accuracy",
        "66.67%"
    )

    st.metric(
        "Average Confidence",
        "60.26%"
    )

    st.markdown("---")

    st.subheader(
        "⚙️ Technologies"
    )

    st.write("🐍 Python")
    st.write("🧠 TensorFlow / Keras")
    st.write("🌐 Streamlit")
    st.write("📊 NumPy")
    st.write("🖼️ PIL")
    st.write("📋 Pandas")

    st.markdown("---")

    st.subheader(
        "🛡️ Reliability Check"
    )

    st.write(
        f"Confidence Threshold: "
        f"{CONFIDENCE_THRESHOLD:.0f}%"
    )

    st.markdown("---")

    st.caption(
        "Academic Project Demonstration"
    )


# ==========================================
# MAIN TITLE
# ==========================================

st.markdown(
    '<div class="main-title">'
    '🌱 AI-Based Smart Cotton Plant'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">'
    'Disease Detection & Advisory System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a cotton leaf image to detect its condition'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📷 Choose a cotton leaf image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Uploaded Cotton Leaf",
        width=400
    )

    # ==========================================
    # PREDICT BUTTON
    # ==========================================

    if st.button(
        "🔍 Predict Disease",
        use_container_width=True
    ):

        # ==========================================
        # IMAGE PREPROCESSING
        # ==========================================

        resized_image = image.resize(
            IMG_SIZE
        )

        image_array = np.array(
            resized_image
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        # ==========================================
        # MODEL PREDICTION
        # ==========================================

        predictions = model.predict(
            image_array,
            verbose=0
        )

        # ==========================================
        # GET PREDICTION
        # ==========================================

        predicted_index = np.argmax(
            predictions[0]
        )

        predicted_condition = (
            CLASS_NAMES[
                predicted_index
            ]
        )

        confidence = (
            predictions[0][
                predicted_index
            ] * 100
        )

        # ==========================================
        # CONFIDENCE STATUS
        # ==========================================

        if confidence >= CONFIDENCE_THRESHOLD:

            prediction_status = (
                "Reliable Prediction"
            )

        else:

            prediction_status = (
                "Low Confidence / Uncertain"
            )

        # ==========================================
        # SAVE PREDICTION TO HISTORY
        # ==========================================

        new_record = pd.DataFrame([
            {
                "Date & Time":
                    datetime.now().strftime(
                        "%d-%m-%Y %H:%M:%S"
                    ),

                "Image Name":
                    uploaded_file.name,

                "Predicted Condition":
                    predicted_condition,

                "Confidence":
                    f"{confidence:.2f}%",

                "Status":
                    prediction_status
            }
        ])

        if os.path.exists(
            HISTORY_FILE
        ):

            old_history = pd.read_csv(
                HISTORY_FILE
            )

            updated_history = pd.concat(
                [
                    old_history,
                    new_record
                ],
                ignore_index=True
            )

        else:

            updated_history = new_record

        updated_history.to_csv(
            HISTORY_FILE,
            index=False
        )

        # ==========================================
        # DETECTION RESULT
        # ==========================================

        st.markdown("---")

        st.subheader(
            "🌿 Detection Result"
        )

        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                f"Predicted Condition: "
                f"{predicted_condition}"
            )

        else:

            st.warning(
                f"Predicted Condition: "
                f"{predicted_condition}"
            )

        # ==========================================
        # CONFIDENCE
        # ==========================================

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        # ==========================================
        # RELIABILITY CHECK
        # ==========================================

        if confidence >= CONFIDENCE_THRESHOLD:

            st.success(
                "✅ Reliable Prediction"
            )

            st.write(
                "The model confidence is above "
                f"the {CONFIDENCE_THRESHOLD:.0f}% "
                "threshold."
            )

        else:

            st.warning(
                "⚠️ Low Confidence / Uncertain"
            )

            st.write(
                "The model confidence is below "
                f"the {CONFIDENCE_THRESHOLD:.0f}% "
                "threshold."
            )

            st.info(
                "Please upload a clear cotton "
                "leaf image for a more reliable "
                "prediction."
            )

        # ==========================================
        # CLASS PROBABILITY ANALYSIS
        # ==========================================

        st.markdown("---")

        st.subheader(
            "📊 Class Probability Analysis"
        )

        probability_data = []

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                predictions[0][i] * 100
            )

            probability_data.append({
                "Condition":
                    class_name,

                "Probability":
                    probability
            })

            st.write(
                f"**{class_name} — "
                f"{probability:.2f}%**"
            )

            st.progress(
                float(
                    predictions[0][i]
                )
            )

        # ==========================================
        # ADVISORY
        # ==========================================

        st.markdown("---")

        st.subheader(
            "💡 Recommended Advisory"
        )

        if confidence >= CONFIDENCE_THRESHOLD:

            for advice in ADVISORIES[
                predicted_condition
            ]:

                st.write(
                    "• " + advice
                )

        else:

            st.warning(
                "Because the prediction confidence "
                "is low, the advisory should not be "
                "treated as reliable. Please upload "
                "a clearer image or consult an "
                "agricultural expert."
            )

        # ==========================================
        # DISCLAIMER
        # ==========================================

        st.info(
            "This advisory is provided for "
            "project demonstration and should "
            "not replace professional "
            "agricultural advice."
        )


# ==========================================
# ANALYTICS DASHBOARD
# ==========================================

if os.path.exists(
    HISTORY_FILE
):

    history_df = pd.read_csv(
        HISTORY_FILE
    )

    if not history_df.empty:

        st.markdown("---")

        st.header(
            "📊 Prediction Analytics Dashboard"
        )

        st.write(
            "Analytics generated from the "
            "prediction history of the system."
        )

        # ==========================================
        # CLEAN CONFIDENCE VALUES
        # ==========================================

        history_df["Confidence Value"] = (
            history_df["Confidence"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .astype(float)
        )

        # ==========================================
        # DASHBOARD METRICS
        # ==========================================

        total_predictions = len(
            history_df
        )

        average_confidence = (
            history_df["Confidence Value"]
            .mean()
        )

        most_common_condition = (
            history_df[
                "Predicted Condition"
            ]
            .value_counts()
            .idxmax()
        )

        reliable_predictions = (
            history_df[
                "Confidence Value"
            ] >= CONFIDENCE_THRESHOLD
        ).sum()

        uncertain_predictions = (
            history_df[
                "Confidence Value"
            ] < CONFIDENCE_THRESHOLD
        ).sum()

        # ==========================================
        # METRIC CARDS
        # ==========================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Total Predictions",
                total_predictions
            )

        with col2:

            st.metric(
                "Average Confidence",
                f"{average_confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Most Detected",
                most_common_condition
            )

        with col4:

            st.metric(
                "Reliable Predictions",
                reliable_predictions
            )

        # ==========================================
        # STATUS SUMMARY
        # ==========================================

        st.markdown("---")

        st.subheader(
            "🛡️ Prediction Reliability"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Reliable",
                reliable_predictions
            )

        with col2:

            st.metric(
                "Low Confidence",
                uncertain_predictions
            )

        # ==========================================
        # PREDICTION DISTRIBUTION
        # ==========================================

        st.markdown("---")

        st.subheader(
            "📈 Prediction Distribution"
        )

        condition_counts = (
            history_df[
                "Predicted Condition"
            ]
            .value_counts()
            .reindex(
                CLASS_NAMES,
                fill_value=0
            )
        )

        st.bar_chart(
            condition_counts
        )

        # ==========================================
        # CONFIDENCE ANALYSIS
        # ==========================================

        st.markdown("---")

        st.subheader(
            "🎯 Confidence Analysis"
        )

        confidence_chart = (
            history_df[
                [
                    "Confidence Value"
                ]
            ]
            .reset_index(drop=True)
        )

        confidence_chart.index = (
            confidence_chart.index + 1
        )

        confidence_chart.columns = [
            "Confidence (%)"
        ]

        st.line_chart(
            confidence_chart
        )

        # ==========================================
        # CLASS SUMMARY TABLE
        # ==========================================

        st.markdown("---")

        st.subheader(
            "📋 Class-wise Prediction Summary"
        )

        class_summary = pd.DataFrame({

            "Condition":
                CLASS_NAMES,

            "Predictions":
                [
                    (
                        history_df[
                            "Predicted Condition"
                        ] == class_name
                    ).sum()

                    for class_name
                    in CLASS_NAMES
                ]
        })

        st.dataframe(
            class_summary,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # FULL HISTORY
        # ==========================================

        st.markdown("---")

        st.subheader(
            "📋 Complete Prediction History"
        )

        display_history = history_df.drop(
            columns=[
                "Confidence Value"
            ]
        )

        st.dataframe(
            display_history,
            use_container_width=True,
            hide_index=True
        )

        # ==========================================
        # DOWNLOAD HISTORY
        # ==========================================

        csv_data = display_history.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction History",
            data=csv_data,
            file_name=(
                "prediction_history.csv"
            ),
            mime="text/csv",
            use_container_width=True
        )

        # ==========================================
        # CLEAR HISTORY
        # ==========================================

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            os.remove(
                HISTORY_FILE
            )

            st.rerun()


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "AI-Based Smart Cotton Plant Disease "
    "Detection and Advisory System | MobileNetV2"
)