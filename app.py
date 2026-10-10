import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Cotton Plant Disease Detection and Advisory System",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

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

.result-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    text-align: center;
    margin-bottom: 15px;
}

.result-title {
    font-size: 28px;
    font-weight: bold;
}

.confidence-value {
    font-size: 32px;
    font-weight: bold;
}

.section-box {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #dddddd;
    margin-top: 10px;
    margin-bottom: 15px;
}

.advisory-box {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #dddddd;
    margin-top: 10px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

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


# ============================================================
# MODEL SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 60.0


# ============================================================
# 7 COTTON LEAF CLASSES
# ============================================================

CLASS_NAMES = [
    "Bacterial Blight",
    "Curl Virus",
    "Healthy Leaf",
    "Herbicide Growth Damage",
    "Leaf Hopper Jassids",
    "Leaf Redding",
    "Leaf Variegation"
]


# ============================================================
# CONDITION TYPES
# ============================================================

CONDITION_TYPES = {

    "Bacterial Blight":
        "Disease",

    "Curl Virus":
        "Disease",

    "Healthy Leaf":
        "Healthy Condition",

    "Herbicide Growth Damage":
        "Non-disease Damage",

    "Leaf Hopper Jassids":
        "Pest-related Condition",

    "Leaf Redding":
        "Physiological / Stress Condition",

    "Leaf Variegation":
        "Leaf Condition"
}


# ============================================================
# CONDITION INFORMATION
# ============================================================

CONDITION_INFORMATION = {

    "Bacterial Blight": {

        "description":
            "A bacterial disease that can affect cotton leaves "
            "and cause visible lesions and discoloration.",

        "symptoms":
            "Leaf spots, lesions, discoloration and progressive "
            "damage to affected leaves."
    },

    "Curl Virus": {

        "description":
            "A viral condition associated with abnormal cotton "
            "leaf growth and curling.",

        "symptoms":
            "Leaf curling, distortion and abnormal plant growth."
    },

    "Healthy Leaf": {

        "description":
            "The image is classified as a healthy cotton leaf "
            "without strong visual indications of the other "
            "conditions.",

        "symptoms":
            "Normal leaf appearance without significant visible "
            "disease or damage symptoms."
    },

    "Herbicide Growth Damage": {

        "description":
            "A condition that may occur when cotton plants are "
            "affected by inappropriate herbicide exposure or drift.",

        "symptoms":
            "Abnormal growth, leaf distortion or other visible "
            "herbicide-related damage."
    },

    "Leaf Hopper Jassids": {

        "description":
            "A pest-related cotton condition associated with "
            "jassid activity.",

        "symptoms":
            "Leaf discoloration, curling or damage associated "
            "with pest activity."
    },

    "Leaf Redding": {

        "description":
            "A visible reddish or red coloration of cotton leaves "
            "that may be associated with plant stress or other factors.",

        "symptoms":
            "Reddish or abnormal leaf coloration."
    },

    "Leaf Variegation": {

        "description":
            "A leaf condition characterized by variation in "
            "leaf coloration or appearance.",

        "symptoms":
            "Uneven coloration or visible variation across the leaf."
    }
}


# ============================================================
# SMART ADVISORY
# ============================================================

SMART_ADVISORIES = {

    "Bacterial Blight": {

        "priority": "Disease Management",

        "immediate_action": [
            "Monitor affected cotton plants regularly.",
            "Inspect nearby plants for similar symptoms.",
            "Remove or manage severely affected plant material where recommended."
        ],

        "prevention": [
            "Maintain good field sanitation.",
            "Avoid unnecessary overhead irrigation.",
            "Reduce prolonged leaf wetness where possible.",
            "Follow recommended cotton disease-management practices."
        ],

        "monitoring": [
            "Check whether leaf lesions are increasing.",
            "Observe whether nearby plants develop similar symptoms.",
            "Record repeated occurrences for future crop management."
        ]
    },

    "Curl Virus": {

        "priority": "Disease and Vector Monitoring",

        "immediate_action": [
            "Inspect the plant and surrounding cotton plants.",
            "Monitor for continued leaf curling and abnormal growth.",
            "Check for possible insect-vector activity."
        ],

        "prevention": [
            "Maintain field sanitation.",
            "Monitor and manage insect vectors according to local guidance.",
            "Regularly inspect young cotton plants."
        ],

        "monitoring": [
            "Observe whether curling spreads to new leaves.",
            "Check nearby plants for similar symptoms.",
            "Seek expert guidance if symptoms become widespread."
        ]
    },

    "Healthy Leaf": {

        "priority": "Routine Crop Monitoring",

        "immediate_action": [
            "No major visible abnormality was detected by the model.",
            "Continue normal cotton crop-management practices."
        ],

        "prevention": [
            "Maintain proper irrigation.",
            "Maintain appropriate crop nutrition.",
            "Continue regular field inspection.",
            "Monitor for early signs of disease and pests."
        ],

        "monitoring": [
            "Inspect leaves regularly.",
            "Monitor for changes in colour or leaf structure.",
            "Continue routine crop health monitoring."
        ]
    },

    "Herbicide Growth Damage": {

        "priority": "Crop Damage Management",

        "immediate_action": [
            "Review recent herbicide application activities.",
            "Check whether unintended herbicide drift may have occurred.",
            "Observe the affected plants for continued growth abnormalities."
        ],

        "prevention": [
            "Follow recommended herbicide dosage.",
            "Avoid spraying during unsuitable weather conditions.",
            "Minimize herbicide drift onto cotton plants.",
            "Follow label and agricultural recommendations."
        ],

        "monitoring": [
            "Monitor newly developing leaves.",
            "Observe whether symptoms are improving or increasing.",
            "Consult an agricultural expert if severe damage continues."
        ]
    },

    "Leaf Hopper Jassids": {

        "priority": "Pest Management",

        "immediate_action": [
            "Inspect the crop for jassid activity.",
            "Check the underside and edges of affected leaves.",
            "Monitor nearby plants for similar damage."
        ],

        "prevention": [
            "Perform regular pest monitoring.",
            "Use integrated pest-management practices.",
            "Maintain healthy crop growth.",
            "Use recommended pest-control measures when required."
        ],

        "monitoring": [
            "Check plants regularly for increasing pest activity.",
            "Observe leaf discoloration and curling.",
            "Record areas with repeated pest occurrence."
        ]
    },

    "Leaf Redding": {

        "priority": "Plant Stress Monitoring",

        "immediate_action": [
            "Inspect the plant for additional symptoms.",
            "Check irrigation and general crop conditions.",
            "Observe whether reddish coloration is spreading."
        ],

        "prevention": [
            "Maintain appropriate irrigation.",
            "Check soil and plant nutrition.",
            "Maintain suitable crop-management practices."
        ],

        "monitoring": [
            "Monitor new leaves for colour changes.",
            "Check nearby plants for similar symptoms.",
            "Consult an agricultural expert if the condition persists."
        ]
    },

    "Leaf Variegation": {

        "priority": "Leaf Health Monitoring",

        "immediate_action": [
            "Inspect the affected leaf carefully.",
            "Check nearby leaves for similar colour variation.",
            "Monitor the plant for additional symptoms."
        ],

        "prevention": [
            "Maintain proper crop nutrition.",
            "Maintain appropriate irrigation.",
            "Monitor for pest or disease-related symptoms."
        ],

        "monitoring": [
            "Observe whether colour variation increases.",
            "Inspect surrounding plants.",
            "Consult an agricultural expert if symptoms become severe."
        ]
    }
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


try:

    model = load_model()

except Exception as e:

    st.error(
        "❌ Unable to load the MobileNetV2 model."
    )

    st.error(
        f"Model path: {MODEL_PATH}"
    )

    st.error(
        f"Error details: {e}"
    )

    st.stop()


# ============================================================
# FIND GRAD-CAM LAYER
# ============================================================

def find_gradcam_layer(model):

    for layer in reversed(model.layers):

        try:

            output_shape = layer.output.shape

            if len(output_shape) == 4:

                return layer

        except Exception:

            continue

    return None


gradcam_layer = find_gradcam_layer(model)


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    image_array,
    predicted_index,
    target_layer
):

    if target_layer is None:

        return None

    try:

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                target_layer.output,
                model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(
                image_array,
                training=False
            )

            class_channel = predictions[
                :,
                predicted_index
            ]

        grads = tape.gradient(
            class_channel,
            conv_outputs
        )

        if grads is None:

            return None

        pooled_grads = tf.reduce_mean(
            grads,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(
            conv_outputs * pooled_grads,
            axis=-1
        )

        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_value = tf.reduce_max(
            heatmap
        )

        heatmap = heatmap / (
            max_value +
            tf.keras.backend.epsilon()
        )

        return heatmap.numpy()

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )

        return None


# ============================================================
# CREATE GRAD-CAM OVERLAY
# ============================================================

def create_gradcam_overlay(
    original_image,
    heatmap
):

    original_array = np.array(
        original_image
    )

    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis],
        (
            original_array.shape[0],
            original_array.shape[1]
        )
    )

    heatmap_resized = (
        heatmap_resized
        .numpy()
        .squeeze()
    )

    colormap = plt.get_cmap(
        "jet"
    )

    colored_heatmap = colormap(
        heatmap_resized
    )[:, :, :3]

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(
        np.uint8
    )

    original_float = (
        original_array.astype(
            np.float32
        ) / 255.0
    )

    heatmap_float = (
        colored_heatmap.astype(
            np.float32
        ) / 255.0
    )

    overlay = (
        0.55 * original_float
        +
        0.45 * heatmap_float
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    overlay = (
        overlay * 255
    ).astype(
        np.uint8
    )

    return overlay


# ============================================================
# SAVE PREDICTION HISTORY
# ============================================================

def save_prediction_history(
    image_name,
    predicted_condition,
    confidence,
    prediction_status
):

    new_record = pd.DataFrame([
        {

            "Date & Time":
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M:%S"
                ),

            "Image Name":
                image_name,

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

        try:

            old_history = pd.read_csv(
                HISTORY_FILE
            )

        except Exception:

            old_history = pd.DataFrame()

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


# ============================================================
# SIDEBAR
# ============================================================

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
        "🧠 Explainable AI"
    )

    st.write(
        "Grad-CAM"
    )

    st.markdown("---")

    st.subheader(
        "💡 Feature 4"
    )

    st.write(
        "Smart Advisory & Recommended Action"
    )

    st.markdown("---")

    st.subheader(
        "📊 Feature 5"
    )

    st.write(
        "Prediction History & Analytics"
    )

    st.markdown("---")

    st.subheader(
        "🛡️ Reliability"
    )

    st.write(
        f"Confidence Threshold: "
        f"{CONFIDENCE_THRESHOLD:.0f}%"
    )

    st.markdown("---")

    st.caption(
        "Academic Project Demonstration"
    )


# ============================================================
# MAIN TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🌱Smart Cotton Plant Disease Detection'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-title">'
    'and Advisory System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Capture or upload a clear cotton leaf image for AI-based analysis'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SMART IMAGE CAPTURE & UPLOAD
# ============================================================

st.markdown("---")

st.subheader(
    "📷 Cotton Leaf Image"
)

st.write(
    "Click the button below to choose how you want to "
    "provide the cotton leaf image."
)


# ============================================================
# IMAGE SOURCE POPUP
# ============================================================

camera_image = None
gallery_image = None

with st.popover(
    "📷 Select Image Source",
    use_container_width=True
):

    st.markdown(
        "### 📷 Choose Image Source"
    )

    st.write(
        "Select one of the following options:"
    )

    st.markdown("---")


    # ========================================================
    # CAMERA
    # ========================================================

    st.markdown(
        "#### 📸 Camera Capture"
    )

    camera_image = st.camera_input(
        "Take a photo of the cotton leaf",
        key="camera_capture"
    )


    st.markdown("---")


    # ========================================================
    # GALLERY
    # ========================================================

    st.markdown(
        "#### 📁 Upload From Gallery"
    )

    gallery_image = st.file_uploader(
        "Choose a cotton leaf image from your device",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="gallery_upload",
        help="Upload a clear and well-lit cotton leaf image."
    )


# ============================================================
# SELECT IMAGE SOURCE
# ============================================================

uploaded_file = None
image_source = None


if camera_image is not None:

    uploaded_file = camera_image

    image_source = "📸 Camera Capture"

elif gallery_image is not None:

    uploaded_file = gallery_image

    image_source = "📁 Gallery Upload"


# ============================================================
# IMAGE VALIDATION
# ============================================================

image_valid = False
image = None


if uploaded_file is not None:

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        width, height = image.size


        # ====================================================
        # SELECTED IMAGE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🖼️ Selected Cotton Leaf Image"
        )

        col1, col2 = st.columns(
            [1.5, 1]
        )


        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        with col1:

            st.image(
                image,
                caption=image_source,
                use_container_width=True
            )


        # ====================================================
        # IMAGE INFORMATION
        # ====================================================

        with col2:

            st.subheader(
                "📋 Image Information"
            )

            st.write(
                f"**Source:** {image_source}"
            )

            st.write(
                f"**File Name:** "
                f"{uploaded_file.name}"
            )

            st.write(
                f"**Image Size:** "
                f"{width} × {height} pixels"
            )

            st.write(
                f"**Image Format:** "
                f"{image.format if image.format else 'Image'}"
            )


        # ====================================================
        # IMAGE RESOLUTION VALIDATION
        # ====================================================

        if width < 224 or height < 224:

            st.error(
                "❌ Image resolution is too low."
            )

            st.info(
                "Please capture or select an image "
                "with at least 224 × 224 pixels."
            )

        else:

            image_valid = True

            st.success(
                "✅ Image is ready for AI analysis."
            )


    except Exception:

        st.error(
            "❌ Unable to process this image."
        )

        st.info(
            "Please capture or upload a valid "
            "JPG, JPEG or PNG image."
        )


# ============================================================
# PREDICTION
# ============================================================

if image_valid:

    st.markdown("---")

    if st.button(
        "🔍 Analyze Cotton Leaf",
        use_container_width=True
    ):

        # ====================================================
        # PREPROCESS IMAGE
        # ====================================================

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


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        predictions = model.predict(
            image_array,
            verbose=0
        )

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


        # ====================================================
        # CONFIDENCE STATUS
        # ====================================================

        if confidence >= 80:

            prediction_status = (
                "High Confidence"
            )

        elif confidence >= CONFIDENCE_THRESHOLD:

            prediction_status = (
                "Moderate Confidence"
            )

        else:

            prediction_status = (
                "Low Confidence"
            )


        # ====================================================
        # SAVE HISTORY
        # ====================================================

        save_prediction_history(
            image_name=uploaded_file.name,
            predicted_condition=predicted_condition,
            confidence=confidence,
            prediction_status=prediction_status
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown("---")

        st.header(
            "🌿 AI Prediction Result"
        )

        col1, col2, col3 = st.columns(3)


        with col1:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.write(
                "🌿 **Predicted Condition**"
            )

            st.markdown(
                f'<div class="result-title">'
                f'{predicted_condition}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with col2:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.write(
                "🎯 **Confidence Score**"
            )

            st.markdown(
                f'<div class="confidence-value">'
                f'{confidence:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

            st.progress(
                float(
                    predictions[0][
                        predicted_index
                    ]
                )
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        with col3:

            st.markdown(
                '<div class="result-card">',
                unsafe_allow_html=True
            )

            st.write(
                "📌 **Condition Type**"
            )

            st.markdown(
                f'<div class="result-title">'
                f'{CONDITION_TYPES[predicted_condition]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '</div>',
                unsafe_allow_html=True
            )


        # ====================================================
        # RELIABILITY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🛡️ Prediction Reliability"
        )

        if confidence >= 80:

            st.success(
                "🟢 High Confidence — "
                "The model shows strong confidence "
                "in this prediction."
            )

        elif confidence >= CONFIDENCE_THRESHOLD:

            st.warning(
                "🟡 Moderate Confidence — "
                "The result should be interpreted carefully."
            )

        else:

            st.error(
                "🔴 Low Confidence — "
                "The model is uncertain."
            )

            st.info(
                "Please upload a clearer image "
                "or consult an agricultural expert."
            )


        # ====================================================
        # TOP 3
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🔎 Top 3 Possible Conditions"
        )

        top_3_indices = np.argsort(
            predictions[0]
        )[-3:][::-1]

        for rank, index in enumerate(
            top_3_indices,
            start=1
        ):

            top_class = CLASS_NAMES[
                index
            ]

            top_confidence = (
                predictions[0][index] * 100
            )

            col1, col2 = st.columns(
                [3, 1]
            )

            with col1:

                st.write(
                    f"**{rank}. {top_class}**"
                )

            with col2:

                st.write(
                    f"**{top_confidence:.2f}%**"
                )

            st.progress(
                float(
                    predictions[0][index]
                )
            )


        # ====================================================
        # CONDITION INFORMATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📖 About the Predicted Condition"
        )

        condition_info = CONDITION_INFORMATION[
            predicted_condition
        ]

        st.markdown(
            '<div class="section-box">',
            unsafe_allow_html=True
        )

        st.write(
            "**Description**"
        )

        st.write(
            condition_info["description"]
        )

        st.write(
            "**Common Visual Symptoms**"
        )

        st.write(
            condition_info["symptoms"]
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # FEATURE 3
        # GRAD-CAM
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🧠 AI Visual Explanation — Grad-CAM"
        )

        st.write(
            "Grad-CAM highlights image regions that "
            "contributed to the model prediction."
        )

        heatmap = generate_gradcam(
            model=model,
            image_array=image_array,
            predicted_index=predicted_index,
            target_layer=gradcam_layer
        )

        if heatmap is not None:

            gradcam_overlay = (
                create_gradcam_overlay(
                    image,
                    heatmap
                )
            )

            grad_col1, grad_col2 = st.columns(2)

            with grad_col1:

                st.image(
                    image,
                    caption="Original Cotton Leaf",
                    use_container_width=True
                )

            with grad_col2:

                st.image(
                    gradcam_overlay,
                    caption="Grad-CAM Prediction Heatmap",
                    use_container_width=True
                )

            st.success(
                "✅ Grad-CAM explanation generated."
            )

            st.info(
                f"Highlighted regions contributed "
                f"to the prediction of "
                f"**{predicted_condition}**."
            )

        else:

            st.warning(
                "⚠️ Grad-CAM could not be generated."
            )


        # ====================================================
        # SMART ADVISORY
        # ====================================================

        st.markdown("---")

        st.header(
            "💡 Feature 4 — Smart Advisory & Recommended Action"
        )

        st.write(
            "The system provides condition-specific "
            "recommendations based on the AI prediction."
        )

        if confidence < CONFIDENCE_THRESHOLD:

            st.error(
                "⚠️ Low-confidence prediction"
            )

            st.info(
                "The advisory should not be treated as "
                "a confirmed diagnosis."
            )

        advisory = SMART_ADVISORIES[
            predicted_condition
        ]


        # ====================================================
        # ADVISORY CATEGORY
        # ====================================================

        st.markdown(
            '<div class="advisory-box">',
            unsafe_allow_html=True
        )

        st.subheader(
            "🚨 Advisory Category"
        )

        st.write(
            f"**{advisory['priority']}**"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # IMMEDIATE ACTION
        # ====================================================

        st.subheader(
            "⚡ Immediate Action"
        )

        for action in advisory[
            "immediate_action"
        ]:

            st.write(
                "🔹 " + action
            )


        # ====================================================
        # PREVENTION
        # ====================================================

        st.subheader(
            "🛡️ Prevention & Management"
        )

        for action in advisory[
            "prevention"
        ]:

            st.write(
                "🔹 " + action
            )


        # ====================================================
        # MONITORING
        # ====================================================

        st.subheader(
            "👁️ What to Monitor"
        )

        for action in advisory[
            "monitoring"
        ]:

            st.write(
                "🔹 " + action
            )


        st.info(
            "⚠️ Advisory information is intended for "
            "academic/project demonstration and should "
            "be verified with local agricultural guidance."
        )


        # ====================================================
        # PROBABILITY ANALYSIS
        # ====================================================

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

                "Probability (%)":
                    round(
                        probability,
                        2
                    )
            })

        probability_df = pd.DataFrame(
            probability_data
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PROBABILITY BARS
        # ====================================================

        st.subheader(
            "📈 Probability Visualization"
        )

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = (
                predictions[0][i]
            )

            st.write(
                f"**{class_name} — "
                f"{probability * 100:.2f}%**"
            )

            st.progress(
                float(probability)
            )


        # ====================================================
        # FINAL SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📋 Prediction Summary"
        )

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(
                f"**Condition:** "
                f"{predicted_condition}"
            )

            st.write(
                f"**Condition Type:** "
                f"{CONDITION_TYPES[predicted_condition]}"
            )

        with summary_col2:

            st.write(
                f"**Confidence:** "
                f"{confidence:.2f}%"
            )

            st.write(
                f"**Reliability:** "
                f"{prediction_status}"
            )


        st.markdown("---")

        st.info(
            "⚠️ This AI prediction and advisory are "
            "intended for academic/project demonstration "
            "and should not replace professional "
            "agricultural diagnosis."
        )


# ============================================================

# PREDICTION HISTORY & ANALYTICS
# ============================================================

if os.path.exists(
    HISTORY_FILE
):

    try:

        history_df = pd.read_csv(
            HISTORY_FILE
        )

    except Exception:

        history_df = pd.DataFrame()


    if not history_df.empty:

        st.markdown("---")

        st.header(
            "📊 Feature 5 — Prediction History & Analytics"
        )

        st.write(
            "This dashboard summarizes previous "
            "predictions made by the AI system."
        )


        # ====================================================
        # CLEAN CONFIDENCE VALUES
        # ====================================================

        history_df["Confidence Value"] = (

            history_df[
                "Confidence"
            ]
            .astype(str)
            .str.replace(
                "%",
                "",
                regex=False
            )
            .astype(float)
        )


        # ====================================================
        # BASIC STATISTICS
        # ====================================================

        total_predictions = len(
            history_df
        )

        average_confidence = (
            history_df[
                "Confidence Value"
            ].mean()
        )

        highest_confidence = (
            history_df[
                "Confidence Value"
            ].max()
        )

        lowest_confidence = (
            history_df[
                "Confidence Value"
            ].min()
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


        # ====================================================
        # OVERALL STATISTICS
        # ====================================================

        st.subheader(
            "📌 Overall Statistics"
        )

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


        # ====================================================
        # CONFIDENCE STATISTICS
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🎯 Confidence Statistics"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Highest Confidence",
                f"{highest_confidence:.2f}%"
            )

        with col2:

            st.metric(
                "Lowest Confidence",
                f"{lowest_confidence:.2f}%"
            )

        with col3:

            st.metric(
                "Low Confidence",
                uncertain_predictions
            )


        # ====================================================
        # RELIABILITY SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🛡️ Reliability Summary"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(
                f"🟢 Reliable Predictions: "
                f"**{reliable_predictions}**"
            )

        with col2:

            if uncertain_predictions > 0:

                st.warning(
                    f"🟡 Low Confidence Predictions: "
                    f"**{uncertain_predictions}**"
                )

            else:

                st.success(
                    "🟢 No low-confidence predictions."
                )


        # ====================================================
        # CONDITION DISTRIBUTION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📈 Condition Prediction Distribution"
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

        chart_df = pd.DataFrame({

            "Condition":
                condition_counts.index,

            "Number of Predictions":
                condition_counts.values

        })

        st.bar_chart(
            chart_df.set_index(
                "Condition"
            )
        )


        # ====================================================
        # CONFIDENCE TREND
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📉 Confidence Trend"
        )

        confidence_chart = (

            history_df[
                ["Confidence Value"]
            ]
            .reset_index(
                drop=True
            )
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


        # ====================================================
        # CLASS-WISE SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "📋 Class-wise Prediction Summary"
        )

        class_summary = []

        for class_name in CLASS_NAMES:

            class_records = history_df[
                history_df[
                    "Predicted Condition"
                ] == class_name
            ]

            count = len(
                class_records
            )

            if count > 0:

                average_class_confidence = (
                    class_records[
                        "Confidence Value"
                    ].mean()
                )

            else:

                average_class_confidence = 0

            class_summary.append({

                "Condition":
                    class_name,

                "Predictions":
                    count,

                "Average Confidence (%)":
                    round(
                        average_class_confidence,
                        2
                    )
            })

        class_summary_df = pd.DataFrame(
            class_summary
        )

        st.dataframe(
            class_summary_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # COMPLETE HISTORY
        # ====================================================

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


        # ====================================================
        # DOWNLOAD HISTORY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "⬇️ Export Prediction History"
        )

        csv_data = display_history.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction History CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv",
            use_container_width=True
        )


        # ====================================================
        # CLEAR HISTORY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "🗑️ History Management"
        )

        st.warning(
            "Clearing history will permanently remove "
            "the saved prediction history."
        )

        if st.button(
            "🗑️ Clear Prediction History",
            use_container_width=True
        ):

            try:

                os.remove(
                    HISTORY_FILE
                )

                st.success(
                    "✅ Prediction history cleared."
                )

                st.rerun()

            except Exception as e:

                st.error(
                    f"Unable to clear history: {e}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Smart Cotton Plant Disease Detection "
    "& Advisory System | MobileNetV2 | "
    "Grad-CAM | Feature 4: Smart Advisory | "
    "Feature 5: Prediction History & Analytics"
)