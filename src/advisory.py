# ==========================================
# Cotton Disease Advisory System
# ==========================================

ADVISORIES = {

    "Bacterial Blight": {

        
        "cause": "Bacterial infection affecting cotton leaves.",
        "symptoms": "Water-soaked or dark lesions that may enlarge on leaves.",
        "advice": [
            "Remove and properly dispose of severely affected plant material.",
            "Maintain good field sanitation.",
            "Avoid unnecessary overhead irrigation and prolonged leaf wetness.",
            "Use recommended disease-management practices for cotton."
        ]
    },

    "Curl Virus": {
        "cause": "Viral disease affecting cotton plants.",
        "symptoms": "Leaf curling, distortion and abnormal growth may occur.",
        "advice": [
            "Monitor plants regularly for disease symptoms.",
            "Remove severely affected plants where recommended.",
            "Control insect vectors such as whiteflies according to agricultural guidance.",
            "Maintain proper field sanitation."
        ]
    },

    "Healthy Leaf": {
        "cause": "No visible disease condition detected by the model.",
        "symptoms": "Leaf appears healthy.",
        "advice": [
            "Continue regular monitoring of the crop.",
            "Maintain proper irrigation and nutrition.",
            "Continue normal crop-management practices.",
            "Inspect plants regularly for early signs of disease or pests."
        ]
    },

    "Herbicide Growth Damage": {
        "cause": "Possible injury caused by herbicide exposure.",
        "symptoms": "Abnormal growth, leaf deformation or discoloration may occur.",
        "advice": [
            "Review recent herbicide application and dosage.",
            "Avoid unintended herbicide drift onto cotton plants.",
            "Follow recommended herbicide application practices.",
            "Consult an agricultural expert if severe damage is observed."
        ]
    },

    "Leaf Hopper Jassids": {
        "cause": "Damage associated with jassid insect infestation.",
        "symptoms": "Leaf yellowing, curling, reddening or drying may occur.",
        "advice": [
            "Monitor the crop regularly for jassid activity.",
            "Use appropriate integrated pest-management practices.",
            "Remove heavily damaged plant material where appropriate.",
            "Use recommended pest-control measures according to local agricultural guidance."
        ]
    },

    "Leaf Redding": {
        "cause": "Leaf reddening condition in cotton.",
        "symptoms": "Leaves may develop reddish or reddish-purple coloration.",
        "advice": [
            "Check plant nutrition and soil conditions.",
            "Maintain appropriate irrigation.",
            "Monitor plants for additional disease or pest symptoms.",
            "Consult an agricultural expert if the condition persists or spreads."
        ]
    },

    "Leaf Variegation": {
        "cause": "Variegated leaf appearance detected by the model.",
        "symptoms": "Irregular patterns of different shades may appear on leaves.",
        "advice": [
            "Monitor affected plants regularly.",
            "Check for associated pest, disease or nutritional problems.",
            "Maintain proper crop nutrition and irrigation.",
            "Consult an agricultural expert if symptoms increase."
        ]
    }
}


def get_advisory(condition):

    return ADVISORIES.get(
        condition,
        {
            "cause": "Information not available.",
            "symptoms": "Please consult an agricultural expert.",
            "advice": [
                "Monitor the plant regularly.",
                "Consult a qualified agricultural expert."
            ]
        }
    )


# Test the advisory system
if __name__ == "__main__":

    condition = "Curl Virus"

    advisory = get_advisory(condition)

    print("\n================================")
    print("COTTON DISEASE ADVISORY")
    print("================================")

    print("\nCondition:", condition)

    print("\nCause:")
    print(advisory["cause"])

    print("\nSymptoms:")
    print(advisory["symptoms"])

    print("\nRecommended Advisory:")

    for i, advice in enumerate(advisory["advice"], 1):
        print(f"{i}. {advice}")

    print("================================")