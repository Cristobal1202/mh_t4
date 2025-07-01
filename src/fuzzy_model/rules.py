# src/fuzzy_model/rules.py

rules = [
    {
        "if": {
            "SSLfinal_State": "low",
            "web_traffic": "low"
        },
        "then": "phishing"
    },
    {
        "if": {
            "Request_URL": "high",
            "URL_of_Anchor": "high"
        },
        "then": "legitimate"
    },
    {
        "if": {
            "age_of_domain": "low",
            "SFH": "low"
        },
        "then": "phishing"
    },
    {
        "if": {
            "having_IP_Address": "high",
            "popUpWidnow": "high"
        },
        "then": "phishing"
    },
    {
        "if": {
            "URL_Length": "high",
            "web_traffic": "medium"
        },
        "then": "phishing"
    }
]
