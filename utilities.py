import re, json

def clean_json_output(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def safe_json_parse(text):
    try:
        return json.loads(clean_json_output(text))
    except Exception:
        return None

