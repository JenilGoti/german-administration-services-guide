import re, json

def clean_json_output(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    return text.strip()


def safe_json_parse(text):
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None

    cleaned = clean_json_output(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
            return parsed
        except Exception:
            continue

    return None
