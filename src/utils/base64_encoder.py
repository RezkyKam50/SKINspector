import base64 # pybase64 is more compute efficient than legacy base64

def image_to_base64_data_uri(file_path: str, mime_type: str = "image/jpeg") -> str:
    with open(file_path, "rb") as img_file:
        base64_data = base64.encodebytes(img_file.read()).decode("ascii")
    return f"data:{mime_type};base64,{base64_data}"