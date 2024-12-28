import uvicorn
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import shutil
import os
from running import enhanced_pic

app = FastAPI()

@app.post("/get_image/")
async def get_image(file: UploadFile = File(...), prompt: str = Form(...)):
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    enhanced_images = enhanced_pic(temp_file_path, prompt)

    # Remove the temporary file after enhancement, in case of memory management issue

    # path to image is set under UI key, rest items contains mask paths
    enhanced_image_path = enhanced_images.get('UI')

    # For display at Swagger UI, returning the enhanced image as a response
    return FileResponse(enhanced_image_path, media_type="image/png")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
