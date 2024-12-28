import os
import sys
import cv2
import torch
import supervision as sv
from PIL import Image
import numpy as np
import json
import matplotlib.pyplot as plt
from torchvision.ops import box_convert
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, load_image, predict, annotate
from fastapi.responses import JSONResponse
import matplotlib.colors as mcolors

def show_mask(mask, ax, col_mask='blue'):

    base_color = np.array(mcolors.to_rgb(col_mask))
    alpha = np.array([0.6])
    color = np.concatenate([base_color, alpha], axis=0)
    print(f"Using color: {color}")
    h,w = mask.shape[-2:]
    mask_image = mask.reshape(h,w,1) * color.reshape(1,1,-1)
    ax.imshow(mask_image)

def enhanced_pic(image_path, prompt):
    image_file = image_path
    HOME="L:/interactive_generation/GroundingDINO"
    CONFIG_PATH=os.path.join(HOME,"groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))

    WEIGHTS_NAME="groundingdino_swint_ogc.pth"
    WEIGHTS_PATH=os.path.join(HOME, "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, ";exist:", os.path.isfile(WEIGHTS_PATH))
    model= load_model(CONFIG_PATH, WEIGHTS_PATH)

    #IMAGE_NAME = image_file
    #IMAGE_PATH=os.path.join(HOME, "GroundingDINO", IMAGE_NAME)
    print(WEIGHTS_PATH, ";exist:", os.path.isfile(WEIGHTS_PATH))
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)

    #IMAGE_PATH = os.path.join("C:/Users/lamba/Downloads", image_file)
    TEXT_PROMPT = prompt
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25

    image_source, image = load_image(image_file)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    annotated_frame = annotate(image_source=image_source, boxes= boxes, logits = logits, phrases = phrases)

    ax= cv2.imread(image_file)
    h,w,_=ax.shape

    boxes=boxes * torch.Tensor([w,h,w,h])
    xyxy = box_convert(boxes=boxes, in_fmt='cxcywh', out_fmt="xyxy").numpy()

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path= "L:/interactive_generation/sam_vit_h_4b8939.pth"
    sam = sam_model_registry['vit_h'](checkpoint=checkpoint_path).to(device)
    predictor = SamPredictor(sam)
    
    your_img_path=image_file
    your_img = Image.open(your_img_path)
    your_img = np.array(your_img)
    predictor.set_image(your_img)

    input_boxes = torch.tensor(xyxy, device=device)
    print("predictor_GPU:____:", device)

    transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, your_img.shape[:2])
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    your_img_shape= your_img.shape

    fig=plt.figure(figsize=(your_img_shape[1]/100, your_img_shape[0]/100))
    plt.imshow(your_img)
    col_ch1='orange'
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), col_mask=col_ch1)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    print("img__file: ",image_file)
    image_file=os.path.split(image_file)[-1]
    image_file_=image_file.split(".")
    mask_title=image_file_[0]
    print("mask_title: ",mask_title)
    output={}
    mask_save_path="L:/interactive_generation/"+mask_title+".png"
    output["UI"]=mask_title+".png"
    plt.savefig(mask_save_path)
    mask_cpu=masks.cpu().numpy()
    for i, mask_ in enumerate(mask_cpu):
        mask_image = mask_.reshape(your_img.shape[0], your_img.shape[1])
        mask_image = (mask_image > 0).astype(np.uint8) * 255 # converting to binary
        plt.imsave(f'L:/interactive_generation/mask_save/{mask_title}_{i}.tiff', mask_image, cmap='binary', vmin=0, vmax=255)
        out_name="mask_"+str(i)
        output[out_name]=mask_title+"_"+str(i)+".tiff"
        del mask_image
    del mask_cpu
    print("output is____:",output)
    return output
