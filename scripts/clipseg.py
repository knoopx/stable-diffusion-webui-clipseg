from abc import ABC
from enum import Enum
from clipseg.clipseg import CLIPDensePredT
from kornia.morphology import dilation, erosion
from math import floor
from modules.processing import StableDiffusionProcessingImg2Img, process_images
from pathlib import Path
from PIL import ImageDraw, ImageChops, Image
from pyparsing import ParserElement
from torchvision import transforms
from typing import Optional, Sequence
import cv2
import gradio as gr
import modules.scripts as scripts
import numpy as np
import operator
import os.path
import PIL.Image
import pyparsing as pp
import requests
import torch
import torchvision.transforms.functional as F

ParserElement.enablePackrat()


class Mask(ABC):
    def get_mask_for_image(self, img):
        pass

    def gather_text_descriptions(self):
        return set()

    def apply_masks(self, mask_cache):
        pass


class SimpleMask(Mask):
    def __init__(self, text):
        self.text = text

    @classmethod
    def from_simple_prompt(cls, instring, tokens_start, ret_tokens):
        return cls(text=ret_tokens[0])

    def __repr__(self):
        return f"'{self.text}'"

    def gather_text_descriptions(self):
        return {self.text}

    def apply_masks(self, mask_cache):
        return mask_cache[self.text]


class ModifiedMask(Mask):
    ops = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        # '%': operator.mod,
        # '^': operator.xor,
    }

    def __init__(self, mask, modifier):
        if modifier:
            modifier = modifier.strip("{}")
        self.mask = mask
        self.modifier = modifier
        self.operand_str = modifier[0]
        self.operand = self.ops[self.operand_str]
        self.value = float(modifier[1:])

    @classmethod
    def from_modifier_parse(cls, instring, tokens_start, ret_tokens):
        return cls(mask=ret_tokens[0][0], modifier=ret_tokens[0][1])

    def __repr__(self):
        return f"{repr(self.mask)}{self.modifier}"

    def gather_text_descriptions(self):
        return self.mask.gather_text_descriptions()

    def apply_masks(self, mask_cache):
        mask = self.mask.apply_masks(mask_cache)
        if self.operand_str in {"+", "-"}:
            # kernel must be odd
            kernel_size = int(round(self.value))
            kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
            morph_method = dilation if self.operand_str == "+" else erosion
            mask = mask.unsqueeze_(0).unsqueeze_(0)
            mask = morph_method(mask, torch.ones(kernel_size, kernel_size))
            mask = mask.squeeze()
            return mask
        return torch.clamp(self.operand(mask, self.value), 0, 1)


class NestedMask(Mask):
    def __init__(self, masks, op):
        self.masks = masks
        self.op = op

    @classmethod
    def from_or(cls, instring, tokens_start, ret_tokens):
        sub_masks = [t for t in ret_tokens[0] if isinstance(t, Mask)]
        return cls(masks=sub_masks, op="OR")

    @classmethod
    def from_and(cls, instring, tokens_start, ret_tokens):
        sub_masks = [t for t in ret_tokens[0] if isinstance(t, Mask)]
        return cls(masks=sub_masks, op="AND")

    @classmethod
    def from_not(cls, instring, tokens_start, ret_tokens):
        sub_masks = [t for t in ret_tokens[0] if isinstance(t, Mask)]
        assert len(sub_masks) == 1
        return cls(masks=sub_masks, op="NOT")

    def __repr__(self):
        if self.op == "NOT":
            return f"NOT {self.masks[0]}"
        sub = f" {self.op} ".join(repr(m) for m in self.masks)
        return f"({sub})"

    def gather_text_descriptions(self):
        return set().union(*[m.gather_text_descriptions() for m in self.masks])

    def apply_masks(self, mask_cache):
        submasks = [m.apply_masks(mask_cache) for m in self.masks]
        mask = submasks[0]
        if self.op == "OR":
            for submask in submasks:
                mask = torch.maximum(mask, submask)
        elif self.op == "AND":
            for submask in submasks:
                mask = torch.minimum(mask, submask)
        elif self.op == "NOT":
            mask = 1 - mask
        else:
            raise ValueError(f"Invalid operand {self.op}")
        return torch.clamp(mask, 0, 1)


AND = (pp.Literal("AND") | pp.Literal("&")).setName("AND").setResultsName("op")
OR = (pp.Literal("OR") | pp.Literal("|")).setName("OR").setResultsName("op")
NOT = (pp.Literal("NOT") | pp.Literal("!")).setName("NOT").setResultsName("op")

PROMPT_MODIFIER = (
    pp.Regex(r"{[*/+-]\d+\.?\d*}")
    .setName("prompt_modifier")
    .setResultsName("prompt_modifier")
)

PROMPT_TEXT = (
    pp.Regex(r"[a-z0-9]?[a-z0-9 -]*[a-z0-9]")
    .setName("prompt_text")
    .setResultsName("prompt_text")
)

SIMPLE_PROMPT = PROMPT_TEXT.setResultsName("simplePrompt")
SIMPLE_PROMPT.setParseAction(SimpleMask.from_simple_prompt)

COMPLEX_PROMPT = pp.infixNotation(
    SIMPLE_PROMPT,
    [
        (PROMPT_MODIFIER, 1, pp.opAssoc.LEFT, ModifiedMask.from_modifier_parse),
        (NOT, 1, pp.opAssoc.RIGHT, NestedMask.from_not),
        (AND, 2, pp.opAssoc.LEFT, NestedMask.from_and),
        (OR, 2, pp.opAssoc.LEFT, NestedMask.from_or),
    ],
)

MASK_PROMPT = pp.Group(COMPLEX_PROMPT).setResultsName("complexPrompt")

RED_COLOR = "#FF0000"
GREEN_COLOR = "#00FF00"


class BrushMaskMode(Enum):
    Add = 0
    Substract = 1
    Discard = 2


def base_path(*parts):
    filename = os.path.join(scripts.basedir(), *parts)
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    return filename


def download_file(filename, url):
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        for block in response.iter_content(4096):
            fout.write(block)


def download_weights(file):
    print("Downloading clipseg model weights...")
    # https://github.com/timojl/clipseg/raw/master/weights/rd64-uni.pth
    download_file(
        file, "https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download?path=%2F&files=rd64-uni-refined.pth")


def model_path():
    path = Path("./models/clipseg/rd64-uni-refined.pth")
    if not os.path.exists(path):
        # Download model weights if we don't have them yet
        path.parent.mkdir(parents=True, exist_ok=True)
        download_weights(path)
    return path


def load_model():
    # load model
    model = CLIPDensePredT(
        version='ViT-B/16',
        reduce_dim=64,
        complex_trans_conv=True)

    model.eval()

    # non-strict, because we only stored decoder weights (not CLIP weights)
    model.load_state_dict(torch.load(
        model_path(), map_location=torch.device('cuda')), strict=False)

    return model


def fit_image_within(
    image: PIL.Image.Image, max_height, max_width, convert="RGB", snap_size=8
):
    image = image.convert(convert)
    w, h = image.size
    resize_ratio = 1
    if w > max_width or h > max_height:
        resize_ratio = min(max_width / w, max_height / h)
    elif w < max_width and h < max_height:
        # it's smaller than our target image, enlarge
        resize_ratio = max(max_width / w, max_height / h)

    if resize_ratio != 1:
        w, h = int(w * resize_ratio), int(h * resize_ratio)
    # resize to integer multiple of snap_size
    w -= w % snap_size
    h -= h % snap_size

    if (w, h) != image.size:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    return image


def apply_brush_mask(image: Image.Image, brush_mask: Optional[Image.Image], brush_mask_mode: BrushMaskMode):
    if brush_mask:
        # brush_mask = brush_mask.resize(image.size).convert("L")
        if brush_mask_mode == BrushMaskMode.Add:
            return ImageChops.add(image, brush_mask)
        elif brush_mask_mode == BrushMaskMode.Substract:
            return ImageChops.subtract(image, brush_mask)

    return image


def debug_mask(img: Image.Image, color: str, alpha=0.25):
    mask = img.convert('L')
    result = img.convert("RGBA")
    draw = ImageDraw.Draw(result)
    draw.rectangle(((0, 0), img.size), fill=color)
    result.putalpha(mask.point(lambda i: floor(i*alpha)))
    return result


def mask_preview(image, pred_mask, brush_mask, brush_mask_mode, inpainting_mask_invert):
    add = RED_COLOR
    substract = GREEN_COLOR

    if inpainting_mask_invert:
        add = GREEN_COLOR
        substract = RED_COLOR

    result = image.convert("RGBA")

    if pred_mask:
        result = Image.alpha_composite(result, debug_mask(pred_mask, add))

    if brush_mask:
        if brush_mask_mode == 1:
            result = Image.alpha_composite(
                result, debug_mask(brush_mask, add))
        elif brush_mask_mode == 2:
            result = Image.alpha_composite(
                result, debug_mask(brush_mask, substract))

    return result.convert("RGB")


class Script(scripts.Script):
    def title(self):
        return "clipseg"

    def show(self, is_img2img: bool):
        return is_img2img

    def ui(self, is_img2img: bool):
        with gr.Group():
            with gr.Row():
                mask_prompt = gr.Textbox(label="Prompt", lines=1)

            with gr.Row():
                brush_mask_mode = gr.Radio(
                    label="Brush Mask",
                    choices=['Add', 'Substract', 'Discard'],
                    value='Add',
                    type="index"
                )

                debug = gr.Checkbox(label="Debug", value=False)

        return mask_prompt, brush_mask_mode, debug

    def get_img_mask(
        self,
        img: PIL.Image.Image,
        prompt: str,
        max_height, max_width,
        threshold: Optional[float] = None,
    ):
        parsed = MASK_PROMPT.parseString(prompt)
        parsed_mask = parsed[0][0]
        descriptions = list(parsed_mask.gather_text_descriptions())
        orig_size = img.size
        img = fit_image_within(img, max_height, max_width)
        mask_cache = self.get_img_masks(img, descriptions)
        mask = parsed_mask.apply_masks(mask_cache)

        kernel = np.ones((3, 3), np.uint8)
        mask_g = mask.clone()

        # trial and error shows 0.5 threshold has the best "shape"
        if threshold is not None:
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1

        mask_np = mask.to(torch.float32).cpu().numpy()
        smoother_strength = 2
        # grow the mask area to make sure we've masked the thing we care about
        for _ in range(smoother_strength):
            mask_np = cv2.dilate(mask_np, kernel)
        # todo: add an outer blur (not gaussian)
        mask = torch.from_numpy(mask_np)

        mask_img = F.to_pil_image(mask).resize(
            orig_size, resample=PIL.Image.Resampling.LANCZOS
        )
        mask_img_g = F.to_pil_image(mask_g).resize(
            orig_size, resample=PIL.Image.Resampling.LANCZOS
        )
        return mask_img, mask_img_g

    def get_img_masks(self, img, mask_descriptions: Sequence[str]):
        a, b = img.size
        orig_size = b, a

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225]),
                transforms.Resize((352, 352)),
            ]
        )
        img = transform(img).unsqueeze(0)

        model = load_model()
        with torch.no_grad():
            preds = model(
                img.repeat(len(mask_descriptions), 1, 1, 1), mask_descriptions
            )[0]
        preds = transforms.Resize(orig_size)(preds)

        preds = [torch.sigmoid(p[0]) for p in preds]

        preds_dict = {}
        for p, desc in zip(preds, mask_descriptions):
            preds_dict[desc] = p

        return preds_dict

    def run(self, pipeline: StableDiffusionProcessingImg2Img, mask_prompt: str, brush_mask_mode: BrushMaskMode, debug: bool):
        image = pipeline.init_images[0]
        mask, mask_grayscale = self.get_img_mask(
            image, mask_prompt,
            threshold=0.1,
            max_width=pipeline.width,
            max_height=pipeline.height,
        )

        brush_mask = pipeline.image_mask
        final_mask = apply_brush_mask(
            mask, brush_mask, BrushMaskMode(brush_mask_mode))

        pipeline.image_mask = final_mask
        pipeline.mask_for_overlay = final_mask
        pipeline.latent_mask = None

        processed = process_images(pipeline)

        if debug:
            processed.images.append(mask)
            if brush_mask:
                processed.images.append(brush_mask)
            processed.images.append(final_mask)
            processed.images.append(mask_preview(
                image,
                mask,
                brush_mask,
                brush_mask_mode,
                inpainting_mask_invert=pipeline.inpainting_mask_invert
            ))

        return processed
