# stable-diffusion-webui-clipseg

Automatically create masks for inpainting with Stable Diffusion using natural language.

## Introduction

stable-diffusion-webui-clipseg is an addon for [AUTOMATIC1111's Stable Diffusion Web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows you to enter a text string in img2img mode which automatically creates an image mask.

## Usage

From the img2img screen, select clipseg as your active script:

In the `Prompt` field, enter the text to search for within your image:

Mask syntax:

- mask descriptions must be lowercase
- keywords (`AND`, `OR`, `NOT`) must be uppercase
- parentheses are supported
- mask modifiers may be appended to any mask or group of masks. Example: `(dog OR cat){+5}` means that we'll
  select any dog or cat and then expand the size of the mask area by 5 pixels. Valid mask modifiers: - `{+n}` - expand mask by n pixels - `{-n}` - shrink mask by n pixels - `{*n}` - multiply mask strength. will expand mask to areas that weakly matched the mask description - `{/n}` - divide mask strength. will reduce mask to areas that most strongly matched the mask description. probably not useful

When writing strength modifiers keep in mind that pixel values are between 0 and 1.

# Credits

- https://github.com/brycedrennan/imaginAIry/
- https://github.com/ThereforeGames/txt2mask/
- https://github.com/timojl/clipseg
