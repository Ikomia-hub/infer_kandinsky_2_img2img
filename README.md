<div align="center">
  <img src="images/einstein.jpg" alt="Algorithm icon">
  <h1 align="center">infer_kandinsky_2_img2img</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_kandinsky_2_img2img">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_kandinsky_2_img2img">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_kandinsky_2_img2img/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_kandinsky_2_img2img.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Kandinsky 2.2 image-to-image is a text-conditional diffusion model based on unCLIP and latent diffusion, composed of a transformer-based image prior model, a unet diffusion model, and a decoder.


*Note: This algorithm requires 10GB GPU RAM*

![original image](https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg)

![output image](https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/fantasy_land.png)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_kandinsky_2_img2img", auto_connect=False)

# Run directly on your image
wf.run()

# Display the image
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'kandinsky-community/kandinsky-2-2-decoder': Name of the latent diffusion model. 
- **prompt** (str) - default 'portrait of a young women, blue eyes, cinematic' : Text prompt to guide the image generation .
- **negative_prompt** (str, *optional*) - default 'low quality, bad quality': The prompt not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
- **prior_num_inference_steps** (int) - default '25': Number of denoising steps of the prior model (CLIP).
- **prior_guidance_scale** (float) - default '4.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **num_inference_steps** (int) - default '100': The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
- **guidance_scale** (float) - default '1.0':  Higher guidance scale encourages to generate images that are closely linked to the text prompt, usually at the expense of lower image quality. (minimum: 1; maximum: 20).
- **strength** (int) - default '0.2':  Conceptually, indicates how much to transform the reference image. Must be between 0 and 1. image will be used as a starting point, adding more noise to it the larger the strength. The number of denoising steps depends on the amount of noise initially added. When strength is 1, added noise will be maximum and the denoising process will run for the full number of iterations specified in num_inference_steps. A value of 1, therefore, essentially ignores image.
- **height** (int) - default '768: The height in pixels of the generated image.
- **width** (int) - default '768: The width in pixels of the generated image.
- **seed** (int) - default '-1': Seed value. '-1' generates a random number between 0 and 191965535.


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name = "infer_kandinsky_2_img2img", auto_connect=True)

algo.set_parameters({
    'model_name': 'kandinsky-community/kandinsky-2-2-decoder',
    'prompt': 'A fantasy landscape, Cinematic lighting',
    'negative_prompt': 'low quality, bad quality',
    'prior_num_inference_steps': '25',
    'prior_guidance_scale': '4.0',
    'num_inference_steps': '100',
    'guidance_scale': '4.0',
    'strength': '0.4',
    'seed': '1231689',
    'width': '768',
    'height': '512',
    })

# Run
wf.run_on(url="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg")

# Display the image
display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_kandinsky_2_img2img", auto_connect=True)

# Run on your image  
wf.run_on(url="https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```