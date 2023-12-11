import copy
from ikomia import core, dataprocess, utils
import torch
import numpy as np
import random
from diffusers import AutoPipelineForImage2Image
import os
from PIL import Image

# --------------------
# - Class to handle the algorithm parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferKandinsky2Img2imgParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        self.model_name = "kandinsky-community/kandinsky-2-2-decoder"
        self.prompt = "A fantasy landscape, Cinematic lighting"
        self.cuda = torch.cuda.is_available()
        self.prior_guidance_scale = 4.0
        self.guidance_scale = 1.0
        self.negative_prompt = "low quality, bad quality"
        self.height = 768
        self.width = 768
        self.prior_num_inference_steps = 25
        self.num_inference_steps = 100
        self.strength = 0.3
        self.seed = -1
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = str(param_map["model_name"])
        self.prompt = str(param_map["prompt"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.guidance_scale = float(param_map["guidance_scale"])
        self.prior_guidance_scale = float(param_map["prior_guidance_scale"])
        self.negative_prompt = str(param_map["negative_prompt"])
        self.seed = int(param_map["seed"])
        self.height = int(param_map["height"])
        self.width = int(param_map["width"])
        self.num_inference_steps = int(param_map["num_inference_steps"])
        self.prior_num_inference_steps = int(param_map["prior_num_inference_steps"])
        self.strength = float(param_map["strength"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["model_name"] = str(self.model_name)
        param_map["prompt"] = str(self.prompt)
        param_map["negative_prompt"] = str(self.negative_prompt)
        param_map["cuda"] = str(self.cuda)
        param_map["guidance_scale"] = str(self.guidance_scale)
        param_map["prior_guidance_scale"] = str(self.prior_guidance_scale)
        param_map["height"] = str(self.height)
        param_map["width"] = str(self.width)
        param_map["num_inference_steps"] = str(self.num_inference_steps)
        param_map["prior_num_inference_steps"] = str(self.prior_num_inference_steps)
        param_map["strength"] = str(self.strength)
        param_map["seed"] = str(self.seed)

        return param_map


# --------------------
# - Class which implements the algorithm
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferKandinskyImg2img(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the algorithm here
        self.remove_input(1)
        self.remove_output(1)

        # Create parameters object
        if param is None:
            self.set_param_object(InferKandinsky2Img2imgParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.device = torch.device("cpu")
        self.pipe = None
        self.generator = None
        self.seed = None
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")

    def get_progress_steps(self):
        # Function returning the number of progress steps for this algorithm
        # This is handled by the main progress bar of Ikomia Studio
        return 1

    def convert_and_resize_img(self, scr_image, input_height, input_width):
        img = Image.fromarray(scr_image)
        # Stride of 128
        new_width = 128 * (input_width // 128)
        new_height = 128 * (input_height // 128)

        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)

        return resized_img, new_height, new_width

    def load_model(self, param, local_files_only):
        torch_tensor_dtype = torch.float16 if param.cuda and torch.cuda.is_available() else torch.float32
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            param.model_name,
            torch_dtype=torch_tensor_dtype,
            use_safetensors=True,
            cache_dir=self.model_folder,
            local_files_only=local_files_only
            )

        self.pipe.enable_model_cpu_offload()
        param.update = False

    def generate_seed(self, seed):
        if seed == -1:
            self.seed = random.randint(0, 191965535)
        else:
            self.seed = seed
        self.generator = torch.Generator(self.device).manual_seed(self.seed)

    def run(self):
        # Main function of your algorithm
        # Call begin_task_run() for initialization
        self.begin_task_run()

        # Get parameters
        param = self.get_param_object()

        # Load pipeline
        if param.update or self.pipe is None:
            self.device = torch.device("cuda") if param.cuda and torch.cuda.is_available() else torch.device("cpu")

            try:
                self.load_model(param, local_files_only=True)
            except Exception as e:
                self.load_model(param, local_files_only=False)

            self.generate_seed(param.seed)

        # Get image input
        input_image = self.get_input(0).get_image()
        image_guide, height, width = self.convert_and_resize_img(
                                                        input_image,
                                                        param.height,
                                                        param.width
        )

        with torch.no_grad():
            result = self.pipe(prompt=param.prompt,
                          negative_prompt=param.negative_prompt,
                          image=image_guide,
                          prior_num_inference_steps=param.prior_num_inference_steps,
                          prior_guidance_scale=param.prior_guidance_scale,
                          guidance_scale=param.guidance_scale,
                          height=height,
                          width=width,
                          generator=self.generator,
                          num_inference_steps=param.num_inference_steps,
                          strength=param.strength
                          ).images[0]

        print(f"Prompt:\t{param.prompt}\nSeed:\t{self.seed}")

        # Get and display output
        image = np.array(result)
        output_img = self.get_output(0)
        output_img.set_image(image)

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()

        # Step progress bar (Ikomia Studio):
        self.emit_step_progress()

        # Call end_task_run() to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferKandinskyImg2imgFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set algorithm information/metadata here
        self.info.name = "infer_kandinsky_2_img2img"
        self.info.short_description = "Kandinsky 2.2 image-to-image diffusion model."
        # relative path -> as displayed in Ikomia Studio algorithm tree
        self.info.path = "Plugins/Python/Diffusion"
        self.info.version = "1.0.0"
        self.info.icon_path = "images/einstein.jpg"
        self.info.authors = "A. Shakhmatov, A. Razzhigaev, A. Nikolich, V. Arkhipkin, I. Pavlov, A. Kuznetsov, D. Dimitrov"
        self.info.article = "https://aclanthology.org/2023.emnlp-demo.25/"
        self.info.journal = "ACL Anthology"
        self.info.year = 2023
        self.info.license = "Apache 2.0 License"
        # URL of documentation
        self.info.documentation_link = "https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_kandinsky_2_img2img"
        self.info.original_repository = "https://github.com/ai-forever/Kandinsky-2"
        # Keywords used for search
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "IMAGE_GENERATION"
        self.info.keywords = "Latent Diffusion,Hugging Face,Kandinsky,text2image,Generative"
    def create(self, param=None):
        # Create algorithm object
        return InferKandinskyImg2img(self.info.name, param)
