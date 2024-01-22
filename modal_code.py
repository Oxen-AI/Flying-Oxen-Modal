import io
from pathlib import Path
import random
import uuid
from fastapi import FastAPI, Request
from oxen import RemoteRepo

from modal import Image, Mount, Stub, asgi_app, build, enter, gpu, method

sdxl_image = (
    Image.debian_slim()
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    .pip_install(
        "diffusers~=0.19",
        "invisible_watermark~=0.1",
        "transformers~=4.31",
        "accelerate~=0.21",
        "safetensors~=0.3",
        "oxenai",
    )
)

stub = Stub("oxen-stable-diffusion-test")

with sdxl_image.imports():
    import fastapi.staticfiles
    import torch
    from diffusers import DiffusionPipeline
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    from huggingface_hub import snapshot_download

@stub.function(image=sdxl_image)
def generate_random_prompt(username):
    '''
    generate the 
    '''
    adjectives = ["grand", "noble", "splendid", "majestic", "regal", "royal",
                     "stately", "august", "magnificent", "dignified", "imposing",
                     "sublime", "resplendent", "glorious", "impressive", "exalted",
                     "heroic", "grandiose", "epic", "monumental", "palatial",
                     "awesome", "breathtaking", "marvelous", "sumptuous", "funny"
                     "wonderful", "excellent", "terrific", "fantastic", "fabulous",
                     "charismatic"]

    colors = [
        "red", "orange", "yellow", "green", "blue", "purple", "brown", "black", "white", "gray"
    ]

    locations = [
        "the clouds", "space", "the sky", "the ocean", "the sea", "the mountains", "the forest", "the desert", "the jungle", "the savannah", "the tundra", "the arctic", "the antarctic", "the moon", "the sun", "the stars", "the galaxy", "the universe", "the matrix"
    ]

    styles = ["cartoon", "vibrant colors", "Geometric shapes", "Abstract patterns", "Movement and flow", "Texture and layers", "dreamlike", "surreal landscapes", "mystical creatures", "simplicity", "minimal", "negative space", "impressionist", "painted", "pastel colors", "digital art", "realistic portrait", "realistic landscape", "bold colors", "stylized portrait", "pop art still life", "black and white", "night time", "daylight", "street art", "fauvism", "graffiti art", "modernism", "abstract art", "action packed", "pop art", "neo-dada", "water color", "conceptual art", "fluxus", "cubism"]

    adjective = random.choice(adjectives)
    color = random.choice(colors)
    location = random.choice(locations)
    style = random.choice(styles)
    prompt = f"{username}, {adjective} {color} ox with wings flying through {location}, {style}"
    # file_names = imagine(prompt)

    # commit_message = f"{username} gave an Ox it's wings"
    # save_to_oxen("ox/FlyingOxen", "main", prompt, file_names, commit_message)
    print(prompt)
    return prompt

@stub.function(image=sdxl_image)
def save_to_oxen(repo_name, branch_name, prompt, file_names, commit_message):
    '''
    save the file to oxen repo
    '''

    repo = RemoteRepo(repo_name)
    repo.checkout(branch_name)

    # Add all files in one commmit
    paths = []
    for file_name in file_names:
        repo.add(file_name, "images")
        repo_path = f"images/{file_name}"
        paths.append(repo_path)
        repo.add_df_row("annotations.jsonl", {"prompt": prompt, "image": repo_path, "model": model_id})
                                                             
    commit = repo.commit(commit_message)
    images = []
    for path in paths:
        images.append({
            "path": path,
            "content_url": f"https://hub.oxen.ai/api/repos/{repo_name}/file/{commit.commit_id}/{path}",
            # https://www.oxen.ai/ox/GenerativeImagePlayground/file/edfdd2bc11396d86/images/a-white-fluffy-ox-427b3fa6-fed2-4379-b9a8-04babf476bc8.png
            "view_url": f"https://oxen.ai/{repo_name}/file/{commit.commit_id}/{path}",
        })
    print(f"Got Commit: {commit}")
    os.remove(file_name)
    return commit, images    


@stub.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=sdxl_image)
class Model:
    @build()
    def build(self):
        ignore = [
            "*.bin",
            "*.onnx_data",
            "*/diffusion_pytorch_model.safetensors",
        ]
        snapshot_download(
            "stabilityai/stable-diffusion-xl-base-1.0", ignore_patterns=ignore
        )

    @enter()
    def enter(self):
        load_options = dict(
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            device_map="auto",
        )

        # Load base model
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", **load_options
        )

    @method()
    def imagine(self, prompt, num_images=1, num_steps=50, width=1024, height=1024):
        file_base_name = str(prompt).split(" ")[:10]
        print(f"Generating {str(prompt)}")

        images = self.pipe(
            prompt,
            height=height,
            width=width,
            num_images_per_prompt=num_images,
            num_inference_steps=num_steps
        ).images   # type: ignore
        print(images)
        file_names = []

        for image in images:
            file_name = "-".join(file_base_name) + "-" + str(uuid.uuid4()) + ".png"
            image.save(file_name)
            file_names.append(file_name)
        # print(file_names)
        return file_names   


app = FastAPI()

@app.post("/")
async def generate_image(
    request: Request
):
    data = await request.json()
    print(f"POST /imagine - received data={data}")

    if data['action'] == 'created': 
        data['prompt'] = generate_random_prompt.remote(data['sender']['login'])
    
    if data['prompt']:
        prompt = data['prompt']
        file_names = Model().imagine.remote(prompt)
        print("Saving to Oxen")
        
        commit_message = f"{prompt}"
        namespace = "ox"
        repo_name = "FlyingOxen"
        branch_name = "main"
        commit, images = save_to_oxen.remote(f"{namespace}/{repo_name}", branch_name, prompt, file_names, commit_message)
        commit_id = commit.commit_id
        return {
            "status": "success",
            "status_message": "Committed: " + commit.message,
            "commit": {
                "url": f"https://www.oxen.ai/{namespace}/{repo_name}/commit/{commit_id}",
                "id": commit_id,
                "message": commit.message
            },
            "images": images
        }
    else:
        return {"status": "error", "status_message": "Must supply 'prompt' field in json"}



@app.post("/echo")
async def foo(request: Request):
    data = await request.json()
    return data

@stub.function(image=sdxl_image)
@asgi_app()
def fastapi_app():
    return app

