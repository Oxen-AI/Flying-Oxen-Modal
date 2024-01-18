from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from modal import Image, Stub, asgi_app

web_app = FastAPI()
stub = Stub('Hello-World-Test')
image = Image.debian_slim()


@web_app.post("/foo")
async def foo(request: Request):
    return 'Hello World!'

@web_app.post("/")
async def hello(request: Request):
    return {"status": "success", "status_message": "Hello World!"}


@stub.function(image=image)
@asgi_app()
def fastapi_app():
    return web_app
