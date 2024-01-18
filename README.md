# Flying-Oxen-Modal
Implementation of generating a flying oxen using modal.com
## Running instruction
1. Install modal python package using 
```
pip install modal 
python3 -m modal setup
```
2. Generate a token [here](https://modal.com/oxen-ai/settings/tokens) and config the token secret.
3. Deploy the script with 
```
modal deploy modal_code.py
``` 
If success, it will return the app url.

4. Hit the api with curl, eg 
```
curl -X POST https://modal.com/apps/oxen-ai/oxen-stable-diffusion-test/test
```
