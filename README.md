# General AI Local

A small local AI web console. The server serves the browser interface and runs chat requests in background threads from the same process.

## Run

Windows:

```bat
run.bat
```

macOS/Linux:

```sh
sh run.sh
```

Direct Python:

```sh
python3 app.py
```

The app binds to `host=0.0.0.0` by default and prints the URLs it is running on.

## Local AI

The server starts `Qwen/Qwen2.5-3B-Instruct` with Hugging Face Transformers. On first run, the launch script downloads `realisticvision5.1.safetensors` automatically and verifies its SHA256 before image generation uses it.

Optional environment variables:

- `PORT`: server port, default `5001`
- `HOST`: bind host, default `0.0.0.0`
- `AI_MODEL`: text model, default `Qwen/Qwen2.5-3B-Instruct`
- `IMAGE_MODEL_PATH`: image checkpoint, default `realisticvision5.1.safetensors`
- `IMAGE_MODEL_URL`: image checkpoint download URL
- `IMAGE_PIPELINE`: image pipeline type, default `sd` for Realistic Vision 5.1
- `IMAGE_STEPS`: image generation steps, default `8`
- `IMAGE_SCHEDULER`: image scheduler, default `dpm`
- `UNLOAD_TEXT_BEFORE_IMAGE`: free text-model VRAM before image generation, default `1`
- `LOAD_MODEL_ON_START`: preload the text model, default `1`
- `LOAD_IMAGE_ON_START`: preload the image model, default `0`

The server binds to `host=0.0.0.0`, prints the URLs it is running on, serves the web UI itself, and runs the local AI jobs in the same process.
