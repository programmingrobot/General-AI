from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from urllib.parse import unquote, urlparse

from download_model import ensure_model


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "5001"))
TEXT_MODEL_ID = os.environ.get("AI_MODEL", "Qwen/Qwen2.5-3B-Instruct")
IMAGE_MODEL_PATH = Path(
    os.environ.get("IMAGE_MODEL_PATH", Path(__file__).resolve().parent / "realisticvision5.1.safetensors")
)
IMAGE_PIPELINE = os.environ.get("IMAGE_PIPELINE", "sd").lower()
IMAGE_STEPS = int(os.environ.get("IMAGE_STEPS", "8"))
IMAGE_WIDTH = int(os.environ.get("IMAGE_WIDTH", "512"))
IMAGE_HEIGHT = int(os.environ.get("IMAGE_HEIGHT", "512"))
IMAGE_SCHEDULER = os.environ.get("IMAGE_SCHEDULER", "dpm").lower()
IMAGE_ATTENTION_SLICING = os.environ.get("IMAGE_ATTENTION_SLICING", "0") == "1"
UNLOAD_TEXT_BEFORE_IMAGE = os.environ.get("UNLOAD_TEXT_BEFORE_IMAGE", "1") != "0"
HF_MAX_NEW_TOKENS = int(os.environ.get("HF_MAX_NEW_TOKENS", "512"))
HF_REQUIRE_CUDA = os.environ.get("HF_REQUIRE_CUDA", "0") == "1"
LOAD_MODEL_ON_START = os.environ.get("LOAD_MODEL_ON_START", "1") != "0"
LOAD_IMAGE_ON_START = os.environ.get("LOAD_IMAGE_ON_START", "0") == "1"
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "40"))
AI_MEMORY_MESSAGES = int(os.environ.get("AI_MEMORY_MESSAGES", "12"))
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    (
        "You are General AI, a local assistant inside an app with chat and image generation. "
        "Answer naturally using recent conversation memory. Image generation is available through the "
        "app's local image model. Treat requests for an image, picture, photo, or drawing as supported "
        "General AI requests and acknowledge them confidently."
    ),
)

ROOT = Path(__file__).resolve().parent
TEMPLATES = ROOT / "templates"
STATIC = ROOT / "static"

hf_tokenizer = None
hf_model = None
image_pipe = None
model_status = "not loaded"
image_status = "not loaded"
model_error: str | None = None
image_error: str | None = None
model_lock = threading.RLock()
image_lock = threading.RLock()


@dataclass
class Message:
    role: str
    content: str
    image: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class Job:
    id: str
    message: str
    status: str = "queued"
    response: str | None = None
    image: str | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


lock = threading.RLock()
jobs: dict[str, Job] = {}
conversation: list[Message] = []


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def parse_version(version_text: str) -> tuple[int, int, int]:
    parts = []
    for item in version_text.split(".")[:3]:
        number = ""
        for char in item:
            if char.isdigit():
                number += char
            else:
                break
        parts.append(int(number or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def visible_history() -> list[dict]:
    with lock:
        return [asdict(message) for message in conversation[-MAX_HISTORY_MESSAGES:]]


def ai_memory() -> list[dict]:
    with lock:
        return [asdict(message) for message in conversation[-AI_MEMORY_MESSAGES:]]


def build_messages(message: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for item in ai_memory():
        if item["role"] in {"user", "assistant"} and item["content"]:
            messages.append({"role": item["role"], "content": item["content"]})
    if not messages or messages[-1].get("content") != message:
        messages.append({"role": "user", "content": message})
    return messages


def release_cuda_memory() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def unload_text_model() -> None:
    global hf_model, hf_tokenizer, model_status
    with model_lock:
        if hf_model is None and hf_tokenizer is None:
            return
        print("Unloading text model before image generation.")
        hf_model = None
        hf_tokenizer = None
        model_status = "not loaded"
    release_cuda_memory()


def load_text_model() -> None:
    global hf_model, hf_tokenizer, model_error, model_status
    with model_lock:
        if hf_model is not None and hf_tokenizer is not None:
            return
        model_status = "loading"
        model_error = None
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            model_status = "error"
            model_error = (
                "Hugging Face text generation needs torch, transformers, accelerate, "
                "sentencepiece, and protobuf installed."
            )
            raise RuntimeError(model_error) from exc

        print(f"Loading Hugging Face text model: {TEXT_MODEL_ID}")
        configure_torch_for_speed(torch)
        if HF_REQUIRE_CUDA and not torch.cuda.is_available():
            model_status = "error"
            model_error = "CUDA was requested, but PyTorch cannot see a CUDA GPU."
            raise RuntimeError(model_error)

        hf_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        options = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available():
            options["torch_dtype"] = torch.float16
            options["device_map"] = {"": "cuda:0"}
        else:
            options["torch_dtype"] = "auto"

        hf_model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL_ID, **options)
        if torch.cuda.is_available():
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for text generation.")
        model_status = "ready"


def ask_huggingface(message: str) -> str:
    load_text_model()
    messages = build_messages(message)
    if hasattr(hf_tokenizer, "apply_chat_template"):
        prompt = hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n".join(f"{item['role']}: {item['content']}" for item in messages) + "\nassistant:"

    inputs = hf_tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(hf_model.device) for key, value in inputs.items()}
    input_tokens = inputs["input_ids"].shape[-1]

    import torch

    with torch.inference_mode():
        output = hf_model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=HF_MAX_NEW_TOKENS,
            pad_token_id=hf_tokenizer.eos_token_id,
        )
    generated = output[0][input_tokens:]
    return hf_tokenizer.decode(generated, skip_special_tokens=True).strip() or "No response generated."


def validate_image_packages() -> None:
    try:
        import transformers
    except ImportError as exc:
        raise RuntimeError("Image generation needs transformers installed.") from exc
    try:
        diffusers_version = package_version("diffusers")
    except PackageNotFoundError as exc:
        raise RuntimeError("Image generation needs diffusers installed.") from exc

    transformers_major = parse_version(transformers.__version__)[0]
    diffusers_major, diffusers_minor, _ = parse_version(diffusers_version)
    if transformers_major >= 5:
        raise RuntimeError("Diffusers single-file checkpoints currently need Transformers 4.x.")
    if diffusers_major == 0 and diffusers_minor > 31:
        raise RuntimeError("This project expects diffusers 0.31.x for the bundled checkpoint.")


def load_image_model() -> None:
    global image_pipe, image_error, image_status
    with image_lock:
        if image_pipe is not None:
            return
        image_status = "loading"
        image_error = None
        if not IMAGE_MODEL_PATH.exists():
            ensure_model()
        if not IMAGE_MODEL_PATH.exists():
            image_status = "error"
            image_error = f"Could not find bundled image model: {IMAGE_MODEL_PATH}"
            raise RuntimeError(image_error)

        try:
            import torch
            from diffusers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline
        except ImportError as exc:
            image_status = "error"
            image_error = "Image generation needs torch, diffusers, safetensors, and pillow installed."
            raise RuntimeError(image_error) from exc

        validate_image_packages()
        configure_torch_for_speed(torch)
        options = {
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "use_safetensors": True,
        }
        print(f"Loading bundled image model: {IMAGE_MODEL_PATH}")
        errors = []
        if IMAGE_PIPELINE in {"auto", "sdxl"}:
            try:
                image_pipe = StableDiffusionXLPipeline.from_single_file(str(IMAGE_MODEL_PATH), **options)
                validate_image_pipeline(image_pipe)
                configure_image_scheduler(image_pipe, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler)
                print("Loaded image model as SDXL.")
            except Exception as exc:
                errors.append(f"SDXL: {exc}")
                image_pipe = None
        if image_pipe is None and IMAGE_PIPELINE in {"auto", "sd", "sd15", "stable-diffusion"}:
            try:
                image_pipe = StableDiffusionPipeline.from_single_file(
                    str(IMAGE_MODEL_PATH),
                    safety_checker=None,
                    requires_safety_checker=False,
                    **options,
                )
                validate_image_pipeline(image_pipe)
                configure_image_scheduler(image_pipe, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler)
                print("Loaded image model as Stable Diffusion.")
            except Exception as exc:
                errors.append(f"Stable Diffusion: {exc}")
                image_pipe = None

        if image_pipe is None:
            image_status = "error"
            image_error = "Could not load realisticvision5.1.safetensors. " + " | ".join(errors)
            raise RuntimeError(image_error)

        if torch.cuda.is_available():
            image_pipe = image_pipe.to("cuda")
            image_pipe.unet.to(memory_format=torch.channels_last)
            if IMAGE_ATTENTION_SLICING:
                try:
                    image_pipe.enable_attention_slicing()
                except Exception:
                    pass
        image_status = "ready"


def configure_torch_for_speed(torch_module) -> None:
    if not torch_module.cuda.is_available():
        return
    torch_module.backends.cuda.matmul.allow_tf32 = True
    torch_module.backends.cudnn.allow_tf32 = True
    torch_module.backends.cudnn.benchmark = True


def configure_image_scheduler(pipe, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler) -> None:
    if IMAGE_SCHEDULER in {"dpm", "dpm++", "dpmsolver", "fast"}:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif IMAGE_SCHEDULER in {"euler", "euler_a", "euler-ancestral"}:
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif IMAGE_SCHEDULER in {"default", "checkpoint"}:
        return
    else:
        raise RuntimeError("IMAGE_SCHEDULER must be 'dpm', 'euler', or 'default'.")


def validate_image_pipeline(pipe) -> None:
    missing = []
    for name in ("unet", "vae", "scheduler", "text_encoder", "tokenizer"):
        if getattr(pipe, name, None) is None:
            missing.append(name)
    if missing:
        raise RuntimeError(f"Loaded image pipeline is missing: {', '.join(missing)}")

    tokenizer = getattr(pipe, "tokenizer", None)
    if not callable(getattr(tokenizer, "tokenize", None)):
        raise RuntimeError(
            "Loaded image pipeline tokenizer is unusable. "
            "Realistic Vision 5.1 should load with IMAGE_PIPELINE=sd."
        )


def generate_image(prompt: str) -> str:
    if UNLOAD_TEXT_BEFORE_IMAGE:
        unload_text_model()
    load_image_model()
    import torch

    started_at = time.monotonic()
    device = getattr(image_pipe, "device", "unknown")
    print(f"Generating image on {device} with {IMAGE_STEPS} steps at {IMAGE_WIDTH}x{IMAGE_HEIGHT}.")
    with torch.inference_mode():
        result = image_pipe(
            prompt=improve_image_prompt(prompt),
            num_inference_steps=IMAGE_STEPS,
            width=IMAGE_WIDTH,
            height=IMAGE_HEIGHT,
        )
    print(f"Image generation finished in {time.monotonic() - started_at:.1f}s.")
    image = result.images[0]
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def is_image_request(message: str) -> bool:
    lowered = message.lower()
    image_words = (
        "generate an image",
        "make an image",
        "create an image",
        "draw",
        "picture of",
        "image of",
        "photo of",
    )
    return any(word in lowered for word in image_words)


def improve_image_prompt(prompt: str) -> str:
    return (
        f"{prompt}, high quality, clear subject, coherent composition, detailed lighting, "
        "sharp focus, visually polished"
    )


def run_job(job_id: str) -> None:
    with lock:
        job = jobs[job_id]
        job.status = "processing"
        job.updated_at = time.time()
        message = job.message

    try:
        if is_image_request(message):
            with lock:
                jobs[job_id].status = "generating_image"
                jobs[job_id].updated_at = time.time()
            image = generate_image(message)
            response = "Here is the generated image."
            with lock:
                job = jobs[job_id]
                job.response = response
                job.image = image
                job.status = "done"
                job.updated_at = time.time()
                conversation.append(Message(role="assistant", content=response, image=image))
        else:
            response = ask_huggingface(message)
            with lock:
                job = jobs[job_id]
                job.response = response
                job.status = "done"
                job.updated_at = time.time()
                conversation.append(Message(role="assistant", content=response))
    except Exception as exc:
        with lock:
            job = jobs[job_id]
            job.error = str(exc)
            job.status = "error"
            job.updated_at = time.time()
            conversation.append(Message(role="assistant", content=job.error))


def safe_static_path(raw_path: str) -> Path | None:
    relative = unquote(raw_path.removeprefix("/static/"))
    candidate = (STATIC / relative).resolve()
    try:
        candidate.relative_to(STATIC.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


class LocalAIHandler(BaseHTTPRequestHandler):
    server_version = "GeneralAILocal/1.0"

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path: Path, content_type: str | None = None) -> None:
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.send_file(TEMPLATES / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/"):
            static_path = safe_static_path(path)
            if static_path:
                self.send_file(static_path)
            else:
                self.send_json({"error": "File not found."}, HTTPStatus.NOT_FOUND)
            return
        if path == "/health":
            with lock:
                pending = sum(1 for job in jobs.values() if job.status in {"queued", "processing", "generating_image"})
                total = len(jobs)
            self.send_json(
                {
                    "ok": True,
                    "host": HOST,
                    "port": PORT,
                    "pending": pending,
                    "jobs": total,
                    "text_model": TEXT_MODEL_ID,
                    "text_model_status": model_status,
                    "text_model_error": model_error,
                    "image_model": str(IMAGE_MODEL_PATH),
                    "image_model_status": image_status,
                    "image_model_error": image_error,
                }
            )
            return
        if path == "/history":
            self.send_json({"messages": visible_history()})
            return
        if path.startswith("/chat/"):
            job_id = path.removeprefix("/chat/").strip()
            with lock:
                job = jobs.get(job_id)
                payload = asdict(job) if job else None
            if not payload:
                self.send_json({"error": "Unknown job."}, HTTPStatus.NOT_FOUND)
                return
            self.send_json(payload)
            return
        self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/chat":
            self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json()
        except json.JSONDecodeError:
            self.send_json({"error": "Invalid JSON."}, HTTPStatus.BAD_REQUEST)
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self.send_json({"error": "Message is required."}, HTTPStatus.BAD_REQUEST)
            return

        job = Job(id=uuid.uuid4().hex, message=message)
        with lock:
            jobs[job.id] = job
            conversation.append(Message(role="user", content=message))

        thread = threading.Thread(target=run_job, args=(job.id,), daemon=True)
        thread.start()
        self.send_json({"id": job.id, "status": job.status}, HTTPStatus.ACCEPTED)


def local_addresses() -> list[str]:
    addresses = ["localhost"]
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            address = info[4][0]
            if address not in addresses:
                addresses.append(address)
    except OSError:
        pass
    return addresses


def preload_models() -> None:
    if LOAD_MODEL_ON_START:
        threading.Thread(target=_preload_text_model, daemon=True).start()
    if LOAD_IMAGE_ON_START:
        threading.Thread(target=_preload_image_model, daemon=True).start()


def load_startup_models() -> None:
    if LOAD_MODEL_ON_START:
        load_text_model()
    if LOAD_IMAGE_ON_START:
        load_image_model()


def _preload_text_model() -> None:
    try:
        load_text_model()
    except Exception as exc:
        print(f"Text model preload failed: {exc}")


def _preload_image_model() -> None:
    try:
        load_image_model()
    except Exception as exc:
        print(f"Image model preload failed: {exc}")


def main() -> int:
    print("General AI local server")
    print(f"Serving from: {ROOT}")
    print(f"Text model: {TEXT_MODEL_ID}")
    print(f"Bundled image model: {IMAGE_MODEL_PATH}")
    print(f"Load text model on start: {LOAD_MODEL_ON_START}")
    print(f"Load image model on start: {LOAD_IMAGE_ON_START}")
    load_startup_models()

    httpd = ThreadingHTTPServer((HOST, PORT), LocalAIHandler)
    print(f"Binding to: host={HOST} port={PORT}")
    print("Open one of these URLs:")
    for address in local_addresses():
        print(f"  http://{address}:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        httpd.server_close()
        release_cuda_memory()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
