import runpod
from runpod.serverless.utils import rp_upload
import os
import shutil
import websocket
import base64
import json
import uuid
import logging
import urllib.request
import urllib.error
import urllib.parse
import binascii


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CUDA 검사 및 설정
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Fatal error: {e}")
    logger.error("Exiting due to CUDA requirements not met")
    exit(1)


server_address = os.getenv('SERVER_ADDRESS', '127.0.0.1')
client_id = str(uuid.uuid4())

# Pasta que o LoadImage do ComfyUI efetivamente lê
COMFY_INPUT_DIR = "/ComfyUI/input"

def save_data_if_base64(data_input, output_filename="input_image.jpg"):
    """
    FIX: Base64 (com ou sem prefixo data URI) e salvo em /ComfyUI/input/,
    retornando APENAS o nome do arquivo (o que o nó LoadImage espera).
    Antes, o prefixo "data:image/...;base64," quebrava o b64decode e o
    valor cru ia para o LoadImage -> HTTP 400 em todo request.
    """
    if not isinstance(data_input, str):
        return data_input

    raw = data_input
    if raw.startswith("data:"):
        raw = raw.split(",", 1)[1]

    try:
        decoded_data = base64.b64decode(raw)
        os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
        file_path = os.path.join(COMFY_INPUT_DIR, output_filename)
        with open(file_path, 'wb') as f:
            f.write(decoded_data)
        print(f"✅ Base64 salvo em '{file_path}'")
        return output_filename
    except (binascii.Error, ValueError):
        print(f"➡️ '{data_input[:60]}...' tratado como caminho/URL")
        return data_input

def queue_prompt(prompt):
    url = f"http://{server_address}:8188/prompt"
    logger.info(f"Queueing prompt to: {url}")
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    # FIX: Content-Type correto + log do body do 400 (diagnóstico)
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        return json.loads(urllib.request.urlopen(req).read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "ignore")
        print("ComfyUI /prompt HTTPError:", e.code, e.reason)
        print("ComfyUI /prompt body:", body)
        raise

def get_image(filename, subfolder, folder_type):
    url = f"http://{server_address}:8188/view"
    logger.info(f"Getting image from: {url}")
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen(f"{url}?{url_values}") as response:
        return response.read()

def get_history(prompt_id):
    url = f"http://{server_address}:8188/history/{prompt_id}"
    logger.info(f"Getting history from: {url}")
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break
        else:
            continue

    history = get_history(prompt_id)[prompt_id]
    for node_id in history['outputs']:
        node_output = history['outputs'][node_id]
        images_output = []
        if 'images' in node_output:
            for image in node_output['images']:
                image_data = get_image(image['filename'], image['subfolder'], image['type'])
                if isinstance(image_data, bytes):
                    import base64
                    image_data = base64.b64encode(image_data).decode('utf-8')
                images_output.append(image_data)
        output_images[node_id] = images_output

    return output_images

def load_workflow(workflow_path):
    with open(workflow_path, 'r') as file:
        return json.load(file)

def handler(job):
    job_input = job.get("input", {})

    logger.info(f"Received job input: {job_input}")
    task_id = f"task_{uuid.uuid4()}"

    image_input = job_input["image_path"]

    # FIX: suporta os 4 formatos de image_path
    if image_input == "/example_image.png":
        os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
        src = "/example_image.png"
        dst = os.path.join(COMFY_INPUT_DIR, "example_image.png")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
        image_path = "example_image.png"
    elif image_input.startswith("http://") or image_input.startswith("https://"):
        os.makedirs(COMFY_INPUT_DIR, exist_ok=True)
        urllib.request.urlretrieve(
            image_input, os.path.join(COMFY_INPUT_DIR, "input_image.jpg")
        )
        image_path = "input_image.jpg"
    else:
        image_path = save_data_if_base64(image_input)

    prompt = load_workflow("/flux_kontext_example.json")

    prompt["41"]["inputs"]["image"] = image_path
    prompt["6"]["inputs"]["text"] = job_input["prompt"]
    prompt["25"]["inputs"]["noise_seed"] = job_input["seed"]
    prompt["26"]["inputs"]["guidance"] = job_input["guidance"]
    prompt["27"]["inputs"]["width"] = job_input["width"]
    prompt["27"]["inputs"]["height"] = job_input["height"]
    prompt["30"]["inputs"]["width"] = job_input["width"]
    prompt["30"]["inputs"]["height"] = job_input["height"]

    ws_url = f"ws://{server_address}:8188/ws?clientId={client_id}"
    logger.info(f"Connecting to WebSocket: {ws_url}")

    http_url = f"http://{server_address}:8188/"
    logger.info(f"Checking HTTP connection to: {http_url}")

    max_http_attempts = 180
    for http_attempt in range(max_http_attempts):
        try:
            import urllib.request
            response = urllib.request.urlopen(http_url, timeout=5)
            logger.info(f"HTTP 연결 성공 (시도 {http_attempt+1})")
            break
        except Exception as e:
            logger.warning(f"HTTP 연결 실패 (시도 {http_attempt+1}/{max_http_attempts}): {e}")
            if http_attempt == max_http_attempts - 1:
                raise Exception("ComfyUI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
            time.sleep(1)

    ws = websocket.WebSocket()
    max_attempts = int(180/5)
    for attempt in range(max_attempts):
        import time
        try:
            ws.connect(ws_url)
            logger.info(f"웹소켓 연결 성공 (시도 {attempt+1})")
            break
        except Exception as e:
            logger.warning(f"웹소켓 연결 실패 (시도 {attempt+1}/{max_attempts}): {e}")
            if attempt == max_attempts - 1:
                raise Exception("웹소켓 연결 시간 초과 (3분)")
            time.sleep(5)
    images = get_images(ws, prompt)
    ws.close()

    if not images:
        return {"error": "이미지를 생성할 수 없습니다."}

    for node_id in images:
        if images[node_id]:
            return {"image": images[node_id][0]}

    return {"error": "이미지를 찾을 수 없습니다."}

runpod.serverless.start({"handler": handler})
