from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
 
def get_chat_vl():

    chat_handler = Qwen25VLChatHandler(
        clip_model_path="./models/mmproj-qwen",
    )
    llm = Llama(
        n_gpu_layers=20,
        model_path="./models/qwen-vl-Q4_K_M.gguf",
        chat_handler=chat_handler,
        flash_attn=True,
        n_threads=1,
        main_gpu=0,
        n_batch=1024,
        n_ctx=8192,
    )

    return chat_handler, llm

def get_chat_llm():

    llm = Llama(
        n_gpu_layers=20,
        model_path="./models/llama-medical-Q4_K_M.gguf", 
        flash_attn=True,
        n_threads=1,
        main_gpu=0,
        n_batch=1024,
        n_ctx=8192,
    )

    return llm

