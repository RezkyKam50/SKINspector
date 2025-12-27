from llama_cpp import Llama
from llama_cpp.llama_chat_format import Qwen25VLChatHandler
 
def get_chat_vl():

    chat_handler = Qwen25VLChatHandler(
        clip_model_path="./models/Qwen2.5-3B-SkinCAP-DoRA/mmproj-Qwen2.5-3b-SkinCAP-DoRA-F16.gguf",
    )
    llm = Llama(
        n_gpu_layers=99,
        model_path="./models/Qwen2.5-3B-SkinCAP-DoRA/Qwen2.5-3B-SkinCAP-DoRA-F16-Q4_K_M.gguf",
        chat_handler=chat_handler,
        flash_attn=True,
        n_threads=4,
        main_gpu=0,
        n_batch=1024,
        n_ctx=8192,
    )

    return chat_handler, llm

def get_chat_llm():

    llm = Llama(
        n_gpu_layers=100,
        model_path="./models/llama-medical-Q4_K_M.gguf", 
        flash_attn=True,
        n_threads=1,
        main_gpu=0,
        n_batch=1024,
        n_ctx=8192,
    )

    return llm
 
