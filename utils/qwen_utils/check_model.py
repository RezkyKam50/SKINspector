from transformers import AutoModelForImageTextToText
import torch
 

def inspect_model(path):
    model = AutoModelForImageTextToText.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        device_map="meta",  
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2"   
    )
    print(model)


if __name__ == "__main__":

    inspect_model("./models/Qwen2.5-VL-3B-Instruct")
    inspect_model("./models/Qwen3-VL-2B-Instruct")

'''
Qwen2_5_VLForConditionalGeneration(
  (model): Qwen2_5_VLModel(
    (visual): Qwen2_5_VisionTransformerPretrainedModel(
      (patch_embed): Qwen2_5_VisionPatchEmbed(
        (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
      )
      (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-31): 32 x Qwen2_5_VLVisionBlock(
          (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
          (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
          (attn): Qwen2_5_VLVisionAttention(
            (qkv): Linear(in_features=1280, out_features=3840, bias=True)
            (proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (mlp): Qwen2_5_VLMLP(
            (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
            (act_fn): SiLUActivation()
          )
        )
      )
      (merger): Qwen2_5_VLPatchMerger(
        (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
        (mlp): Sequential(
          (0): Linear(in_features=5120, out_features=5120, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=5120, out_features=2048, bias=True)
        )
      )
    )
    (language_model): Qwen2_5_VLTextModel(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0-35): 36 x Qwen2_5_VLDecoderLayer(
          (self_attn): Qwen2_5_VLAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (k_proj): Linear(in_features=2048, out_features=256, bias=True)
            (v_proj): Linear(in_features=2048, out_features=256, bias=True)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary_emb): Qwen2_5_VLRotaryEmbedding()
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen2_5_VLRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
'''


'''
Qwen3VLForConditionalGeneration(
  (model): Qwen3VLModel(
    (visual): Qwen3VLVisionModel(
      (patch_embed): Qwen3VLVisionPatchEmbed(
        (proj): Conv3d(3, 1024, kernel_size=(2, 16, 16), stride=(2, 16, 16))
      )
      (pos_embed): Embedding(2304, 1024)
      (rotary_pos_emb): Qwen3VLVisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-23): 24 x Qwen3VLVisionBlock(
          (norm1): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (norm2): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
          (attn): Qwen3VLVisionAttention(
            (qkv): Linear(in_features=1024, out_features=3072, bias=True)
            (proj): Linear(in_features=1024, out_features=1024, bias=True)
          )
          (mlp): Qwen3VLVisionMLP(
            (linear_fc1): Linear(in_features=1024, out_features=4096, bias=True)
            (linear_fc2): Linear(in_features=4096, out_features=1024, bias=True)
            (act_fn): GELUTanh()
          )
        )
      )
      (merger): Qwen3VLVisionPatchMerger(
        (norm): LayerNorm((1024,), eps=1e-06, elementwise_affine=True)
        (linear_fc1): Linear(in_features=4096, out_features=4096, bias=True)
        (act_fn): GELU(approximate='none')
        (linear_fc2): Linear(in_features=4096, out_features=2048, bias=True)
      )
      (deepstack_merger_list): ModuleList(
        (0-2): 3 x Qwen3VLVisionPatchMerger(
          (norm): LayerNorm((4096,), eps=1e-06, elementwise_affine=True)
          (linear_fc1): Linear(in_features=4096, out_features=4096, bias=True)
          (act_fn): GELU(approximate='none')
          (linear_fc2): Linear(in_features=4096, out_features=2048, bias=True)
        )
      )
    )
    (language_model): Qwen3VLTextModel(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0-27): 28 x Qwen3VLTextDecoderLayer(
          (self_attn): Qwen3VLTextAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (k_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (v_proj): Linear(in_features=2048, out_features=1024, bias=False)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (q_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
            (k_norm): Qwen3VLTextRMSNorm((128,), eps=1e-06)
          )
          (mlp): Qwen3VLTextMLP(
            (gate_proj): Linear(in_features=2048, out_features=6144, bias=False)
            (up_proj): Linear(in_features=2048, out_features=6144, bias=False)
            (down_proj): Linear(in_features=6144, out_features=2048, bias=False)
            (act_fn): SiLUActivation()
          )
          (input_layernorm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen3VLTextRMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen3VLTextRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
'''