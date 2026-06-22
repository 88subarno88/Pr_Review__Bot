"""Every knob in one place. Be ready to justify each value in an interview."""

# model
MODEL_ID = "unsloth/Qwen2.5-Coder-3B-Instruct"   # code-pretrained, great for diffs
MAX_SEQ_LEN = 2048

# LoRA
LORA_R = 16
LORA_ALPHA = 32                # ~= 2 * r
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj"]

# training 
LEARNING_RATE = 2e-4
EPOCHS = 2
BATCH_SIZE = 2
GRAD_ACCUM = 4                 # effective batch = 8
WARMUP_RATIO = 0.03
SEED = 42


SYSTEM_PROMPT = (
    "You are a senior code reviewer. Given a single code diff hunk, write ONE "
    "concise, actionable review comment about the most important issue. "
    "No greetings, no preamble, no markdown headers."
)

# paths/serving
ADAPTER_DIR = "adapters/r16"
RESULTS_DIR = "results"
GEMINI_MODEL = "gemini-2.5-flash"   
OLLAMA_MODEL = "pr-reviewer"

# --- eval / serving ---
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_COST_PER_1K = None   
OLLAMA_MODEL = "pr-reviewer"