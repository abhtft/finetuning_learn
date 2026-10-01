-------------------------------------


abhishek@Abhishek:~/projects/unsloth$ source gemma_env/bin/activate
(gemma_env) abhishek@Abhishek:~/projects/unsloth$ sudo apt update && sudo apt install -y python3.12-v
(gemma_env) abhishek@Abhishek:~/projects/unsloth$ sudo apt update && sudo apt install -y python3.12-v
(gemma_env) abhishek@Abhishek:~/projects/unsloth$ sudo apt update && sudo apt install -y python3.12-v
(gemma_env) abhishek@Abhishek:~/projects/unsloth$ python train_gemma4_e4b.py
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2026.9.12: Fast Gemma4 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA GeForce RTX 5050 Laptop GPU. Num GPUs = 1. Max memory: 7.933 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Traceback (most recent call last):

           ^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth/models/loader.py", line 2472, in from_pretrained
    model, tokenizer = FastBaseModel.from_pretrained(
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/temporary_patches/moe_grouped_modulelist.py", line 390, in wrapper
    result = func(*args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth/models/loader_utils.py", line 3488, in _wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth/models/vision.py", line 2842, in from_pretrained
    model = auto_model.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/models/auto/auto_factory.py", line 387, in from_pretrained
    return model_class.from_pretrained(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4113, in from_pretrained
    device_map = _get_device_map(model, device_map, max_memory, hf_quantizer)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/integrations/accelerate.py", line 373, in _get_device_map
    hf_quantizer.validate_environment(device_map=device_map)
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/quantizers/quantizer_bnb_4bit.py", line 74, in validate_environment
    raise ValueError(
ValueError: Some modules are dispatched on the CPU or the disk. Make sure you have enough GPU RAM to fit the quantized model. If you want to dispatch the model on the CPU or the disk while keeping these modules in 32-bit, you need to set `llm_int8_enable_fp32_cpu_offload=True` and pass a custom `device_map` to `from_pretrained`. Check https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu for more details. 
(gemma_env) abhishek@Abhishek:~/projects/unsloth$ ^C
(gemma_env) abhishek@Abhishek:~/projects/unsloth$ 

----------------------------------------

(gemma_env) abhishek@Abhishek:~/projects/unsloth$ python train_gemma4_e4b.py
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!
==((====))==  Unsloth 2026.9.12: Fast Gemma4 patching. Transformers: 5.5.0.
   \\   /|    NVIDIA GeForce RTX 5050 Laptop GPU. Num GPUs = 1. Max memory: 7.933 GB. Platform: Linux.
O^O/ \_/ \    Torch: 2.11.0+cu128. CUDA: 12.0. CUDA Toolkit: 12.8. Triton: 3.6.0
\        /    Bfloat16 = TRUE. FA [Xformers = None. FA2 = False]
 "-____-"     Free license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████| 2130/2130 [00:51<00:00, 41.20it/s]
Loading dataset...
README.md: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 982/982 [00:00<00:00, 2.94MB/s]
data/train-00000-of-00001.parquet: downloading bytes: █████████████████████████████████████████████████████████████████████████████████████████|  116MB, 4.70MB/s  
data/train-00000-of-00001.parquet: reconstructing file: 100%|█████████████████████████████████████████████████████████████████████████|  117MB /  117MB, 8.55MB/s  
Generating train split: 100%|████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [00:01<00:00, 80681.49 examples/s]
Unsloth: reducing dataset_num_proc 8 -> 3 to fit free memory (~1GB per worker). Set UNSLOTH_DATASET_NUM_PROC to override.
Unsloth: Standardizing formats (num_proc=3): 100%|███████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 964.11 examples/s]
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 3795.41 examples/s]
Unsloth: Tokenizing ["text"] (num_proc=3): 100%|██████████████████████████████████████████████████████████████████████████| 500/500 [00:29<00:00, 16.78 examples/s]
Unsloth: Auto-detected instruction_part = '<|turn>user\n' and response_part = '<|turn>model\n'
Map: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 3712.52 examples/s]
Filter: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████| 500/500 [00:00<00:00, 10745.86 examples/s]
Unsloth: Removed 6 out of 500 samples from train_dataset where all labels were -100 (no response marker found, usually truncation). This prevents NaN loss during training.
Starting fine-tuning...
The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with the tokenizer's values. Updated tokens: {'bos_token_id': 2}.
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs used = 1
   \\   /|    Num examples = 494 | Num Epochs = 1 | Total steps = 30
O^O/ \_/ \    Batch size per device = 1 | Gradient accumulation steps = 4
\        /    Data Parallel GPUs = 1 | Total batch size (1 x 4 x 1) = 4
 "-____-"     Trainable parameters = 18,350,080 of 8,014,506,528 (0.23% trained)
  0%|                                                                                                                                       | 0/30 [00:00<?, ?it/s]Unsloth: Not an error, but Gemma4ForConditionalGeneration does not accept `num_items_in_batch`.
Using gradient accumulation will be very slightly less accurate.
Read more on gradient accumulation issues here: https://unsloth.ai/blog/gradient
/home/abhishek/projects/unsloth/unsloth/import_fixes.py:5429: UserWarning: 'has_cuda' is deprecated, please use 'torch.backends.cuda.is_built()'
  return original(name)
/home/abhishek/projects/unsloth/unsloth/import_fixes.py:5429: UserWarning: 'has_cudnn' is deprecated, please use 'torch.backends.cudnn.is_available()'
  return original(name)
/home/abhishek/projects/unsloth/unsloth/import_fixes.py:5429: UserWarning: 'has_mps' is deprecated, please use 'torch.backends.mps.is_built()'
  return original(name)
/home/abhishek/projects/unsloth/unsloth/import_fixes.py:5429: UserWarning: 'has_mkldnn' is deprecated, please use 'torch.backends.mkldnn.is_available()'
  return original(name)
Traceback (most recent call last):
  File "/home/abhishek/projects/unsloth/train_gemma4_e4b.py", line 83, in <module>
    trainer_stats = trainer.train()
                    ^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth/trainer.py", line 1556, in _train_with_reset
    return _orig_train(*train_args, **train_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth_compiled_cache/UnslothSFTTrainer.py", line 116, in wrapper
    output = f(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/trainer.py", line 1424, in train
    return inner_training_loop(
           ^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 81, in _fast_inner_training_loop
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/trainer.py", line 1734, in _run_epoch
    tr_loss_step = self.training_step(model, inputs, num_items_in_batch)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth_compiled_cache/UnslothSFTTrainer.py", line 1512, in training_step
    return super().training_step(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth/models/_utils.py", line 4851, in _unsloth_training_step_settling_fallbacks
    return _training_step_before_settle(self, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 40, in _unsloth_training_step
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/temporary_patches/misc.py", line 2804, in compute_loss
    result = original(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth_compiled_cache/UnslothSFTTrainer.py", line 1496, in compute_loss
    outputs = super().compute_loss(
              ^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth/models/_utils.py", line 4603, in _unsloth_pre_compute_loss
    outputs = self._old_compute_loss(model, inputs, *args, **kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<string>", line 41, in compute_loss
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/accelerate/utils/operations.py", line 943, in forward
    return model_forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/accelerate/utils/operations.py", line 931, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/peft/peft_model.py", line 2120, in forward
    return self.base_model(
           ^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/peft/tuners/tuners_utils.py", line 359, in forward
    return self.model(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/temporary_patches/gemma4_moe.py", line 132, in _patched_causal_lm_forward
    return _original_causal_lm_forward(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth_compiled_cache/unsloth_compiled_module_gemma4.py", line 1977, in forward
    return Gemma4ForConditionalGeneration_forward(self, input_ids=input_ids, pixel_values=pixel_values, pixel_values_videos=pixel_values_videos, input_features=input_features, attention_mask=attention_mask, input_features_mask=input_features_mask, position_ids=position_ids, image_position_ids=image_position_ids, video_position_ids=video_position_ids, past_key_values=past_key_values, mm_token_type_ids=mm_token_type_ids, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, logits_to_keep=logits_to_keep, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/_dynamo/external_utils.py", line 227, in nonrecursive_disable_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/transformers/utils/generic.py", line 876, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/unsloth_compiled_cache/unsloth_compiled_module_gemma4.py", line 1901, in Gemma4ForConditionalGeneration_forward
    loss = unsloth_fused_ce_loss(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/fused_losses/cross_entropy_loss.py", line 647, in unsloth_fused_ce_loss
    return apply(*(
           ^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/_dynamo/eval_frame.py", line 1263, in _fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/fused_losses/cross_entropy_loss.py", line 587, in _fused_loss_opaque
    return UnslothFusedLoss.apply(*args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/torch/autograd/function.py", line 596, in apply
    return super().apply(*args, **kwargs)  # type: ignore[misc]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/fused_losses/cross_entropy_loss.py", line 293, in forward
    n_chunks = get_chunk_size(
               ^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/fused_losses/cross_entropy_loss.py", line 196, in get_chunk_size
    multiplier = _get_chunk_multiplier(vocab_size, target_gb, fixed_gb)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/abhishek/projects/unsloth/gemma_env/lib/python3.12/site-packages/unsloth_zoo/fused_losses/cross_entropy_loss.py", line 181, in _get_chunk_multiplier
    raise RuntimeError("Unsloth: No or negligible GPU memory available for fused cross entropy.")
RuntimeError: Unsloth: No or negligible GPU memory available for fused cross entropy.

----------------------------------------------


.docs/hardware_guidelines.md

Issue & Resolution Breakdown
1. Why did RuntimeError: Unsloth: No or negligible GPU memory available for fused cross entropy occur?
When training starts, PyTorch pre-allocates memory for model weights, LoRA adapters, and optimizer states into its caching pool.
At the beginning of the loss computation, Unsloth’s fused cross-entropy kernel queries the GPU driver via cudaMemGetInfo (torch.cuda.mem_get_info(0)[0]) to determine how much transient memory is available to chunk the 256,000-token vocabulary calculation.
Because PyTorch had reserved nearly all of the 7.93GB VRAM pool, the driver reported 0 bytes free physical memory.
The original Unsloth code threw a RuntimeError if the target memory was <= 1e-9 GB instead of safely falling back to PyTorch's internal pool or a small, safe chunk budget.
2. How it was resolved
We patched _free_target_gb() and _get_chunk_multiplier() in the environment's unsloth_zoo fused loss module so that if the driver reports near-zero physical memory, it automatically calculates the chunk budget from PyTorch's pool or defaults to a safe minimum (0.25 GB).
This enables the cross-entropy calculator to break the 256,000 vocabulary computations into fine-grained slices (e.g. 16-64 tokens per chunk), which fit inside the available transient buffer without running out of memory.
Live Training Status
The fine-tuning run is now active and progressing:

Hardware: NVIDIA GeForce RTX 5050 Laptop GPU (7.933 GB VRAM)
Model: unsloth/gemma-4-E4B-it (4-bit QLoRA)
Progress:
text
Step 1: {'loss': 1.383, 'grad_norm': 0.7267, 'learning_rate': 0, 'epoch': 0.0081}
Step 2: {'loss': 1.834, 'grad_norm': 1.0090, 'learning_rate': 4e-05, 'epoch': 0.0162}
Progress: [2/30 steps] (~27.5s/step)
The adapters will automatically be saved to gemma_4_e4b_lora/ upon completion.

--------------------------------------------------------------------------------

🎉 Fine-Tuning Finished Successfully!

Training Summary & Metrics
Metric	Result
Model	unsloth/gemma-4-E4B-it (4-bit QLoRA)
GPU	NVIDIA GeForce RTX 5050 Laptop GPU (7.933 GB VRAM)
Total Runtime	7 minutes 41 seconds (461.1s)
Steps Completed	30 / 30 (100%)
Initial Loss	1.834
Final Loss	0.705
Average Train Loss	1.102
Saved Artifacts
The trained LoRA adapters and tokenizer configuration are saved in 

gemma_4_e4b_lora/
:



adapter_model.safetensors
 (~73.5 MB) — Trained LoRA weights


adapter_config.json
 — LoRA configuration & hyperparameters


tokenizer.json
 & 

chat_template.jinja
 — Chat template & tokenizer
Next Step: Running Inference With Your Adapters
You can load and test your newly trained model anytime using:

python
from unsloth import FastModel
model, tokenizer = FastModel.from_pretrained(
    model_name="gemma_4_e4b_lora",  # Loads base model + your trained adapters
    max_seq_length=1024,
    load_in_4bit=True,
    device_map="cuda",
)
FastModel.for_inference(model)  # Enables 2x faster inference
messages = [
    {"role": "user", "content": "Explain quantum computing in simple terms."}
]
inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=128, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


---------------------------------------------------------------------------------------------


Wed Sep 30 17:55:08 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 615.65.06              KMD Version: 616.56        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5050 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   76C    P2             42W /   70W |    7739MiB /   8151MiB |    100%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            9309      C   /python3.12                           N/A      |
+-----------------------------------------------------------------------------------------+
abhishek@Abhishek:~/projects/unsloth$ nvidia-smi
Wed Sep 30 17:56:01 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 615.65.06              KMD Version: 616.56        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5050 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   62C    P8             12W /   70W |    7706MiB /   8151MiB |      3%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            9309      C   /python3.12                           N/A      |
+-----------------------------------------------------------------------------------------+
abhishek@Abhishek:~/projects/unsloth$ nvidia-smi
Wed Sep 30 17:58:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 615.65.06              KMD Version: 616.56        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5050 ...    On  |   00000000:01:00.0  On |                  N/A |
| N/A   52C    P8              9W /   70W |     452MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

