"""Generate an English poem using the local LoRA adapter on CPU."""

import argparse
import re
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
from adaptateurs import BASE_MODEL, adapter_info
from poesie import (build_messages, complete_poem, DEFAULT_REPETITION_PENALTY,
                    DEFAULT_NO_REPEAT_NGRAM_SIZE, DEFAULT_TEMPERATURE)


class PoemLineStoppingCriteria(StoppingCriteria):
    """Arrête à une frontière de vers ; ne retire jamais les tokens déjà émis."""

    def __init__(self, tokenizer, prompt_length, line_count, first_line=""):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.line_count = line_count
        self.first_line = first_line
        self.reason = None

    def __call__(self, input_ids, scores, **kwargs):
        continuation = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=True)
        content = complete_poem(continuation, self.first_line)
        count = sum(bool(line.strip()) for line in content.splitlines())
        if count > self.line_count:
            # Le dépassement reste dans la sortie et rendra le candidat non conforme.
            self.reason = "line_overflow"
        elif count == self.line_count and re.search(r"[\r\n][^\S\r\n]*$", continuation):
            self.reason = "line_boundary"
        return torch.full((input_ids.shape[0],), self.reason is not None,
                          dtype=torch.bool, device=input_ids.device)


def load_model(version="v1"):
    info = adapter_info(version)
    # Un tokenizer commun : les identifiants de tokens restent ceux de l'application V1.
    tokenizer_path = Path(__file__).resolve().parent
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        # Le décodage token par token souffre de la surallocation des threads.
        # Garder des ressources pour l'interface sur les postes Windows.
        torch.set_num_threads(min(4, torch.get_num_threads()))
    dtype = torch.float16 if device == "cuda" else torch.float32
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    model = PeftModel.from_pretrained(base, info["adapter_path"], adapter_name=version, is_trainable=False)
    model.to(device)
    model.eval()
    model._poetry_adapter_fingerprints = {
        version: (info["adapter_sha256"], info["config_sha256"])}
    return tokenizer, model


def generate_poem(tokenizer, model, theme="", first_line="", temperature=DEFAULT_TEMPERATURE,
                  top_p=0.9, max_new_tokens=200, line_count=8, required_words=None,
                  correction="", return_metadata=False, tone="Libre",
                  repetition_penalty=DEFAULT_REPETITION_PENALTY,
                  no_repeat_ngram_size=DEFAULT_NO_REPEAT_NGRAM_SIZE, seed=None):
    if not 0 < temperature or not 0 < top_p <= 1 or repetition_penalty < 1:
        raise ValueError("Température > 0, top_p dans ]0, 1] et pénalité >= 1 requis.")
    if not isinstance(no_repeat_ngram_size, int) or no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size doit être un entier positif ou nul.")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens doit être un entier strictement positif.")
    if seed is not None and (not isinstance(seed, int) or not 0 <= seed < 2 ** 63):
        raise ValueError("seed doit être un entier positif ou nul inférieur à 2**63.")
    prepared_at = time.perf_counter()
    messages = build_messages(theme, first_line, line_count, required_words, correction, tone)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Préremplir la réponse garantit une première ligne exacte, sans la réécrire.
    if first_line.strip():
        prompt += first_line + "\n"
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
    line_stop = PoemLineStoppingCriteria(tokenizer, inputs["input_ids"].shape[1], line_count, first_line)
    # Restaure les RNG à la sortie : comparer des seeds ne modifie pas les autres sessions.
    rng_context = torch.random.fork_rng() if seed is not None else nullcontext()
    generate_at = time.perf_counter()
    with torch.inference_mode(), rng_context:
        if seed is not None:
            torch.manual_seed(seed)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            stopping_criteria=StoppingCriteriaList([line_stop]),
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded_at = time.perf_counter()
    generated = output[0, inputs["input_ids"].shape[1]:]
    raw_continuation = tokenizer.decode(generated, skip_special_tokens=True)
    content = complete_poem(raw_continuation, first_line)
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    ended_eos = bool(len(generated)) and int(generated[-1]) in eos_ids
    capped = len(generated) >= max_new_tokens and not ended_eos and line_stop.reason is None
    if return_metadata:
        return {"content": content, "raw_continuation": raw_continuation,
                "token_limit_reached": capped,
                "prompt_tokens": inputs["input_ids"].shape[1], "generated_tokens": len(generated),
                "preparation_seconds": generate_at - prepared_at,
                "model_generate_seconds": decoded_at - generate_at,
                "decoding_seconds": time.perf_counter() - decoded_at,
                "stop_reason": line_stop.reason or ("token_limit" if capped else "eos" if ended_eos else "unknown")}
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--theme", default="the sea at sunset")
    parser.add_argument("--adapter", choices=["v1", "v2", "v3"], default="v1")
    args = parser.parse_args()
    print("Chargement du modele sur CPU...", flush=True)
    tokenizer, model = load_model(args.adapter)
    print("Generation du poeme...", flush=True)
    print("\n" + generate_poem(tokenizer, model, args.theme))


if __name__ == "__main__":
    main()
