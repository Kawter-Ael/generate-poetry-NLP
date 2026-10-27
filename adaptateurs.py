"""Validation et activation des adaptateurs locaux, sans charger Qwen à l'import."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_LABELS = {"v1": "LoRA V1 — Ancien modèle", "v2": "LoRA V2 — Deuxième modèle",
                  "v3": "LoRA V3 — Nouveau dataset 416"}
V3_SHA256 = "6f113d8e720cff777a0ffbcd6e78c89b87e3bbcd330634ff337d4c367ea97c18"


def comparison_versions(selected):
    """Conserver la comparaison V1/V2 ; inclure V3 lorsqu'il est sélectionné."""
    if selected not in ADAPTER_LABELS:
        raise ValueError("Version LoRA inconnue.")
    return ("v1", "v2", "v3") if selected == "v3" else ("v1", "v2")


@lru_cache(maxsize=8)
def _digest(path, size, modified_ns):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(path):
    stat = path.stat()
    return _digest(str(path), stat.st_size, stat.st_mtime_ns)


def adapter_info(version="v1", root=None):
    if version not in ADAPTER_LABELS:
        raise ValueError("Version LoRA inconnue : choisir v1, v2 ou v3.")
    root = Path(root) if root is not None else PROJECT_ROOT
    directory = (root if version == "v1" else root / ("adapter_V2" if version == "v2" else "adapter_VERSION3")).resolve()
    # V3 peut être livré dans un sous-dossier : refuser une sélection ambiguë.
    if version == "v3":
        candidates = sorted(p.parent for p in directory.rglob("adapter_config.json")
                            if (p.parent / "adapter_model.safetensors").is_file())
        if len(candidates) != 1:
            raise FileNotFoundError(f"Adaptateur V3 incomplet ou ambigu : {directory} ({len(candidates)} jeux de poids trouvés).")
        directory = candidates[0]
    config_path, weights = directory/"adapter_config.json", directory/"adapter_model.safetensors"
    for path in (config_path, weights):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Adaptateur {version.upper()} incomplet : {path}")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    if config.get("base_model_name_or_path") != BASE_MODEL:
        raise ValueError(f"{version.upper()} incompatible : modèle de base attendu {BASE_MODEL}.")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError(f"{version.upper()} doit être un adaptateur LoRA de génération textuelle.")
    if not isinstance(config.get("r"), int) or config["r"] < 1:
        raise ValueError(f"Rang LoRA invalide pour {version.upper()}.")
    sha256 = fingerprint(weights)
    if version == "v3":
        if sha256 != V3_SHA256:
            raise ValueError(f"SHA-256 V3 incorrect : attendu {V3_SHA256}, obtenu {sha256}.")
        if (config.get("r"), config.get("lora_alpha"), config.get("lora_dropout")) != (16, 32, 0.05):
            raise ValueError("Configuration V3 inattendue : r=16, alpha=32, dropout=0.05 requis.")
    return {"version": version, "label": ADAPTER_LABELS[version], "base_model": BASE_MODEL,
            "adapter_path": str(directory), "adapter_sha256": sha256,
            "config_sha256": fingerprint(config_path)}


def activate_adapter(model, version="v1"):
    """Appeler sous le verrou partagé jusqu'à la fin des trois tentatives."""
    info = adapter_info(version)
    registered = getattr(model, "_poetry_adapter_fingerprints", {})
    identity = (info["adapter_sha256"], info["config_sha256"])
    if version in registered and registered[version] != identity:
        raise RuntimeError("L'adaptateur a changé sur disque. Redémarrer Streamlit pour recharger ses poids.")
    if version not in model.peft_config:
        model.load_adapter(info["adapter_path"], adapter_name=version, is_trainable=False)
    model.set_adapter(version, inference_mode=True)
    model.eval()
    registered[version] = identity
    model._poetry_adapter_fingerprints = registered
    return info


def model_label(poem):
    """Les anciens poèmes sans provenance ne reçoivent pas une version inventée."""
    return (poem.get("model") or {}).get("label") or "Version du modèle non renseignée"
