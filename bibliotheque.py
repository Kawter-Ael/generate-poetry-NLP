"""Stockage local : lecture, import et ajout sans remplacer les anciens poèmes."""
import csv
import hashlib
import io
import json
import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

LIBRARY_PATH = Path(__file__).resolve().parent / "data" / "poemes_enregistres.json"


def load_poems(path=LIBRARY_PATH):
    """Un fichier invalide est signalé, jamais remplacé par une liste vide."""
    path = Path(path)
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or any(
        not isinstance(p, dict) or not isinstance(p.get("content"), str)
        or not isinstance(p.get("id"), str) for p in records
    ):
        raise ValueError("La bibliothèque JSON ne contient pas une liste de poèmes valide.")
    return records


@contextmanager
def library_lock(path):
    """Évite que deux sessions écrivent simultanément dans le même fichier."""
    lock = path.with_suffix(".lock")
    deadline = time.monotonic() + 5
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError("La bibliothèque est occupée. Réessayez dans quelques secondes.")
            time.sleep(0.05)
    try:
        yield
    finally:
        os.close(descriptor)
        lock.unlink()


def save_poems(new_records, path=LIBRARY_PATH):
    """Ajout par identifiant et écriture atomique pour protéger les anciens textes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with library_lock(path):
        records = load_poems(path)
        known = {p["id"] for p in records}
        added = 0
        for poem in new_records:
            if poem["id"] not in known:
                records.append(poem)
                known.add(poem["id"])
                added += 1
        if not added:
            return 0
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                             suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(records, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return added


def import_records(filename, raw):
    """CSV : colonnes françaises/anglaises ; Markdown : un fichier = un poème."""
    text = raw.decode("utf-8-sig")
    if filename.lower().endswith(".csv"):
        try:
            dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    else:
        lines = text.splitlines()
        heading = lines[0].startswith("# ") if lines else False
        rows = [{"title": lines[0][2:].strip() if heading else Path(filename).stem,
                 "content": "\n".join(lines[1:]).strip() if heading else text.strip()}]
    result = []
    for row in rows:
        row = {str(k).strip().lower(): v for k, v in row.items() if k is not None}
        def field(*names):
            return next((row[n].strip() for n in names if isinstance(row.get(n), str)
                         and row[n].strip()), "")
        content = field("content", "contenu", "poem", "poème", "poeme", "text", "texte", "generated_poem")
        if not content:
            continue
        title = field("title", "titre") or content.splitlines()[0][:100]
        theme = field("theme", "thème")
        # Empêche les doublons lors d'un nouvel import du même texte.
        digest = hashlib.sha256(json.dumps([title, theme, content], ensure_ascii=False).encode()).hexdigest()
        result.append({"id": "import-" + digest, "title": title, "theme": theme,
                       "first_line": field("first_line", "first", "première ligne", "premiere_ligne"),
                       "content": content, "created_at": field("created_at", "date", "date_creation") or None,
                       "parameters": {}, "source": filename})
    if not result:
        raise ValueError("Aucun poème reconnu. Utilisez une colonne content, poem, texte ou contenu.")
    return result
