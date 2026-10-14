"""./run eval for models trained with memory.py: the same trials, each remembering its first agentview frame.
Results in smolvla-finetune/results/<model>/.

Usage: ./run memory-eval [libero-4s-dart-xl-sd-mem ...] [--jobs 8]
"""
import evaluate as ev
import memory  # noqa: F401  (its patches, at import)
import runs

SCRIPT = "smolvla-memory/VLM-cache/memory.py"
_load_model = ev.load_model


def load_model(name):
    """evaluate.load_model, after checking the model was trained with the memory."""
    if runs.MODELS[name.partition("@")[0]].get("script") != SCRIPT:
        raise SystemExit(f"{name} isn't trained with the memory ({SCRIPT}): use ./run eval")
    return _load_model(name)


ev.load_model = load_model
ev.MODELS = [m for m, c in runs.MODELS.items() if c.get("script") == SCRIPT]  # default: every memory model

if __name__ == "__main__":
    ev.main()
