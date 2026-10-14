"""./run memory-eval for models trained with isolated.py. Results in smolvla-finetune/results/<model>/.

Usage: ./run isolated-eval [libero-4s-dart-xl-sd-memiso ...] [--jobs 8]
"""
import eval_memory as em
import evaluate as ev
import isolated  # noqa: F401  (its patches, at import)
import runs

em.SCRIPT = "smolvla-memory/VLM-cache-isolated/isolated.py"
ev.MODELS = [m for m, c in runs.MODELS.items() if c.get("script") == em.SCRIPT]  # default: every isolated model

if __name__ == "__main__":
    ev.main()
