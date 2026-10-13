"""./run eval for models trained with relative.py: eval_grasp_position.py's trials (same grasp detection), with
the grasp slots made relative as in training. Results in smolvla-finetune/results/<model>/.

Usage: ./run relative-eval [libero-4s-dart-xl-sd-rel ...] [--jobs 8]
"""
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

import eval_grasp_position as egp
import evaluate as ev
import relative
import runs

SmolVLAPolicy.prepare_state = relative.prepare_state
egp.SCRIPTS[:] = ["smolvla-memory/relative-actions/relative.py"]  # only these: a -gp model needs absolute slots
ev.MODELS = [m for m, c in runs.MODELS.items() if c.get("script") in egp.SCRIPTS]

if __name__ == "__main__":
    ev.main()
