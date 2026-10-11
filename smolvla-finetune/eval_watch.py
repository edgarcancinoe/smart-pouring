"""Watch one SmolVLA model water the plants, live in the MuJoCo viewer (window closed = stop). Saves nothing.

The trials are those of ./run eval (same scenes and seeds), so what you see is what the results count.

Usage: ./run eval-watch
"""
import mujoco
import mujoco.viewer

import evaluate as ev

MODEL = "libero-4"                         # model to watch (<init>-<k>: README.md)
UNITS = ["left-1", "middle-6", "right-8"]  # <plant>-<spot>, shown in this order
TRIALS = 1                                 # trials per unit (at most evaluate.TRIALS)


def main():
    model = ev.load_model(MODEL)
    for unit in UNITS:
        plant, spot = unit.split("-")[0], int(unit.split("-")[1])
        e = ev.scene(plant).env
        m, d = e.sim.model._model, e.sim.data._data
        with mujoco.viewer.launch_passive(m, d) as viewer:  # one window per unit: each plant is its own scene
            with viewer.lock():
                viewer.cam.fixedcamid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            for t in range(TRIALS):
                r = ev.trial(ev.scene(plant), model, plant, spot, ev.trial_seed(plant, spot, t), viewer=viewer)
                if not viewer.is_running():
                    return
                print(f"{MODEL} {unit} #{t}: {ev.describe(r, plant)}", flush=True)


if __name__ == "__main__":
    main()
