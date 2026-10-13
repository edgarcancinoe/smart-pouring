"""grasp_position.py, but slots 9-10 hold where the gripper is relative to the grasp (x - gx, y - gy) instead of the
grasp itself: putting the mug back is then a task of reducing this to zero.

Usage: python relative.py <share> <lerobot_train args ...>   (./run train <model with this script>)
"""
import grasp_position as gp
import state_dropout


def relative(state, xy):
    """state -> (B, 32) with grasp slots 8-10 = [grasped, gx, gy]; xy -> (B, 2) gripper x, y, both normalized.
    Slots 9-10 become xy - (gx, gy); the mean cancels, so it's (x - gx) / std. Zeros before the grasp."""
    state[:, 9:11] = (xy - state[:, 9:11]) * state[:, 8:9]
    return state


def prepare_state(self, batch):
    """grasp_position's state, with the grasp slots made relative. Gripper's xy comes from the state before dropout
    """
    xy = state_dropout._prepare_state(self, batch)[:, :2]
    return relative(gp.prepare_state(self, batch), xy)


if __name__ == "__main__":
    gp.main(prepare=prepare_state)
