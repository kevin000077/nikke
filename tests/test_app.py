from marble_aim.app import ApplicationController, SceneSnapshot
from marble_aim.geometry import Obstacle, Rect
from marble_aim.vision import DetectionResult


def test_block_signature_ignores_global_sprite_bobbing():
    controller = ApplicationController.__new__(ApplicationController)
    board = Rect(621, 139, 1042, 702)
    first = SceneSnapshot(
        board,
        (
            Obstacle(Rect(622, 210, 689, 277), 0),
            Obstacle(Rect(692, 210, 759, 277), 0),
            Obstacle(Rect(762, 280, 829, 347), 0),
        ),
        (830, 702),
        11.5,
    )
    bobbed = SceneSnapshot(
        board,
        (
            Obstacle(Rect(622, 212, 689, 279), 0),
            Obstacle(Rect(692, 212, 759, 279), 0),
            Obstacle(Rect(762, 282, 829, 349), 0),
        ),
        (830, 702),
        11.5,
    )
    shifted_down_one_row = SceneSnapshot(
        board,
        (
            Obstacle(Rect(622, 280, 689, 347), 0),
            Obstacle(Rect(692, 280, 759, 347), 0),
            Obstacle(Rect(762, 350, 829, 417), 0),
        ),
        (830, 702),
        11.5,
    )

    assert controller._block_signature(first) == controller._block_signature(bobbed)
    assert (
        controller._block_signature(first)
        != controller._block_signature(shifted_down_one_row)
    )


def test_new_aim_keeps_checking_same_blocks_for_clean_frames():
    controller = ApplicationController.__new__(ApplicationController)
    board = Rect(100, 50, 520, 610)
    obstacles = (Obstacle(Rect(101, 120, 168, 187), 0),)
    scene = SceneSnapshot(board, obstacles, (300, 610), 14.5)
    detection = DetectionResult(
        board=board,
        obstacles=list(obstacles),
        launch_origin=scene.origin,
        aim_line=((300, 610), (500, 200)),
    )

    class DetectorStub:
        def detect(self, frame, *, debug_masks=False):
            return detection

    controller.detector = DetectorStub()
    controller.debug_view = False
    controller.latest_scene = scene
    controller.locked_block_signature = controller._block_signature(scene)
    controller.round_candidate_signature = None
    controller.round_candidate_count = 0
    controller.round_candidate_origins = []
    controller.transition_same_count = 0
    controller._make_scene = lambda current: scene

    assert controller._check_scene_transition(None) == "pending"
    assert controller._check_scene_transition(None) == "pending"
    assert controller._check_scene_transition(None) == "pending"
    assert controller._check_scene_transition(None) == "same"
