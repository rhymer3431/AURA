import cv2
from pathlib import Path

from application.scene_understanding import SceneUnderstandingUseCase
from domain.pipeline.entity.scene_state import SceneState


def run_offline_video(
    video_path: str,
    use_case: SceneUnderstandingUseCase,
    visualizer=None,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state = SceneState(frame_id=frame_id, timestamp=0.0, raw_frame=frame)
        frame_id += 1
        state = use_case.execute(state)

        if visualizer:
            vis_frame = visualizer.draw(state.raw_frame, state.detections, state.scene_graph)
            cv2.imshow("offline", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit("Wire up your pipeline and call run_offline_video().")
