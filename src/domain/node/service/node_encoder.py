import torch
from torchvision import ops

from config.settings import DEFAULT_YOLO_WEIGHT, DEVICE
from infrastructure.model.model_registry import get_yolo


class ObjectNodeEncoder:
    """
    Encodes YOLO-World tracking outputs into node-level tensors by combining
    ROI-aligned visual features with the model's text embeddings.
    """

    def __init__(self, weight_path: str | None = None):
        self.model = get_yolo(weight_path or DEFAULT_YOLO_WEIGHT)
        self.model.eval()

        self._init_text_embedding()
        self.features = {"P3": None}
        self._register_p3_hook()

    def _init_text_embedding(self):
        txt = self.model.model.txt_feats
        if txt.dim() == 3 and txt.shape[0] == 1:
            txt = txt.squeeze(0)

        self.text_emb = txt.detach().to(DEVICE)
        self.names = self.model.model.names

    def _register_p3_hook(self):
        self.model.model.model[12].register_forward_hook(self._hook_p3)

    def _hook_p3(self, module, inp, out):
        self.features["P3"] = out

    def nodes_from_result(self, result):
        if result.boxes is None or len(result.boxes) == 0:
            return []

        P3 = self.features["P3"]
        if P3 is None:
            raise RuntimeError("P3 feature is None. Check that the hook is registered.")

        _, _, Hf, Wf = P3.shape
        frame_bgr = result.orig_img
        H, W = frame_bgr.shape[:2]

        nodes = []
        for b in result.boxes:
            node = self._encode_single_object(b, P3, H, W, Hf, Wf)
            nodes.append(node)

        return nodes

    def _encode_single_object(self, box, P3, H, W, Hf, Wf):
        track_id = int(box.id.item()) if box.id is not None else -1
        cls_id = int(box.cls.item())
        score = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = self.names[cls_id]

        sx = Wf / float(W)
        sy = Hf / float(H)
        x1f, y1f, x2f, y2f = x1 * sx, y1 * sy, x2 * sx, y2 * sy

        roi_box = torch.tensor(
            [[0, x1f, y1f, x2f, y2f]], device=DEVICE, dtype=torch.float32
        )

        roi_feat = ops.roi_align(
            input=P3,
            boxes=roi_box,
            output_size=(7, 7),
            spatial_scale=1.0,
            sampling_ratio=-1,
            aligned=True,
        )
        roi_feat = roi_feat.mean(dim=(2, 3)).squeeze(0).detach()

        text_feat = self.text_emb[cls_id]
        node_feat = torch.cat([roi_feat, text_feat], dim=0)

        return {
            "track_id": track_id,
            "cls_id": cls_id,
            "cls_name": cls_name,
            "score": score,
            "bbox": [x1, y1, x2, y2],
            "roi_feat": roi_feat,
            "text_feat": text_feat,
            "node_feat": node_feat,
        }
