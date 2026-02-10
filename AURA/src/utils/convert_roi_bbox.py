def convert_roi_bbox(Wf, Hf, W, H, x1,y1,x2,y2):
    # ROIAlign 좌표 변환
    sx = Wf / float(W)
    sy = Hf / float(H)
    x1f, y1f, x2f, y2f = x1 * sx, y1 * sy, x2 * sx, y2 * sy
    return (x1f, y1f, x2f, y2f)