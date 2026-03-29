import cv2

cap = cv2.VideoCapture(7)   # 0 改成你的相机编号

if not cap.isOpened():
    print("相机打开失败")
    raise SystemExit(1)

prop_names = {
    "FRAME_WIDTH": "CAP_PROP_FRAME_WIDTH",
    "FRAME_HEIGHT": "CAP_PROP_FRAME_HEIGHT",
    "FPS": "CAP_PROP_FPS",
    "BRIGHTNESS": "CAP_PROP_BRIGHTNESS",
    "CONTRAST": "CAP_PROP_CONTRAST",
    "SATURATION": "CAP_PROP_SATURATION",
    "HUE": "CAP_PROP_HUE",
    "GAIN": "CAP_PROP_GAIN",
    "EXPOSURE": "CAP_PROP_EXPOSURE",
    "AUTO_EXPOSURE": "CAP_PROP_AUTO_EXPOSURE",
    "SHARPNESS": "CAP_PROP_SHARPNESS",
    "GAMMA": "CAP_PROP_GAMMA",
    "TEMPERATURE": "CAP_PROP_TEMPERATURE",
    "FOCUS": "CAP_PROP_FOCUS",
    "AUTOFOCUS": "CAP_PROP_AUTOFOCUS",
    "ZOOM": "CAP_PROP_ZOOM",
    "WB_BLUE_U": "CAP_PROP_WB_BLUE_U",
    "WB_RED_V": "CAP_PROP_WB_RED_V",
    "BACKLIGHT": "CAP_PROP_BACKLIGHT",
    "BUFFERSIZE": "CAP_PROP_BUFFERSIZE",
    "FOURCC": "CAP_PROP_FOURCC",
    "FORMAT": "CAP_PROP_FORMAT",
    "MODE": "CAP_PROP_MODE",
}

for name, attr_name in prop_names.items():
    pid = getattr(cv2, attr_name, None)
    if pid is None:
        continue
    value = cap.get(pid)
    print(f"{name}: {value}")

cap.release()
