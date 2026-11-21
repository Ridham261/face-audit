"""Pack everything the 8-category classifier needs into ONE file: D:\\audit\\models\\appearance_model.pt

Contents: CLIP image encoder weights (+config), the project's ConvNeXt-Large weights,
and the trained classifier heads (headwear / specs / hair).
"""
import numpy as np
import torch
from transformers import CLIPVisionModelWithProjection

OUT = r"D:\audit\models\appearance_model.pt"
clip = CLIPVisionModelWithProjection.from_pretrained(r"D:\audit\models\clip").eval()
ck = torch.load(r"D:\audit\models\best_convnext_large.pth", map_location="cpu")
cnx = ck.get("model_state_dict", ck)
z = np.load(r"D:\audit\models\appearance_head.npz", allow_pickle=False)
heads = {h: {"classes": [str(c) for c in z[f"{h}__classes"]],
             "coef": torch.from_numpy(z[f"{h}__coef"].astype("float32")),
             "bias": torch.from_numpy(z[f"{h}__intercept"].astype("float32"))}
         for h in ("headwear", "specs", "hair")}
torch.save({
    "version": 2,
    "classes": ["black_hair", "specs_black_hair", "white_hair", "specs_white_hair",
                "doctor_cap", "doctor_cap_specs", "turban", "turban_specs"],
    "feature_set": str(z["feature_set"]),
    "clip_vision_config": clip.config.to_dict(),
    "clip_vision_state": clip.state_dict(),
    "convnext_state": cnx,
    "convnext_classes": ["cap", "female", "hijab", "male", "turban"],
    "heads": heads,
}, OUT)
import os
print("saved", OUT, f"{os.path.getsize(OUT)/1e6:.0f} MB")
