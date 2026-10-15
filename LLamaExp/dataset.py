"""
Neutrino LArTPC Dataset for LLaMA-3.2 Vision Models
===================================================
Handles data loading, 16-bit TIFF energy deposition normalization,
label formatting, explanation scrubbing, and chat-template prompt construction.
"""

import os
import random
import re
import numpy as np
from PIL import Image
import tifffile as tiff
from torch.utils.data import Dataset

SYSTEM_MESSAGE = """
You are a neutrino physicist analyzing simulated LArTPC events.

Each event has two 2D projections of detector energy deposition:
- zx projection
- zy projection
The z-axis is the beam direction.

Classify each event as exactly one of:
1. ν_e CC
2. ν_μ CC
3. Neutral Current (NC)

Use ONLY spatial, morphological, and topological evidence visible in the projections.

ν_e CC:
Look for a primary electron-induced electromagnetic shower originating at or very near the interaction vertex. It may be diffuse, fuzzy, branching, or spatially extended. A small shower can occur at low energy.
Do not assume every electromagnetic shower is a primary electron. Photons can also produce showers; when visible, a conversion gap favors a photon interpretation. A π⁰ can produce two photon showers that may be separate or overlap and resemble a single electron-like shower.

ν_μ CC:
Look for a primary muon candidate with a continuous, relatively straight, MIP-like track. Muons can be short or dim, especially at low energy. Do not require a long or bright track.
Distinguish the primary muon from hadronic activity. Brighter, thicker, or more complicated structures are not automatically the primary lepton.
A muon may decay and produce a Michel electron near the end of its track. Do not mistake such secondary electromagnetic activity for the primary electron of a ν_e CC event.

Neutral Current (NC):
NC events can contain charged hadrons, photons, π⁰s, tracks, and showers. Do NOT classify an event as NC merely because it is sparse, small, short, or dim.
Favor NC when there is no convincing primary electron or muon signature and the visible topology is compatible with hadronic or photon-induced activity.

Important rules:
- Low-energy events have shorter tracks and smaller showers and are harder to classify.
- Consider BOTH projections. A feature may be clearer in one view than the other.
- Image intensity is normalized separately for each event. Do not compare absolute brightness between events or use it as a direct proxy for neutrino energy.
- Do not infer exact neutrino energy, invisible particles, interaction mechanisms, or particle presence/absence that cannot be established from the images.
- Do not use a single weak visual cue to determine the class.
- Use uncertainty-aware language such as "consistent with", "suggests", or "appears to be" when appropriate.

Describe the observable topology FIRST, then connect it to the classification.
You must still select exactly one: ν_e CC, ν_μ CC, or Neutral Current (NC).
"""

DEFAULT_CLASSES = {
    "nuecc": "ν_e CC",
    "numucc": "ν_μ CC",
    "nc": "Neutral Current"
}


def process_tiff_array(filepath_or_arr):
    """
    Applies logarithmic energy deposition scaling and min-max normalization
    to convert raw LArTPC TIFF detector readouts to 3-channel RGB PIL Images.
    
    1. Value clamping: Clamping values < 0.25 to 0.0 and > 30.0 to 30.0 MeV.
    2. Logarithmic scaling: log1p(x) to compress high dynamic range depositions.
    3. Min-Max normalization to [0.0, 1.0].
    4. Conversion to uint8 RGB Image required by VLM vision encoders.
    """
    if isinstance(filepath_or_arr, str):
        img_array = tiff.imread(filepath_or_arr).astype(np.float32)
    else:
        img_array = np.array(filepath_or_arr, dtype=np.float32)

    img_array[img_array < 0.25] = 0.0
    img_array[img_array > 30.0] = 30.0

    log_arr = np.log1p(img_array)
    log_min = np.log1p(0.0)    # 0.0
    log_max = np.log1p(30.0)   # ~3.434

    norm_arr = (log_arr - log_min) / (log_max - log_min)
    norm_arr = np.clip(norm_arr, 0.0, 1.0)

    uint8_arr = (norm_arr * 255.0).astype(np.uint8)
    return Image.fromarray(uint8_arr, mode='L').convert("RGB")


def scrub_explanation_labels(explanation: str) -> str:
    """
    Scrubs explicit classification labels from training explanations
    to prevent teacher-forcing data leakage during next-token prediction.
    """
    if not explanation:
        return ""

    nue_pattern = re.compile(
        r"(?i)\b(?:electron\s+neutrino(?:\s+(?:charged\s+current|cc))?|(?:nu|[νv])[-_\s]?e\s*cc|nuecc|nu_e\s*cc|(?:nu|[νv])[-_\s]?e|nue|nu_e)\b"
    )
    numu_pattern = re.compile(
        r"(?i)\b(?:muon\s+neutrino(?:\s+(?:charged\s+current|cc))?|(?:nu|[νv])[-_\s]?(?:mu|[μu])\s*cc|numucc|nu_mu\s*cc|(?:nu|[νv])[-_\s]?(?:mu|[μu])|numu|nu_mu)\b"
    )
    nc_pattern = re.compile(
        r"(?i)\b(?:neutral\s+current|nc(?:\s+(?:event|interaction))?)\b"
    )

    scrubbed = nue_pattern.sub('ν_e CC', explanation)
    scrubbed = numu_pattern.sub('ν_μ CC', scrubbed)
    scrubbed = nc_pattern.sub('Neutral Current', scrubbed)
    return scrubbed


def format_data(sample):
    """
    Formats the sample dictionary into chat template messages compatible with LLaMA 3.2 Vision.
    """
    if "explanation" in sample and sample["explanation"]:
        user_text = (
            "Classify the neutrino event as ν_e CC, ν_μ CC, or Neutral Current "
            "and explain your reasoning.\n"
            "Give your answer in the following format:\n"
            "<explanation>\n"
            "The reasoning is {{REASONING}}\n"
            "</explanation>\n\n"
            "<label>\n"
            "{{LABEL}}\n"
            "</label>\n"
        )
        assistant_text = (
            "<explanation>\n"
            f"{sample['explanation']}\n"
            "</explanation>\n\n"
            "<label>\n"
            f"{sample['class']}\n"
            "</label>"
        )
    else:
        user_text = (
            "Classify the neutrino event as ν_e CC, ν_μ CC, or Neutral Current.\n"
            "Give your answer in the following format:\n"
            "The event is <label> {{LABEL}} </label>.\n"
        )
        assistant_text = (
            f"The event is <label> {sample['class']} </label>.\n"
        )

    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_MESSAGE.strip()}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["zx"]},
                    {"type": "image", "image": sample["zy"]},
                    {
                        "type": "text",
                        "text": user_text,
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": assistant_text,
                    }
                ],
            },
        ],
        "mask_loss": sample.get("mask_loss", False),
        "has_real_desc": sample.get("has_real_desc", False),
    }


class NeutrinoDataset(Dataset):
    """
    PyTorch Dataset for LArTPC neutrino event classification with LLaMA 3.2 Vision.
    """
    def __init__(self, paths, classes=None, explainations=False, require_desc=False, oversample_desc=1):
        self.classes = classes or DEFAULT_CLASSES
        self.explainations = explainations
        self.require_desc = require_desc
        self.oversample_desc = oversample_desc

        if require_desc:
            self.paths = [p for p in paths if os.path.exists(os.path.join(p, "desc.txt"))]
        else:
            if oversample_desc > 1 and explainations:
                desc_paths = [p for p in paths if os.path.exists(os.path.join(p, "desc.txt"))]
                self.paths = list(paths) + desc_paths * (oversample_desc - 1)
            else:
                self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]

        try:
            path_parts = path.split(os.sep)
            if len(path_parts) >= 3 and path_parts[-2].lower() == "cc":
                key = path_parts[-3].lower() + "cc"
                gt = self.classes.get(key, "ν_μ CC")
            else:
                gt = self.classes.get("nc", "Neutral Current")

            zx = process_tiff_array(os.path.join(path, "xz.tiff"))
            zy = process_tiff_array(os.path.join(path, "yz.tiff"))

            explanation = None
            has_real_desc = False

            if self.explainations:
                desc_path = os.path.join(path, "desc.txt")
                if os.path.exists(desc_path):
                    with open(desc_path, "r", encoding="utf-8") as f:
                        raw_desc = f.read().strip()
                    explanation = scrub_explanation_labels(raw_desc)
                    has_real_desc = True

            return format_data({
                "zx": zx,
                "zy": zy,
                "class": gt,
                "explanation": explanation,
                "mask_loss": False,
                "has_real_desc": has_real_desc
            })

        except Exception as e:
            random_idx = random.randint(0, len(self.paths) - 1)
            return self.__getitem__(random_idx)
