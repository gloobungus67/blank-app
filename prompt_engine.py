"""Prompt composer for the personal image studio. No network calls here."""

from __future__ import annotations

from dataclasses import dataclass
import random

PRESETS = {
    "Candid photography": "An authentic, spontaneous real-camera photograph, unstaged and full of human detail.",
    "Editorial lookbook": "An editorial fashion photograph with believable clothing construction, distinctive styling, and an intentional composition.",
    "Character continuity": "A new candid photograph featuring the same adult character as the provided reference images, preserving recognizable facial identity and characteristic features.",
    "Casting portraits": "A polished but natural casting photograph, with a clear face and accurate features, approachable expression, minimal retouching, and believable skin texture.",
    "Cinematic still": "A cinematic film still, with motivated practical light, rich environmental storytelling and a deliberate camera composition.",
    "Four-frame contact sheet": "One image laid out as a clean 2 by 2 photographic contact sheet. Show four distinct moments or angles of the same adult subject, without duplicated poses.",
}

SURPRISE_OPTIONS = {
    "wardrobe": [
        "an oversized striped shirt and textured trousers",
        "a vintage velvet jacket and worn-in jeans",
        "a playful vintage summer dress with chunky shoes",
        "a layered thrift-store outfit with a colorful scarf",
        "a tailored linen set with quirky jewelry",
        "a soft knit cardigan and mismatched vintage accessories",
    ],
    "scene": [
        "a lived-in apartment kitchen at dusk",
        "a quiet tram stop after rain",
        "a cluttered independent bookshop",
        "a sunlit greenhouse with patterned shadows",
        "a lively flea market on an overcast day",
        "a cozy café corner photographed from a booth",
    ],
    "angle": [
        "eye-level three-quarter view",
        "candid low angle from a nearby chair",
        "direct-to-camera waist-up framing",
        "full-body photo from slightly above",
        "side profile with a glance back at the camera",
    ],
}


@dataclass(frozen=True)
class PromptSpec:
    idea: str
    preset: str
    identity: str = ""
    face: str = ""
    build: str = ""
    hair: str = ""
    outfit: str = ""
    scene: str = ""
    pose: str = ""
    angle: str = ""
    vibe: str = ""
    has_references: bool = False
    invent_details: bool = True


def surprise_fields(rng: random.Random | None = None) -> dict[str, str]:
    rng = rng or random.Random()
    return {key: rng.choice(items) for key, items in SURPRISE_OPTIONS.items()}


def build_prompt(spec: PromptSpec) -> str:
    lines = [
        "Generate original visual artwork as a photorealistic photograph of clearly adult people (age 21+ if any people appear).",
        PRESETS.get(spec.preset, PRESETS["Candid photography"]),
        "Primary creative request: " + (spec.idea.strip() or "Create an original, believable character photograph."),
    ]
    if spec.has_references:
        lines.append(
            "Reference photos are supplied. When the intention is character continuity, preserve the subject's recognizable face, facial proportions, skin details, and distinguishing characteristics; change only what is requested. Do not merely recreate the reference composition."
        )
    elif spec.identity.strip():
        lines.append("Invent a specific, individual adult identity, not a generic beauty-template face.")
    labels = (
        ("Character / identity", spec.identity),
        ("Face and distinguishing features", spec.face),
        ("Body shape / silhouette", spec.build),
        ("Hair", spec.hair),
        ("Wardrobe", spec.outfit),
        ("Scene / environment", spec.scene),
        ("Pose and activity", spec.pose),
        ("Camera angle / framing", spec.angle),
        ("Expression / mood", spec.vibe),
    )
    lines.extend(f"{label}: {value.strip()}" for label, value in labels if value.strip())
    if spec.invent_details:
        lines.append("Use creative freedom to invent incidental details that feel personal, specific, and unrehearsed; keep the result coherent.")
    lines.append(
        "Visual quality: authentic camera optics, natural hands and joints, realistic fabric drape, credible human anatomy, subtle skin texture, small asymmetries, physically coherent light and shadows. Avoid artificial face smoothing, repeated stock-photo poses, duplicate body parts, and overprocessed AI aesthetics."
    )
    return "\n\n".join(lines)
