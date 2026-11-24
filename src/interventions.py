"""Intervention families used for estimation and reporting."""

ADAPTIVITY = "Adaptivity and learner choice"
FAMILIES = (
    "Generic control/treatment",
    ADAPTIVITY,
    "Scaffolding, hints, and examples",
    "Practice-to-mastery threshold",
    "Motivation and affect",
    "Feedback, media, and response format",
)


def intervention_family(contrast: str) -> str:
    text = str(contrast).lower()
    if text.startswith(("control vs.", "control vs ")):
        return FAMILIES[0]
    if "motivational" in text or "emotion" in text:
        return FAMILIES[4]
    if "ptm" in text:
        return FAMILIES[3]
    if any(
        term in text
        for term in ("adaptive", "choice", "problem difficulty", "problems required for mastery")
    ):
        return ADAPTIVITY
    if any(term in text for term in ("scaffolding", "hint", "example")):
        return FAMILIES[2]
    return FAMILIES[5]
