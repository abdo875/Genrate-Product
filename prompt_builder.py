"""
Utilities for composing consistent prompts for multi-view product renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


VIEW_DESCRIPTIONS: Dict[str, str] = {
    "front": "front view, straight angle",
    "side": "90-degree sleeve-side view, no front visible",
    "back": "back view, straight angle",
}

GLOBAL_STYLING = (
    "same clothing item for every view, same color and fabric, clean white background, "
    "isolated e-commerce product photo, studio soft lighting, sharp fabric texture, "
    "strictly no humans, no models, no mannequins, no props"
)


@dataclass
class PromptBuilder:
    """Builds harmonized prompts for the different garment views."""

    base_tags: List[str] = field(
        default_factory=lambda: [
            "4k render",
            "sharp texture",
            "soft shadows",
        ]
    )

    def compose_core_prompt(
        self,
        product_name: str,
        design_details: str,
        extra_instructions: str,
    ) -> str:
        """
        Merge the fields from the GUI into a single textual description.
        """
        parts = []

        if product_name:
            parts.append(product_name.strip())

        if design_details:
            parts.append(design_details.strip())

        if extra_instructions:
            parts.append(extra_instructions.strip())

        parts.append(GLOBAL_STYLING)
        parts.extend(self.base_tags)

        return ", ".join(filter(None, parts))

    def prompt_for_view(self, core_prompt: str, view: str) -> str:
        if view not in VIEW_DESCRIPTIONS:
            raise ValueError(f"Unsupported view: {view}")

        prompt = f"{core_prompt}, {VIEW_DESCRIPTIONS[view]}"
        return self._truncate(prompt)

    def prompts_for_all_views(self, core_prompt: str) -> Dict[str, str]:
        return {view: self.prompt_for_view(core_prompt, view) for view in VIEW_DESCRIPTIONS}

    @staticmethod
    def _truncate(text: str, max_words: int = 70) -> str:
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])


