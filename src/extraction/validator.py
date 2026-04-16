"""Post-extraction validation and targeted correction re-prompt loop."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from src.models import ExtractionSchema, RawRegion, RegionType

if TYPE_CHECKING:
    from src.extraction.vlm_client import VLMClient

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """A single named validation rule scoped to a specific region type.

    Attributes:
        region_type: The RegionType this rule applies to, or None to apply to
            all region types.
        name: Short identifier used in log messages.
        check: Callable that receives the raw VLM response dict and returns
            True when the response is considered valid for this rule.
        correction_hint: Human-readable instruction injected into the
            correction re-prompt when check returns False.
    """

    region_type: RegionType | None
    name: str
    check: Callable[[dict], bool]
    correction_hint: str


TABLE_ROW_CONSISTENCY = ValidationRule(
    region_type=RegionType.TABLE,
    name="row_column_count",
    check=lambda r: (
        not r.get("rows")
        or not r.get("headers")
        or all(len(row) == len(r["headers"]) for row in r["rows"])
    ),
    correction_hint=(
        "Each data row must have exactly the same number of columns as the headers list. "
        "If a cell is empty, use an empty string."
    ),
)

FORMULA_LATEX_PRESENT = ValidationRule(
    region_type=RegionType.FORMULA,
    name="latex_present",
    check=lambda r: bool(r.get("latex", "").strip()),
    correction_hint=(
        "Return the formula as LaTeX in the 'latex' field. "
        "Do not leave it empty or missing."
    ),
)

OVERVIEW_PRESENT = ValidationRule(
    region_type=None,
    name="overview_present",
    check=lambda r: bool(r.get("overview", "").strip()),
    correction_hint=(
        "Include a 1-2 sentence plain-English description of this region "
        "in the 'overview' field."
    ),
)

BUILTIN_RULES: list[ValidationRule] = [
    TABLE_ROW_CONSISTENCY,
    FORMULA_LATEX_PRESENT,
    OVERVIEW_PRESENT,
]


class Validator:
    """Validates VLM extraction responses and triggers targeted correction re-prompts.

    On validation failure the Validator rebuilds the system prompt by appending
    all failed correction_hints, then re-calls the VLM via the client's
    extract_with_system_prompt method. This correction loop repeats up to
    max_passes times. If the response still fails validation after all passes,
    confidence_score is set to 0.0 and correction_applied is set to True on
    the returned result.
    """

    def __init__(
        self,
        vlm_client: "VLMClient",
        max_passes: int = 2,
        extra_rules: list[ValidationRule] | None = None,
    ) -> None:
        self._vlm = vlm_client
        self._max_passes = max_passes
        self._rules: list[ValidationRule] = list(BUILTIN_RULES) + (
            extra_rules or []
        )

    async def validate_and_correct(
        self,
        region: RawRegion,
        response: dict,
        schema: ExtractionSchema | None,
    ) -> tuple[dict, bool, float]:
        """Validate response and apply a correction loop if any rules fail.

        For each correction pass, the failed rules' correction_hints are
        appended to the original extraction prompt and the VLM is called again.
        Passes continue until all rules pass or max_passes is reached.

        Returns:
            tuple[dict, bool, float]:
                - corrected_response dict (may be the original if no correction needed)
                - correction_applied (True if at least one correction pass ran)
                - confidence_score (0.0 if still failing after max_passes, else 1.0)
        """
        applicable_rules = self._get_applicable_rules(region.region_type)
        failed_rules = [r for r in applicable_rules if not r.check(response)]

        if not failed_rules:
            return response, False, 1.0

        original_prompt = self._vlm.build_extraction_prompt(region.region_type, schema)
        current_response = response
        correction_applied = False

        for pass_num in range(1, self._max_passes + 1):
            correction_prompt = self._build_correction_prompt(
                original_prompt, failed_rules
            )
            logger.info(
                "Correction pass %d/%d for region %s (%s): failed rules: %s",
                pass_num,
                self._max_passes,
                region.region_id,
                region.region_type.value,
                [r.name for r in failed_rules],
            )
            current_response = await self._vlm.extract_with_system_prompt(
                region=region,
                system_prompt=correction_prompt,
                max_tokens=self._vlm._config.extraction_max_tokens,
            )
            correction_applied = True

            failed_rules = [
                r for r in applicable_rules if not r.check(current_response)
            ]
            if not failed_rules:
                logger.info(
                    "Correction succeeded on pass %d for region %s.",
                    pass_num,
                    region.region_id,
                )
                return current_response, True, 1.0

        logger.warning(
            "All %d correction passes exhausted for region %s (%s); "
            "still failing rules: %s.",
            self._max_passes,
            region.region_id,
            region.region_type.value,
            [r.name for r in failed_rules],
        )
        return current_response, correction_applied, 0.0

    def _get_applicable_rules(self, region_type: RegionType) -> list[ValidationRule]:
        """Return all rules applicable to the given region_type.

        A rule is applicable when its region_type attribute matches the given
        type exactly, or when it is None (applies to all types).

        Returns:
            list[ValidationRule]: matching rules in definition order.
        """
        return [
            r for r in self._rules
            if r.region_type is None or r.region_type == region_type
        ]

    @staticmethod
    def _build_correction_prompt(
        original_prompt: str,
        failed_rules: list[ValidationRule],
    ) -> str:
        """Build a correction prompt by appending hints from all failed rules.

        Each hint is labelled with the rule name and set off on its own line to
        ensure the model treats each constraint independently.

        Returns:
            str: augmented prompt string with correction instructions appended.
        """
        hint_lines: list[str] = [
            f"CORRECTION REQUIRED ({rule.name}): {rule.correction_hint}"
            for rule in failed_rules
        ]
        correction_block = (
            "The previous response did not satisfy the following requirements. "
            "Re-extract and fix only the issues listed below:\n\n"
            + "\n".join(hint_lines)
        )
        return original_prompt + "\n\n" + correction_block
