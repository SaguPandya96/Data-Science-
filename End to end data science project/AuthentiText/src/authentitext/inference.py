"""Versioned local inference contract for the frozen AuthentiText baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from authentitext.modeling.baselines import positive_scores, sha256_file
from authentitext.modeling.calibration import calibrate_scores

INFERENCE_SCHEMA_VERSION = 1
MAX_CHARACTERS = 100_000
SHORT_TEXT_TOKENS = 50
DEVELOPMENT_MIN_TOKENS = 6
DEVELOPMENT_MAX_TOKENS = 10_090
CANONICAL_DISCLAIMER = (
    "This system provides a statistical estimate and should not be treated as proof of authorship."
)

LIMITATIONS = [
    {
        "code": "not_authorship_proof",
        "message": CANONICAL_DISCLAIMER,
    },
    {
        "code": "english_mage_baseline",
        "message": (
            "The model was developed on English MAGE data and can fail under new domains, "
            "generators, languages, editing, or paraphrasing."
        ),
    },
    {
        "code": "uneven_subgroups",
        "message": (
            "Held-out results show substantial domain and short-text error variation; "
            "aggregate rates do not guarantee an individual result."
        ),
    },
]


class PredictionError(ValueError):
    """Raised when an inference request violates the public contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PredictorIdentity:
    dataset_id: str
    revision: str
    base_model_sha256: str
    calibration_sha256: str


class AuthentiTextPredictor:
    """Load frozen artifacts once and return calibrated, abstaining decisions."""

    def __init__(
        self,
        *,
        base_model: dict[str, Any],
        calibration: dict[str, Any],
        identity: PredictorIdentity,
    ) -> None:
        if base_model.get("model_type") != "word_tfidf_logistic":
            raise PredictionError("invalid_artifact", "Base artifact has an invalid model type")
        if calibration.get("model_type") != "calibration_policy":
            raise PredictionError(
                "invalid_artifact", "Calibration artifact has an invalid model type"
            )
        if calibration.get("base_model_sha256") != identity.base_model_sha256:
            raise PredictionError(
                "artifact_link_mismatch", "Calibration artifact is linked to another base model"
            )
        human_threshold = calibration.get("human_threshold")
        machine_threshold = calibration.get("machine_threshold")
        if (
            not isinstance(human_threshold, int | float)
            or not isinstance(machine_threshold, int | float)
            or not 0 <= human_threshold < machine_threshold <= 1
        ):
            raise PredictionError("invalid_artifact", "Calibration thresholds are invalid")
        self._base_model = base_model
        self._calibration = calibration
        self.identity = identity

    @classmethod
    def from_reports(
        cls,
        *,
        training_report_path: Path,
        calibration_report_path: Path,
        artifact_root: Path,
    ) -> AuthentiTextPredictor:
        """Load reports and reject missing, altered, or mislinked artifacts."""
        try:
            training_report = json.loads(training_report_path.read_text(encoding="utf-8"))
            calibration_report = json.loads(calibration_report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PredictionError("invalid_report", f"Cannot load model report: {error}") from error
        try:
            base_identity = next(
                artifact
                for artifact in training_report["artifacts"]
                if artifact["model_type"] == "word_tfidf_logistic"
            )
            calibration_identity = calibration_report["artifact"]
            base_path = artifact_root / base_identity["relative_path"]
            calibration_path = artifact_root / calibration_identity["relative_path"]
        except (KeyError, StopIteration, TypeError) as error:
            raise PredictionError(
                "invalid_report", "Model report is missing artifact identity"
            ) from error
        cls._verify_identity(base_path, base_identity, "base")
        cls._verify_identity(calibration_path, calibration_identity, "calibration")
        try:
            base_model = joblib.load(base_path)
            calibration = joblib.load(calibration_path)
        except Exception as error:
            raise PredictionError(
                "invalid_artifact", f"Cannot load model artifact: {error}"
            ) from error
        identity = PredictorIdentity(
            dataset_id=training_report["dataset_id"],
            revision=training_report["revision"],
            base_model_sha256=base_identity["sha256"],
            calibration_sha256=calibration_identity["sha256"],
        )
        return cls(base_model=base_model, calibration=calibration, identity=identity)

    @staticmethod
    def _verify_identity(path: Path, identity: dict[str, Any], label: str) -> None:
        if not path.is_file():
            raise PredictionError("missing_artifact", f"Missing {label} artifact")
        if path.stat().st_size != identity.get("bytes"):
            raise PredictionError(
                "artifact_size_mismatch", f"{label.title()} artifact size mismatch"
            )
        if sha256_file(path) != identity.get("sha256"):
            raise PredictionError(
                "artifact_hash_mismatch", f"{label.title()} artifact SHA-256 mismatch"
            )

    @property
    def human_threshold(self) -> float:
        return float(self._calibration["human_threshold"])

    @property
    def machine_threshold(self) -> float:
        return float(self._calibration["machine_threshold"])

    def _validate_text(self, text: Any) -> str:
        if not isinstance(text, str):
            raise PredictionError("text_type", "Text must be a string")
        if not text.strip():
            raise PredictionError("text_blank", "Text must contain a non-whitespace character")
        if "\0" in text:
            raise PredictionError("text_null", "Text must not contain a NUL character")
        if len(text) > MAX_CHARACTERS:
            raise PredictionError(
                "text_too_long", f"Text must contain at most {MAX_CHARACTERS} characters"
            )
        return text

    def predict(self, text: Any) -> dict[str, Any]:
        """Score one text without persisting or returning the submitted text."""
        validated = self._validate_text(text)
        whitespace_tokens = len(validated.split())
        raw_score = float(
            np.asarray(positive_scores(self._base_model, [validated]), dtype=np.float64)[0]
        )
        calibrated_score = float(
            calibrate_scores(
                self._calibration["calibration_method"],
                self._calibration["calibrator"],
                np.asarray([raw_score], dtype=np.float64),
            )[0]
        )
        if calibrated_score <= self.human_threshold:
            category = "likely_human"
        elif calibrated_score >= self.machine_threshold:
            category = "likely_machine"
        else:
            category = "uncertain"

        warnings = []
        if whitespace_tokens < SHORT_TEXT_TOKENS:
            warnings.append(
                {
                    "code": "short_text_low_evidence",
                    "message": (
                        "Inputs below 50 whitespace tokens performed poorly in held-out "
                        "testing; treat this result as low evidence."
                    ),
                }
            )
        if whitespace_tokens < DEVELOPMENT_MIN_TOKENS or whitespace_tokens > DEVELOPMENT_MAX_TOKENS:
            warnings.append(
                {
                    "code": "length_outside_development_range",
                    "message": "Input length is outside the observed development-data range.",
                }
            )
        unusual_features = self._out_of_profile_features(validated)
        if unusual_features:
            warnings.append(
                {
                    "code": "format_out_of_profile",
                    "message": (
                        "Input contains formatting not observed by the predefined development "
                        "profile."
                    ),
                    "features": unusual_features,
                }
            )
        return {
            "schema_version": INFERENCE_SCHEMA_VERSION,
            "category": category,
            "calibrated_machine_likelihood": round(calibrated_score, 6),
            "raw_model_score": round(raw_score, 6),
            "evidence_quality": "low" if warnings else "standard",
            "input_summary": {
                "characters": len(validated),
                "whitespace_tokens": whitespace_tokens,
            },
            "thresholds": {
                "likely_human_max": round(self.human_threshold, 12),
                "likely_machine_min": round(self.machine_threshold, 12),
            },
            "warnings": warnings,
            "limitations": LIMITATIONS,
            "model": {
                "name": "word_tfidf_logistic_isotonic",
                "dataset_id": self.identity.dataset_id,
                "dataset_revision": self.identity.revision,
                "base_model_sha256": self.identity.base_model_sha256,
                "calibration_sha256": self.identity.calibration_sha256,
            },
        }

    @staticmethod
    def _out_of_profile_features(text: str) -> list[str]:
        casefolded = text.casefold()
        flags = {
            "markdown_fence": "```" in text,
            "newline": "\n" in text or "\r" in text,
            "non_ascii": any(ord(character) > 127 for character in text),
            "repeated_space": "  " in text,
            "url_marker": "http://" in casefolded or "https://" in casefolded,
            "surrounding_whitespace": text != text.strip(),
            "heading_marker": text.lstrip().startswith("# "),
        }
        return [name for name, present in flags.items() if present]
