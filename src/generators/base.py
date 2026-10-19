import re
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

# Standard prompt used across all generators for consistency.
STANDARD_PROMPT = (
    "Describe the findings in this chest X-ray image. "
    "Provide a detailed radiology report covering all visible abnormalities "
    "and normal findings. Do not include headers, greetings, or conversational text."
)

# Variant for text-only models that cannot see the image.
STANDARD_PROMPT_TEXT_ONLY = (
    "Write a chest X-ray radiology findings section based on the clinical context below. "
    "Note: no image is available to this model. "
    "Provide a plausible detailed report covering common findings. "
    "Do not include headers, greetings, or conversational text."
)


def clean_report(text: str) -> str:
    """Shared post-processing to strip common LLM artifacts from generated reports."""
    if not text:
        return text

    # Remove conversational prefixes
    conv_prefixes = [
        r"^Sure[,.]?\s*[Hh]ere(?:'s| is)\s*(?:the\s+)?(?:detailed\s+)?(?:radiology\s+)?(?:report|findings)[^:]*:\s*",
        r"^Here(?:'s| is)\s*(?:the\s+)?(?:detailed\s+)?(?:radiology\s+)?(?:report|findings)[^:]*:\s*",
        r"^(?:Of course|Certainly)[!,.]?\s*(?:Here(?:'s| is)[^:]*:\s*)?",
    ]
    for pat in conv_prefixes:
        text = re.sub(pat, "", text, count=1, flags=re.IGNORECASE)

    text = re.sub(
        r"^Based on my review of the provided image,?\s+here(?:'s| is)\s+a\s+summary\s*:?\s*",
        "",
        text,
        count=1,
        flags=re.IGNORECASE,
    )

    # Remove markdown-style headers like **Findings:**, **Impression:**, ## Findings, etc.
    text = re.sub(r"\*\*(?:Findings|Impression|Report|Radiology Report|Conclusion)\s*:?\*\*\s*", "", text, flags=re.IGNORECASE)
    # Remove ALL markdown headers (## ..., ### ...), not just specific keywords
    text = re.sub(r"^#{1,6}\s+[^\n]*\n?", "", text, flags=re.MULTILINE)
    # Plain "FINAL REPORT" is a common radiology section header, not an assistant tail.
    text = re.sub(r"^\s*final\s+report\s*:?\s*", "", text, count=1, flags=re.IGNORECASE)

    # Remove [Insert ...] placeholders
    text = re.sub(r"\[(?:Insert|State|Enter|Add)[^\]]*\]", "", text, flags=re.IGNORECASE)

    # Remove template header lines (Patient Name, MRN, Date of Exam, etc.)
    text = re.sub(
        r"^\*\*(?:Patient Name|Medical Record|MRN|Date of Exam|Exam Type|Indication|Technique|Clinical History)\*\*[^\n]*\n?",
        "", text, flags=re.IGNORECASE | re.MULTILINE,
    )

    # Remove remaining XML-like tags (except known model-specific ones handled elsewhere)
    text = re.sub(r"</?(?:findings|impression|report)>", "", text, flags=re.IGNORECASE)

    # Remove all remaining markdown bold markers (e.g. **Heart Size:** -> Heart Size:)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)

    # Remove markdown bullet points (*, -, numbered) at line start -> plain sentences
    text = re.sub(r"^\s*[\*\-]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)

    # Remove JSON code blocks (closed or truncated/unclosed) and inline JSON artifacts
    text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
    text = re.sub(r"```.*", "", text, flags=re.DOTALL)  # unclosed blocks -> strip to end
    text = re.sub(r"\{[^{}]*\}", "", text)
    text = re.sub(r"&#x[0-9A-Fa-f]+;", " ", text)
    text = re.sub(r"\$?\\boxed(?:\{[^}\n]*\})?\$?", "", text)

    # Remove terminal assistant/math/code sections that often follow the
    # generated report and distort report-level evaluation.
    terminal_pats = [
        r"\n?\s*(?:-{3,}|\*{3,}|_{3,})\s*\n.*\Z",
        r"\n?\s*(?:Final Answer|The final answer is|Output)\s*:?.*\Z",
        r"\n?\s*(?:Important (?:Considerations|Note)|Disclaimer|Limitations)\s*:?.*\Z",
        r"\n?\s*(?:Additional information|Additional considerations)\s*:?.*\Z",
        r"\n\s*(?:python\s*\n|def\s+analyze_chest_xray\b).*\Z",
    ]
    for pat in terminal_pats:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove citation-only lines that some instruction-tuned models emit.
    text = re.sub(
        r"^.*\b(?:doi:|https?://doi\.org/|pmid:|arxiv:).*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    # Remove disclaimers / non-clinical boilerplate (entire line or paragraph)
    disclaimer_pats = [
        r"This (?:information|response|report|image|is not|should not).*?(?:alone|decisions|diagnosis|professional|substitute|replacement)[s.]?[^\n]*",
        r"(?:Note|Disclaimer|Important|Warning)\s*:.*?(?:alone|decisions|diagnosis|professional|consult|substitute)[s.]?[^\n]*",
        r"It(?:'s| is) important (?:that you |to )?(?:consult|note|remember)[^\n]*",
        r"(?:Please )?[Cc]onsult (?:an? )?(?:expert |qualified )?(?:radiologist|physician|doctor|healthcare)[^\n]*",
        r"Please note\s*:?[^\n]*",
        r"I recommend[^\n]*",
        r"I (?:would |strongly )?(?:recommend|suggest|advise)[^\n]*",
        r"I am an AI assistant[^\n]*",
        r"I am unable to provide medical advice[^\n]*",
        r"I am ready for your next question[^\n]*",
        r"I have tried to address[^\n]*",
        r"Please ask me any other questions[^\n]*",
        r"Please let me know[^\n]*",
        r"feel free to ask[^\n]*",
        r"(?:Okay|Thanks),?[^\n]*(?:x-ray|question)[^\n]*",
        r"Based on these observations alone I cannot provide any specific diagnosis[^\n]*",
        r"A more comprehensive assessment requires correlation[^\n]*",
        r"No specific recommendation could be made[^\n]*",
        r"thorough follow[- ]?up by qualified healthcare professionals[^\n]*",
        r"This interpretation should not replace[^\n]*",
        r"This analysis[^.\n]*should not replace[^\n]*",
        r"(?:Further|Additional) (?:clinical |imaging )?(?:correlation|evaluation|assessment) (?:is |may be )?(?:recommended|suggested|advised|warranted)[^\n]*",
        r"This (?:AI|model|system|tool) (?:is not|should not|cannot)[^\n]*",
    ]
    for pat in disclaimer_pats:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Remove inline code backticks
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Collapse excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize runs of spaces (but keep newlines)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


class BaseGenerator(ABC):
    """
    Base interface for all report generators.
    """
    name: str = "base"

    @abstractmethod
    def generate(self, image_paths: List[str], prior_image_paths: Optional[List[str]] = None, context: Optional[str] = None) -> str:
        """
        Generate a radiology report from one or more X-ray images.

        Args:
            image_paths: List of paths to current study images (e.g., frontal, lateral).
            prior_image_paths: List of paths to prior study images (if available).
            context: Additional clinical context, indication, or history.

        Returns:
            Generated report text.
        """
        pass

    def generate_batch(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        """
        Generate reports for a batch of studies. Default implementation loops over `generate`.
        Can be overridden by subclasses for optimized batch processing.

        Args:
            batch_data: List of dictionaries containing 'image_paths', 'prior_image_paths', 'context'.

        Returns:
            List of generated report texts.
        """
        return [
            self.generate(
                image_paths=item["image_paths"],
                prior_image_paths=item.get("prior_image_paths"),
                context=item.get("context")
            )
            for item in batch_data
        ]

