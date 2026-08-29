"""
AI drawing extraction for QuoteBuilder.

Uses Google Gemini to read engineering drawings (PDF or image)
and return structured manufacturing data as JSON.

Important:
- Extract only information actually present on the drawing.
- Never invent material, quantity, dimensions, rates, or costs.
- Missing information is explicitly flagged.
"""

import base64
import json
import mimetypes
import os
import time
from pathlib import Path

import httpx
import pymupdf as fitz


# ---------------------------------------------------------
# Gemini configuration
# ---------------------------------------------------------

MODEL = "gemini-3.7-flash"
FALLBACK_MODEL = "gemini-3.6-flash"  # gemini-2.5-flash was deprecated for new users by Google -- updated per their own error message

# PDF rendering: engineering drawings (especially D-size+) pack dense small text
# across the whole sheet. A single full-page render is rarely legible to a vision
# model, so we render one overview + a grid of zoomed, overlapping crops -- the
# same approach a human would use (zoom into each area to read the fine print).
OVERVIEW_DPI = 200
TILE_DPI = 450
TILE_GRID = (2, 2)  # (cols, rows)
TILE_OVERLAP_FRACTION = 0.08  # crops overlap slightly so nothing gets cut off at a seam

SYSTEM_PROMPT = """
You are an expert Manufacturing Engineer and CNC Process Planner
reading a real engineering drawing.

You will receive multiple images. If more than one image is provided, the
FIRST image is a full-page overview of the drawing sheet for context, and the
REMAINING images are zoomed-in, overlapping crops of that SAME single sheet
(different quadrants, sometimes multiple pages). Read the crops for fine
print, dimensions, tolerances, and title block details, and use the overview
to understand how the crops fit together spatially. Treat all images together
as ONE drawing, not separate parts.

Your job is to extract manufacturing information from the drawing.

CRITICAL RULE:
Extract ONLY what is actually printed or clearly shown on the drawing.

DO NOT invent:
- material
- quantity
- dimensions
- tolerances
- machining rates
- process rates
- costs
- missing drawing information

If something is not available on the drawing, return null where appropriate
and add a clear explanation to "missing_info".

Return ONLY valid JSON.
Do not return markdown.
Do not return ```json.
Do not add any explanation before or after the JSON.

Use exactly this structure:

{
  "drawing_no": string or null,
  "part_name": string or null,
  "revision": string or null,
  "material": string or null,
  "material_key": string or null,
  "quantity": number or null,

  "stock": {
    "shape": "one of: ROUND_BAR, HEX_BAR, TUBE, PLATE -- the raw stock this part is machined from, or null if the drawing doesn't show enough to determine it",
    "od_mm": number or null (largest outside diameter for round/tube, across-flats for hex, in mm),
    "id_mm": number or null (through-bore/inside diameter, ONLY if a bore runs the full length -- for TUBE shape),
    "length_mm": number or null (overall length for bars, longest side for plate, in mm),
    "width_mm": number or null (PLATE shape only, mm),
    "thickness_mm": number or null (PLATE shape only, mm),
    "basis": "string or null -- briefly cite which printed dimensions this is based on, e.g. 'from Ø6.870 OD x 647mm overall length on the drawing'"
  },

  "features": [
    {
      "feature": string,
      "operation": string,
      "machine_category": string,
      "est_hours": number or null,
      "confidence": "high" | "inferred" | "unclear" | "missing"
    }
  ],

  "special_processes": [
    {
      "process": string,
      "process_key": string or null,
      "requirement": string,
      "confidence": "high" | "inferred" | "unclear" | "missing"
    }
  ],

  "missing_info": [
    string
  ],

  "raw_notes": string
}

MACHINE CATEGORY RULE (important):
Valid machine_category values are: CNC_LATHE, VMC, HMC, GRINDING, MANUAL_BENCH, DEEP_HOLE_DRILL.
Classify a feature as DEEP_HOLE_DRILL instead of CNC_LATHE/VMC when EITHER:
- the hole/bore's depth-to-diameter ratio is greater than roughly 5:1, OR
- the overall part length exceeds 1000mm AND a bore (through or blind) runs most of that length.
Deep-hole drilling requires specialized equipment (gun drilling / BTA rigs) and costs
significantly more than ordinary lathe boring -- do not under-classify a long bored
mandrel or shaft as ordinary CNC_LATHE work.

EST_HOURS RULE (important -- this is required, not optional):
For every feature where confidence is "high" or "inferred", you MUST provide a
numeric est_hours value -- do not return null just because you lack shop-specific
cycle-time data. This is explicitly a rough planning estimate, not a shop quote,
so a reasonable industry-standard benchmark is expected and appropriate. Use these
as rough anchors and adjust for the feature's actual size/complexity:
- Simple OD/ID turning or facing pass: 0.1-0.4 hr
- Threading (single-point, per thread): 0.2-0.6 hr, more for non-standard/large-pitch threads
- Drilling/tapping a hole (per hole or small group): 0.05-0.2 hr each
- Deep-hole drilling (DEEP_HOLE_DRILL): 0.5-3+ hr depending on depth
- Slot/keyway milling (VMC/HMC), per feature or per repeated pattern: 0.2-0.8 hr
- Grinding pass: 0.2-0.6 hr
Only use null for est_hours when confidence is "unclear" or "missing" -- i.e. you
genuinely cannot tell what the operation even is, not merely because you're unsure
of the exact time.
"""


def _get_mime_type(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
    }
    if suffix in mime_map:
        return mime_map[suffix]
    guessed_type, _ = mimetypes.guess_type(file_path)
    if guessed_type:
        return guessed_type
    raise ValueError(f"Unsupported drawing file type: {suffix or 'unknown'}")


def _render_pdf_to_images(file_path: str) -> list[bytes]:
    """
    Render every page of a PDF into: 1 full-page overview PNG (for context)
    + a grid of high-DPI overlapping crop PNGs (for reading fine print).
    Returns a flat list of PNG bytes across all pages.
    """
    images: list[bytes] = []
    doc = fitz.open(file_path)
    try:
        cols, rows = TILE_GRID
        for page in doc:
            rect = page.rect

            # 1. Overview (whole page, modest resolution, gives spatial context)
            overview_zoom = OVERVIEW_DPI / 72
            overview_pix = page.get_pixmap(matrix=fitz.Matrix(overview_zoom, overview_zoom))
            images.append(overview_pix.tobytes("png"))

            # 2. Zoomed, overlapping crops for legibility of small text/dimensions
            tile_w = rect.width / cols
            tile_h = rect.height / rows
            overlap_w = tile_w * TILE_OVERLAP_FRACTION
            overlap_h = tile_h * TILE_OVERLAP_FRACTION
            tile_zoom = TILE_DPI / 72
            tile_matrix = fitz.Matrix(tile_zoom, tile_zoom)

            for r in range(rows):
                for c in range(cols):
                    x0 = max(rect.x0, rect.x0 + c * tile_w - overlap_w)
                    x1 = min(rect.x1, rect.x0 + (c + 1) * tile_w + overlap_w)
                    y0 = max(rect.y0, rect.y0 + r * tile_h - overlap_h)
                    y1 = min(rect.y1, rect.y0 + (r + 1) * tile_h + overlap_h)
                    clip = fitz.Rect(x0, y0, x1, y1)
                    tile_pix = page.get_pixmap(matrix=tile_matrix, clip=clip)
                    images.append(tile_pix.tobytes("png"))
    finally:
        doc.close()
    return images


def _clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _empty_result(error_message: str, raw_text: str = "") -> dict:
    return {
        "drawing_no": None, "part_name": None, "revision": None,
        "material": None, "material_key": None, "quantity": None, "stock": None,
        "features": [], "special_processes": [],
        "missing_info": [error_message],
        "raw_notes": raw_text[:1000] if raw_text else "",
    }


def extract_drawing(file_path: str) -> dict:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _empty_result("GEMINI_API_KEY is not configured on the server.")

    try:
        mime_type = _get_mime_type(file_path)

        if mime_type == "application/pdf":
            image_bytes_list = _render_pdf_to_images(file_path)
            image_parts = [
                {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(b).decode("ascii")}}
                for b in image_bytes_list
            ]
        else:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
            image_parts = [{"inline_data": {"mime_type": mime_type, "data": base64.b64encode(file_bytes).decode("ascii")}}]

        prompt = (
            "Analyze the attached engineering drawing images (one overview plus "
            "zoomed crops of the same sheet, as described in the system instructions). "
            "Return ONLY the requested JSON object."
        )

        request_body = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [*image_parts, {"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "max_output_tokens": 4000,
            },
        }

        text = None
        last_error = None
        # Same fallback behavior as before: try the primary model, then the
        # fallback model, each with its own short retry loop for transient
        # server-side errors (high demand / 5xx). Using httpx directly instead
        # of the google-genai SDK cuts this process's baseline memory by
        # ~100MB, which is what caused the Render free-tier OOM crashes.
        for model_name in [MODEL, FALLBACK_MODEL]:
            for attempt in range(3):
                try:
                    with httpx.Client(timeout=120.0) as client:
                        resp = client.post(
                            f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
                            params={"key": api_key},
                            json=request_body,
                        )
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]
                        break
                    err_body = resp.text
                    retryable = resp.status_code in (408, 429, 500, 502, 503, 504)
                    last_error = Exception(f"{model_name}: HTTP {resp.status_code} - {err_body[:500]}")
                    if retryable and attempt < 2:
                        time.sleep(3 * (attempt + 1))
                        continue
                    break  # not retryable, or out of attempts for this model
                except Exception as e:
                    last_error = e
                    if attempt < 2:
                        time.sleep(3 * (attempt + 1))
                        continue
            if text is not None:
                break  # got a result from this model, don't try the fallback

        if text is None:
            raise last_error or Exception("No response from either model.")

        text = _clean_json_text(text)
        try:
            result = json.loads(text)
        except json.JSONDecodeError as e:
            return _empty_result(f"AI response could not be parsed as JSON: {e}", text)

        required_keys = ["drawing_no","part_name","revision","material","material_key",
                          "quantity","stock","features","special_processes","missing_info","raw_notes"]
        for key in required_keys:
            if key not in result:
                result[key] = None
        if not isinstance(result.get("features"), list):
            result["features"] = []
        if not isinstance(result.get("special_processes"), list):
            result["special_processes"] = []
        if not isinstance(result.get("missing_info"), list):
            result["missing_info"] = []
        from .weight import coerce_stock
        result["stock"] = coerce_stock(result.get("stock"))
        return result

    except Exception as e:
        return _empty_result(f"Gemini drawing extraction failed (tried {MODEL} and {FALLBACK_MODEL}): {type(e).__name__}: {e}")
