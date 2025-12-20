"""
FastAPI backend for the Nutrition Planner demo.
Features:
- Macro calculation via Mifflin-St Jeor + activity level.
- LLM-generated plan via local Ollama (required).
- Dummy pricing catalog for cost estimation (no real product search).
- CORS enabled for static frontend calls.
"""

import json
import os
import csv
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, validator
import requests

from .providers.hf_recipes import generate_hf_recipes, get_last_hf_error, get_hf_pipeline
from .food_image_to_txt import describe_image_from_bytes
from .rag_service import RagService, load_seed_documents
from .instacart_hook import create_shopping_list_with_search_links, generate_html_page


ROOT_DIR = Path(__file__).resolve().parents[2]
_EXECUTOR = ThreadPoolExecutor(max_workers=4)


import os

app = FastAPI(title="Nutrition Planner", version="0.1.0")

# Allow browser demos from static hosts (e.g., GitHub Pages hitting localhost backend).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_model_catalog() -> dict:
    """
    Load optional model lists from config/models.json with keys 'huggingface', 'ollama', 'openai'.
    """
    path = ROOT_DIR / "config" / "models.json"
    default = {"huggingface": [], "ollama": [], "openai": []}
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text())
        hf = [m for m in data.get("huggingface", []) if isinstance(m, str)]
        ollama = [m for m in data.get("ollama", []) if isinstance(m, str)]
        openai = [m for m in data.get("openai", []) if isinstance(m, str)]
        return {"huggingface": hf, "ollama": ollama, "openai": openai}
    except Exception:
        return default


def _run_with_timeout(func: Callable[[], Tuple], timeout_seconds: int):
    """Run a blocking function in a thread with a timeout. Raises TimeoutError on expiry."""
    future = _EXECUTOR.submit(func)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError as exc:
        future.cancel()
        raise exc


_OLLAMA_LAST_ERROR: Optional[str] = None
_LAST_MODEL_RAW: Optional[str] = None


def _log_model_output(provider: str, text: str) -> None:
    """Lightweight stdout logger for debugging model responses."""
    global _LAST_MODEL_RAW
    _LAST_MODEL_RAW = f"[{provider}] {text}"
    print(f"[MODEL DEBUG] {provider} output:\n{text}\n", flush=True)


class OpenAIRecipeClient:
    def __init__(self, model_id: Optional[str] = None):
        self.model_id = model_id or _default_openai_model(None)

    def _client(self):
        try:
            from openai import OpenAI  # type: ignore

            return OpenAI()
        except Exception:
            return None

    def generate(self, prompt: str) -> Tuple[Optional[str], Optional[str]]:
        client = self._client()
        if not client:
            return None, "OpenAI SDK not available"
        model = self.model_id
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            txt = resp.choices[0].message.content
            _log_model_output(f"openai:{model}", txt)
            return txt, None
        except Exception as exc:
            return None, str(exc)


@app.on_event("startup")
def _init_rag() -> None:
    """
    Initialize RAG service and auto-ingest seed docs/CSVs once.
    Safe to continue without RAG if dependencies are missing.
    """
    try:
        rag = RagService(ROOT_DIR / "data" / "chroma")
        seed_docs = load_seed_documents(ROOT_DIR)
        if rag.is_empty() and seed_docs:
            rag.ingest_documents(seed_docs)
        app.state.rag = rag
        app.state.rag_seed_count = len(seed_docs)
    except Exception as exc:
        app.state.rag = None
        app.state.rag_seed_count = 0
        app.state.rag_error = str(exc)


class Sex(str, Enum):
    male = "male"
    female = "female"


class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    heavy = "heavy"
    athlete = "athlete"

    @property
    def multiplier(self) -> float:
        return {
            ActivityLevel.sedentary: 1.2,
            ActivityLevel.light: 1.375,
            ActivityLevel.moderate: 1.55,
            ActivityLevel.heavy: 1.725,
            ActivityLevel.athlete: 1.9,
        }[self]


class DietaryPreference(str, Enum):
    omnivore = "omnivore"
    non_vegeterian = "omnivore"
    vegetarian = "vegetarian"
    vegan = "vegan"
    pescatarian = "pescatarian"
    keto = "keto"
    paleo = "paleo"


class MacroSplit(BaseModel):
    protein_pct: float = Field(30, ge=0, le=100)
    carbs_pct: float = Field(40, ge=0, le=100)
    fat_pct: float = Field(30, ge=0, le=100)

    @validator("fat_pct")
    def validate_sum(cls, v, values):  # type: ignore[override]
        total = v + values.get("protein_pct", 0) + values.get("carbs_pct", 0)
        if abs(total - 100) > 0.001:
            raise ValueError("protein_pct + carbs_pct + fat_pct must sum to 100")
        return v


class Profile(BaseModel):
    height_cm: float = Field(..., gt=0)
    weight_kg: float = Field(..., gt=0)
    age: int = Field(..., gt=0)
    sex: Sex
    activity_level: ActivityLevel


class PlanRequest(BaseModel):
    profile: Profile
    dietary_preference: DietaryPreference = DietaryPreference.omnivore
    allergies: List[str] = Field(default_factory=list)
    budget_amount: Optional[float] = Field(None, gt=0)
    meals_per_day: int = Field(3, ge=1, le=6)
    macro_split: MacroSplit = MacroSplit()
    target_calories: Optional[int] = Field(None, gt=800, lt=6000)


class Ingredient(BaseModel):
    name: str
    quantity: str
    notes: Optional[str] = None


class Meal(BaseModel):
    name: str
    calories: int
    protein_g: int
    carbs_g: int
    fat_g: int
    ingredients: List[Ingredient]


class DayPlan(BaseModel):
    day: int
    meals: List[Meal]
    total_calories: int
    total_protein_g: int
    total_carbs_g: int
    total_fat_g: int


class PlanResponse(BaseModel):
    target_calories: int
    target_protein_g: int
    target_carbs_g: int
    target_fat_g: int
    budget_amount: Optional[float]
    days: List[DayPlan]
    estimated_cost: Optional[float] = None
    unknown_items: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

class Goal(str, Enum):
    fat_loss = "fat_loss"
    muscle_gain = "muscle_gain"
    maintenance = "maintenance"

class RecommendationPeriod(str, Enum):
    one_day = "one_day"
    seven_days = "seven_days"

class IngredientInput(BaseModel):
    name: str
    quantity: Optional[str] = None
    notes: Optional[str] = None


class IngredientCost(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    cost: Optional[float] = None
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 2) if v is not None else None
        }


class ShoppingItem(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None
    cost: Optional[float] = None
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 2) if v is not None else None
        }


class DishRecommendation(BaseModel):
    day: int
    name: str
    reason: str
    ingredients_used: List[str]
    ingredients: Optional[List[IngredientCost]] = None
    steps: List[str] = Field(default_factory=list)
    calories: Optional[float] = None
    protein_g: Optional[float] = None
    carbs_g: Optional[float] = None
    fat_g: Optional[float] = None
    estimated_cost: Optional[float] = None
    
    class Config:
        json_encoders = {
            float: lambda v: round(v, 2) if v is not None else None
        }


def _parse_quantity_with_unit(raw_quantity) -> Tuple[Optional[float], str]:
    """
    Parse a quantity that may include a unit (e.g., "100g", "2 cups", "1.5 lb").
    Returns (numeric_value, unit_string).
    If no numeric part found, returns (None, original_string).
    """
    if raw_quantity is None:
        return None, ""
    
    if isinstance(raw_quantity, (int, float)):
        return float(raw_quantity), ""
    
    if not isinstance(raw_quantity, str):
        raw_quantity = str(raw_quantity)
    
    raw_quantity = raw_quantity.strip()
    
    # Try to match number followed by optional unit
    # Supports: "100g", "2 cups", "1.5 lb", "1/2 tsp"
    match = re.match(r'\s*([\d./]+)\s*([a-zA-Z]*)\s*$', raw_quantity)
    if match:
        num_str, unit = match.groups()
        try:
            # Handle fractions like "1/2"
            if '/' in num_str:
                parts = num_str.split('/')
                num_val = float(parts[0]) / float(parts[1])
            else:
                num_val = float(num_str)
            return num_val, unit or ""
        except (ValueError, ZeroDivisionError):
            return None, raw_quantity
    
    return None, raw_quantity


def _format_quantity_with_unit(quantity: Optional[float], unit: Optional[str]) -> str:
    """
    Format quantity and unit as a single string (e.g., "100g", "2 cups").
    """
    if quantity is None:
        return unit or ""
    
    unit = (unit or "").strip()
    
    # Format the number nicely (remove unnecessary decimals)
    if quantity == int(quantity):
        qty_str = str(int(quantity))
    else:
        qty_str = f"{quantity:.1f}".rstrip('0').rstrip('.')
    
    if unit:
        return f"{qty_str}{unit}" if unit in ['g', 'kg', 'ml', 'l'] else f"{qty_str} {unit}"
    
    return qty_str


def _clean_item_name(name: str) -> str:
    """
    Remove trailing quantity/unit text from an ingredient name (e.g., 'salmon 1 piece' -> 'salmon').
    """
    cleaned = re.sub(r"\s+\d+(?:\.\d+)?\s*[a-zA-Z]*$", "", name or "").strip()
    return cleaned or name


def _coerce_number(val) -> Optional[float]:
    """
    Convert numeric-looking strings (e.g., "20g", "500 kcal") to floats.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # Extract just the numeric part
        num_val, _ = _parse_quantity_with_unit(val)
        return num_val
    return None

def _to_int(val) -> Optional[int]:
    """
    Convert to nearest int if numeric; otherwise None.
    """
    num = _coerce_number(val)
    if num is None:
        return None
    try:
        return int(round(num))
    except Exception:
        return None


def _normalize_shopping_items(items: Optional[List[ShoppingItem]]) -> List[ShoppingItem]:
    """
    Ensure every shopping item has a numeric quantity and a unit. Defaults: quantity=1, unit="" (no forced 'piece').
    """
    normalized: List[ShoppingItem] = []
    if not items:
        return normalized
    for itm in items:
        qty_val = _coerce_number(itm.quantity)
        if qty_val is None:
            qty_val = 1.0
        unit = (itm.unit or "").strip()
        cost_val = _coerce_number(itm.cost)
        normalized.append(ShoppingItem(name=_clean_item_name(itm.name), quantity=qty_val, unit=unit, cost=cost_val))
    return normalized


def _parse_ingredient_from_dict(ing_dict: dict) -> IngredientCost:
    """
    Parse an ingredient dictionary ensuring quantity and unit are properly separated.
    """
    name = str(ing_dict.get("name") or ing_dict.get("item") or "")
    raw_qty = ing_dict.get("quantity")
    raw_unit = ing_dict.get("unit") or ""
    raw_cost = ing_dict.get("cost")
    
    # Parse quantity - it might be a string with unit like "100g" or "2 cups"
    qty_val, parsed_unit = _parse_quantity_with_unit(raw_qty)
    
    # If unit wasn't in quantity string, use the separate unit field
    if not parsed_unit and raw_unit:
        parsed_unit = str(raw_unit).strip()
    
    # Parse cost
    cost_val = _coerce_number(raw_cost)
    
    return IngredientCost(
        name=name,
        quantity=qty_val,
        unit=parsed_unit,
        cost=cost_val,
    )


def _build_shopping_from_dishes(dishes: List[DishRecommendation]) -> List[dict]:
    """
    Fallback: derive shopping items from dish ingredients_used when the model
    does not return an explicit shopping_list. Quantities default to 1, unit empty.
    """
    if not dishes:
        return []
    seen = set()
    items: List[dict] = []
    for dish in dishes:
        # Prefer structured ingredients if available
        if dish.ingredients:
            for ing in dish.ingredients:
                name = _clean_item_name((ing.name or "").strip())
                key = name.lower()
                if not name or key in seen:
                    continue
                seen.add(key)
                cost_val = _coerce_number(ing.cost)
                qty_val = _coerce_number(ing.quantity)
                if qty_val is None:
                    qty_val = 1.0
                items.append({
                    "name": name,
                    "quantity": qty_val,
                    "unit": ing.unit or "",
                    "cost": cost_val,
                })
            continue
        for ing in dish.ingredients_used:
            name = _clean_item_name(ing.strip())
            key = name.lower()
            if not name or key in seen:
                continue
            seen.add(key)
            items.append({"name": name, "quantity": 1, "unit": ""})
    return items


def _build_shopping_from_dishes_ingredients(dishes_list: list) -> List[ShoppingItem]:
    """
    Aggregate shopping items from dish ingredients, summing quantities by name+unit.
    """
    # Dictionary to aggregate: key = (name.lower(), unit), value = [total_qty, total_cost, original_name]
    aggregated: Dict[Tuple[str, str], List] = {}
    
    for dish in dishes_list:
        if not isinstance(dish, dict):
            continue
            
        ingredients = dish.get("ingredients") or []
        for ing in ingredients:
            if not isinstance(ing, dict):
                continue
                
            ing_parsed = _parse_ingredient_from_dict(ing)
            if not ing_parsed.name:
                continue
            
            key = (ing_parsed.name.lower(), ing_parsed.unit or "")
            
            if key in aggregated:
                # Sum quantities and costs
                aggregated[key][0] += (ing_parsed.quantity or 0)
                aggregated[key][1] += (ing_parsed.cost or 0)
            else:
                # New entry: [quantity, cost, display_name]
                aggregated[key] = [
                    ing_parsed.quantity or 0,
                    ing_parsed.cost or 0,
                    ing_parsed.name
                ]
    
    # Convert to ShoppingItem list
    items = []
    for (name_lower, unit), [total_qty, total_cost, display_name] in aggregated.items():
        items.append(ShoppingItem(
            name=display_name,
            quantity=round(total_qty, 2) if total_qty else None,
            unit=unit or None,
            cost=round(total_cost, 2) if total_cost else None
        ))
    
    return items


class ShoppingLinksRequest(BaseModel):
    shopping_list: List[ShoppingItem]
    shopping_links: Optional[dict] = None
    title: Optional[str] = "LLM Grocery List"

class RecommendRequest(BaseModel):
    ingredients: List[str] = Field(default_factory=list)
    profile: Optional[Profile] = None
    dietary_preference: DietaryPreference = DietaryPreference.omnivore
    goal: Goal = Goal.maintenance
    allergies: List[str] = Field(default_factory=list)
    cuisine: Optional[str] = None
    period: RecommendationPeriod = RecommendationPeriod.one_day
    meals_per_day: int = Field(3, ge=1, le=6)
    macro_split: MacroSplit = MacroSplit()
    target_calories: Optional[int] = Field(None, gt=800, lt=6000)
    budget_amount: Optional[float] = Field(None, gt=0)
    hf_model_id: Optional[str] = None
    ollama_model: Optional[str] = None
    openai_model: Optional[str] = None

class RecommendationResponse(BaseModel):
    period: RecommendationPeriod
    dishes: List[DishRecommendation]
    unknown_items: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    target_calories: Optional[int] = None
    target_protein_g: Optional[int] = None
    target_carbs_g: Optional[int] = None
    target_fat_g: Optional[int] = None
    model_provider: Optional[str] = None
    model_parameters: Optional[int] = None
    cuisine: Optional[str] = None
    shopping_list: Optional[List[ShoppingItem]] = None
    shopping_links: Optional[dict] = None
    total_shopping_cost: Optional[float] = None


class RefineRequest(BaseModel):
    feedback: str
    dishes: List[DishRecommendation]
    request: Optional[RecommendRequest] = None


class RagIngestRequest(BaseModel):
    texts: List[str]
    metadata: Optional[List[dict]] = None


class RagQueryRequest(BaseModel):
    question: str
    k: int = Field(4, ge=1, le=10)



def mifflin_st_jeor(profile: Profile) -> float:
    base = 10 * profile.weight_kg + 6.25 * profile.height_cm - 5 * profile.age
    return base + (5 if profile.sex == Sex.male else -161)


def compute_targets(req: PlanRequest) -> dict:
    bmr = mifflin_st_jeor(req.profile)
    tdee = bmr * req.profile.activity_level.multiplier
    calories = int(req.target_calories or round(tdee))

    protein_cal = calories * (req.macro_split.protein_pct / 100)
    carbs_cal = calories * (req.macro_split.carbs_pct / 100)
    fat_cal = calories * (req.macro_split.fat_pct / 100)

    return {
        "calories": calories,
        "protein_g": int(protein_cal / 4),
        "carbs_g": int(carbs_cal / 4),
        "fat_g": int(fat_cal / 9),
    }


KCAL_PER_GRAM = {
    "protein": 4,
    "carbs": 4,
    "fat": 9,
}

GOAL_CALORIE_MULTIPLIER = {
    "cut": 0.85,
    "maintenance": 1.0,
    "bulk": 1.10,
}


def compute_recommendation_targets(
    profile: Profile,
    macro_split: MacroSplit,
    target_calories: Optional[int] = None,
    goal: Optional[Goal] = Goal.maintenance,
) -> Dict[str, int]:
    """
    Compute daily calories/macros for recommendations using the macro calculator logic.
    - Calories: TDEE +/- goal adjustment unless a target is provided.
    - Protein: ~1g per lb lean body mass (estimated if not provided).
    - Fat: percentage based on estimated body fat (clamped 15-35%) instead of fixed 20%.
    - Carbs: remainder.
    """
    # Base calories
    bmr = mifflin_st_jeor(profile)
    tdee = bmr * profile.activity_level.multiplier
    maintenance = int(round(tdee))

    goal_label = (goal.value if isinstance(goal, Goal) else goal or "maintenance").lower()
    goal_adjust = 0
    if goal_label.startswith("fat_loss") or goal_label == "lose":
        goal_adjust = -500
    elif goal_label.startswith("muscle_gain") or goal_label == "gain":
        goal_adjust = 500

    calories = target_calories or max(1200, maintenance + goal_adjust)

    # Estimate body fat % via Deurenberg
    height_m = profile.height_cm / 100.0
    bmi = profile.weight_kg / (height_m * height_m)
    sex_term = 1 if profile.sex == Sex.male else 0
    bf_pct = 1.20 * bmi + 0.23 * profile.age - 10.8 * sex_term - 5.4
    bf_pct = max(5.0, min(45.0, bf_pct))

    # Lean body mass estimate
    lbm_kg = profile.weight_kg * (1 - bf_pct / 100.0)
    weight_lb = profile.weight_kg * 2.20462
    lbm_lb = max(0.0, lbm_kg * 2.20462)

    protein_g = int(round(lbm_lb if lbm_lb > 0 else weight_lb))
    protein_g = max(60, protein_g)  # guard lower bound

    fat_percent = max(15.0, min(35.0, bf_pct))
    fat_g = (calories * fat_percent / 100.0) / 9.0

    carbs_g = (calories - (protein_g * 4) - (fat_g * 9)) / 4.0
    carbs_g = max(0, carbs_g)

    return {
        "calories": int(calories),
        "protein_g": int(round(protein_g)),
        "carbs_g": int(round(carbs_g)),
        "fat_g": int(round(fat_g)),
    }

def _ollama_generate(text: str, model: str) -> Optional[str]:
    """Call a local Ollama model. Returns raw response text or None on error."""
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    # Normalize URL to include the generate endpoint.
    if "/api/generate" not in url:
        url = url.rstrip("/") + "/api/generate"
    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": text, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        global _OLLAMA_LAST_ERROR
        _OLLAMA_LAST_ERROR = None
        raw = data.get("response") or ""
        _log_model_output(f"ollama:{model}", raw)
        return raw if raw else None
    except Exception as exc:
        _OLLAMA_LAST_ERROR = f"Ollama request failed: {exc}"
        return None


def _get_last_ollama_error() -> Optional[str]:
    return _OLLAMA_LAST_ERROR


def _parse_json_block(raw: str) -> Optional[dict]:
    """Attempt to parse JSON; tolerate leading/trailing prose or code fences."""
    try:
        sanitized = _sanitize_jsonish(raw)
        return json.loads(sanitized)
    except Exception:
        pass
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            return json.loads(_sanitize_jsonish(snippet))
        except Exception:
            return None
    return None


def _sanitize_jsonish(text: str) -> str:
    """
    Best-effort cleanup for near-JSON (e.g., trailing commas).
    This is deliberately light-touch to avoid corrupting well-formed JSON.
    """
    # Strip JS-style comments
    text = re.sub(r"//.*", "", text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Quote numeric-with-unit values like 20g or 1lb when used as values (e.g., "quantity": 20g)
    text = re.sub(r'("quantity"\s*:\s*)(\d+(?:\.\d+)?)([a-zA-Z]+)', r'\1"\2\3"', text)
    text = re.sub(r'("cost"\s*:\s*)(\d+(?:\.\d+)?)([a-zA-Z]+)', r'\1\2', text)

    # Quote unquoted step strings (e.g., steps: [Do this., Do that.])
    def _fix_steps(match: re.Match) -> str:
        body = match.group(1)
        items = []
        for line in body.splitlines():
            stripped = line.strip().rstrip(",")
            if not stripped:
                continue
            if stripped.startswith('"'):
                items.append(stripped.strip(","))
            else:
                items.append(json.dumps(stripped))
        return '"steps": [\n  ' + ",\n  ".join(items) + "\n]"

    text = re.sub(r'"steps"\s*:\s*\[(.*?)\]', _fix_steps, text, flags=re.DOTALL)
    return text


def _extract_shopping_items(data: dict) -> List[ShoppingItem]:
    """
    Parse shopping list items from model JSON with proper quantity/unit parsing.
    """
    if not isinstance(data, dict):
        return []
        
    items: List[ShoppingItem] = []
    raw_list = data.get("shopping_list") or data.get("ingredients") or []
    
    if isinstance(raw_list, list):
        for entry in raw_list:
            if not isinstance(entry, dict):
                continue
                
            name = entry.get("name")
            if not name:
                continue
                
            raw_qty = entry.get("quantity")
            raw_unit = entry.get("unit")
            raw_cost = entry.get("cost")
            
            # Parse quantity with unit
            qty_val, parsed_unit = _parse_quantity_with_unit(raw_qty)
            
            # Use separate unit field if no unit in quantity
            if not parsed_unit and raw_unit:
                parsed_unit = str(raw_unit).strip()
            
            # Parse cost
            cost_val = _coerce_number(raw_cost)
            
            items.append(ShoppingItem(
                name=str(name),
                quantity=qty_val,
                unit=parsed_unit or None,
                cost=cost_val
            ))
    
    # Fallback: build from dish ingredients if no shopping list
    if not items and "dishes" in data and isinstance(data["dishes"], list):
        items = _build_shopping_from_dishes_ingredients(data["dishes"])
    
    return items


def _build_recommendation_prompt(
    req: RecommendRequest,
    ingredients_txt: str,
    allergies_txt: str,
    profile_txt: str,
    targets_txt: str,
    budget_txt: str,
    budget_per_day: str,
    rag_ctx: str,
) -> str:
    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = req.meals_per_day * days
    per_meal_cal = None
    per_meal_protein = None
    per_meal_carbs = None
    per_meal_fat = None
    
    if req.profile:
        targets = compute_recommendation_targets(req.profile, req.macro_split, req.target_calories, req.goal)
        per_meal_cal = max(200, int(targets["calories"] / req.meals_per_day))
        per_meal_protein = max(10, int(targets["protein_g"] / req.meals_per_day))
        per_meal_carbs = max(10, int(targets["carbs_g"] / req.meals_per_day))
        per_meal_fat = max(5, int(targets["fat_g"] / req.meals_per_day))
    
    day_prompt = (
        f"Return exactly {needed_dishes} dishes with day numbers cycling 1..{days}; ensure day 1 and day {days} are present and avoid repeating dish names."
        if days > 1
        else f"Return exactly {needed_dishes} dishes for day 1."
    )
    cuisine_txt = f"Preferred cuisine: {req.cuisine}." if req.cuisine else "Cuisine: flexible."
    
    return f"""
You are a nutrition chef. Recommend simple dishes ONLY using these ingredients: {ingredients_txt}.
Diet: {req.dietary_preference.value}. Goal: {req.goal.replace('_',' ')} (make sure dishes honor this: fat_loss => lighter calories, muscle_gain => higher protein, maintenance => balanced). Allergies: {allergies_txt}.
User profile: {profile_txt}. Targets: {targets_txt}. Budget: {budget_txt} ({budget_per_day}); favor affordable ingredients.
{cuisine_txt}

If profile is provided, aim for each dish to be roughly ~{per_meal_cal or 'target/meal'} kcal, Protein ~{per_meal_protein or 'target/meal'}g, Carbs ~{per_meal_carbs or 'target/meal'}g, Fat ~{per_meal_fat or 'target/meal'}g so totals across all dishes stay close to the daily targets (within ~10%).

Use this nutrition context (prioritize these ingredients/macros if relevant):
{rag_ctx if rag_ctx else "No context provided; rely on general nutrition knowledge."}

Return JSON with this EXACT format:
{{
  "dishes": [
    {{
      "day": 1,
      "name": "Spinach Feta Omelette",
      "reason": "High protein breakfast using available eggs and spinach",
      "ingredients_used": ["eggs", "spinach", "feta cheese", "olive oil"],
      "ingredients": [
        {{
          "name": "eggs",
          "quantity": "2",
          "unit": "count",
          "cost": 0.5
        }},
        {{
          "name": "olive oil",
          "quantity": "1",
          "unit": "tbsp",
          "cost": 0.3
        }},
        {{
          "name": "spinach",
          "quantity": "50",
          "unit": "g",
          "cost": 0.8
        }},
        {{
          "name": "feta cheese",
          "quantity": "30",
          "unit": "g",
          "cost": 1.2
        }}
      ],
      "steps": [
        "Beat eggs in a bowl with salt and pepper.",
        "Heat olive oil in a non-stick pan over medium heat.",
        "Add spinach and cook until wilted.",
        "Pour in beaten eggs and cook for 2-3 minutes.",
        "Add crumbled feta cheese on one half.",
        "Fold omelette in half and cook for another minute.",
        "Slide onto plate and serve hot."
      ],
      "calories": 420,
      "protein_g": 28,
      "carbs_g": 8,
      "fat_g": 32,
      "estimated_cost": 2.8
    }}
  ],
  "shopping_list": [
    {{
      "name": "eggs",
      "quantity": "12",
      "unit": "count",
      "cost": 3.5
    }},
    {{
      "name": "spinach",
      "quantity": "200",
      "unit": "g",
      "cost": 2.5
    }},
    {{
      "name": "feta cheese",
      "quantity": "100",
      "unit": "g",
      "cost": 4.0
    }},
    {{
      "name": "olive oil",
      "quantity": "1",
      "unit": "bottle",
      "cost": 5.5
    }}
  ]
}}

CRITICAL FORMATTING RULES:
1. Each ingredient MUST have: name, quantity (as STRING number), unit, and cost
2. quantity should be a numeric string like "2", "100", "1.5"
3. unit must be one of: g, kg, oz, lb, ml, l, cup, tbsp, tsp, count, piece, bottle
4. For metric weights use: g or kg (e.g., "100" with unit "g")
5. For volumes use: ml, l, cup, tbsp, tsp
6. For countable items use: count or piece
7. cost should be a number (not string)
8. estimated_cost for each dish = sum of all ingredient costs for that dish
9. Steps must be an array of 3-8 clear, actionable strings
10. All numeric fields (calories, protein_g, carbs_g, fat_g) must be integers
11. Never leave any field null - always provide reasonable estimates

Shopping list rules:
- Aggregate quantities across all dishes
- Each item MUST include numeric quantity, unit, and estimated total cost
- Use reasonable package sizes (e.g., eggs by dozen, meat by lb/kg)

Days: {days}. Give specific dish names—no generic "Morning/Lunch/Evening".
Return exactly {needed_dishes} dishes total ({req.meals_per_day} per day), using day values 1..{days}.

Respond with ONLY valid JSON, no markdown code fences, no extra text before or after.
"""


def _build_image_recipe_prompt(caption: str, rag_ctx: str) -> str:
    return f"""
You are a chef. Given this food description: "{caption}" produce a recipe as JSON (no markdown) with fields:
{{
  "name": "Dish name",
  "summary": "One sentence summary",
  "ingredients": [
    {{
      "name": "eggs",
      "quantity": "2",
      "unit": "count",
      "cost": 0.5
    }}
  ],
  "steps": ["Step 1", "Step 2", "Step 3"],
  "calories": 500,
  "protein_g": 30,
  "carbs_g": 50,
  "fat_g": 18
}}
Nutrition context (use if relevant): {rag_ctx if rag_ctx else "none"}.
Rules: return JSON only; include macros and calories; 3-8 concise steps; avoid generic names; strictly valid JSON (no trailing commas, no markdown).
Each ingredient must have quantity as string number, unit, and cost.
"""


def _build_refine_prompt(
    req: RecommendRequest,
    feedback: str,
    dishes: List[DishRecommendation],
    targets_txt: str,
    rag_ctx: str,
) -> str:
    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = req.meals_per_day * days
    dishes_json = json.dumps({"dishes": [d.dict() for d in dishes]}, ensure_ascii=False)
    ingredients_txt = ", ".join(req.ingredients) if req.ingredients else "use pantry staples that fit the diet"
    allergies_txt = ", ".join(req.allergies) if req.allergies else "none"
    cuisine_txt = f"Preferred cuisine: {req.cuisine}." if req.cuisine else "Cuisine: flexible."
    return f"""
You already proposed these dishes (JSON): {dishes_json}
User feedback: "{feedback}". Adjust recommendations to honor this feedback while keeping macros/goal and avoiding repeats.
Diet: {req.dietary_preference.value}. Goal: {req.goal.replace('_',' ')}. Allergies: {allergies_txt}. {cuisine_txt}
Ingredients to prioritize: {ingredients_txt}. Keep affordable and simple.
Targets: {targets_txt if targets_txt else 'keep balanced macros and calories close to targets'}.
Days: {days}, meals per day: {req.meals_per_day}. Return the same number of dishes ({needed_dishes}) with day values 1..{days} and {req.meals_per_day} per day.
Use nutrition context if helpful: {rag_ctx if rag_ctx else 'none'}.
Return JSON only in the same shape as before with proper ingredient formatting:
{{
  "dishes": [
    {{
      "day": 1,
      "name": "...",
      "reason": "...",
      "ingredients_used": ["..."],
      "ingredients": [
        {{
          "name": "chicken breast",
          "quantity": "200",
          "unit": "g",
          "cost": 3.5
        }}
      ],
      "steps": ["..."],
      "calories": 500,
      "protein_g": 40,
      "carbs_g": 50,
      "fat_g": 15,
      "estimated_cost": 5.5
    }}
  ],
  "shopping_list": [
    {{
      "name": "salmon",
      "quantity": "500",
      "unit": "g",
      "cost": 8.5
    }}
  ]
}}
Shopping list rules: each item MUST include quantity as string number, unit from [g, kg, oz, lb, ml, l, cup, tbsp, tsp, piece, count], and cost.
No markdown, no extra text, strictly valid JSON, 3-8 concise steps per dish.
"""


def _normalize_steps(raw_steps: List[str]) -> List[str]:
    """
    Flatten and split steps when the model returns multiple sentences/numbered steps in a single entry.
    Keeps ordering while splitting on newlines, numbered prefixes, and sentence boundaries.
    """
    steps: List[str] = []
    for entry in raw_steps:
        text = " ".join(str(entry).replace("\n", " ").split())
        if not text:
            continue
        text = re.sub(r"\s*\d+\.\s+", "\n", text)
        text = text.replace("; ", "\n")
        parts = re.split(r"\n|(?<=[.!?])\s+(?=[A-Z])", text)
        for part in parts:
            clean = part.strip(" -")
            if clean:
                steps.append(clean)
    cleaned: List[str] = []
    for s in steps:
        txt = re.sub(r"[\s.;,:-]+$", "", s).strip()
        if not txt:
            continue
        if txt[-1] not in ".!?":
            txt = f"{txt}."
        cleaned.append(txt)
    return cleaned[:8] if cleaned else raw_steps


def _validate_and_normalize_dishes(data: dict, req: RecommendRequest) -> Tuple[List[DishRecommendation], Optional[str]]:
    """
    Validate the JSON structure from the model and normalize dishes.
    Updated to properly parse ingredient quantities with units.
    """
    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = req.meals_per_day * days
    raw_list = data.get("dishes")
    if not isinstance(raw_list, list) or not raw_list:
        return [], "no dishes in response"

    dishes: List[DishRecommendation] = []
    seen_names: Dict[str, int] = {}
    
    for d in raw_list:
        day_val = int(d.get("day", 1))
        if day_val < 1 or day_val > days:
            continue
            
        name = (d.get("name") or "Dish").strip() or "Dish"
        count = seen_names.get(name.lower(), 0) + 1
        seen_names[name.lower()] = count
        if count > 1:
            name = f"{name} #{count}"
            
        raw_steps = _normalize_steps([str(s) for s in d.get("steps", []) if s])
        estimated_cost = _coerce_number(d.get("estimated_cost"))
        
        # Parse ingredients with proper quantity/unit handling
        ingredient_costs: Optional[List[IngredientCost]] = None
        ing_list = d.get("ingredients")
        if isinstance(ing_list, list):
            parsed: List[IngredientCost] = []
            for ing in ing_list:
                if not isinstance(ing, dict):
                    continue
                parsed.append(_parse_ingredient_from_dict(ing))
            
            ingredient_costs = parsed if parsed else None
            
            # Calculate estimated_cost from ingredients if not provided
            if estimated_cost is None and ingredient_costs:
                estimated_cost = sum(ic.cost or 0 for ic in ingredient_costs if ic.cost is not None)
                estimated_cost = round(estimated_cost, 2) if estimated_cost else None

        dishes.append(
            DishRecommendation(
                day=day_val,
                name=name,
                reason=d.get("reason", ""),
                ingredients_used=[str(i) for i in d.get("ingredients_used", []) if i],
                ingredients=ingredient_costs,
                steps=raw_steps,
                calories=_coerce_number(d.get("calories")),
                protein_g=_coerce_number(d.get("protein_g")),
                carbs_g=_coerce_number(d.get("carbs_g")),
                fat_g=_coerce_number(d.get("fat_g")),
                estimated_cost=estimated_cost,
            )
        )

    if not dishes:
        return [], "no valid dishes after validation"

    # Validate day coverage
    days_present = {d.day for d in dishes}
    if days > 1 and (1 not in days_present or days not in days_present):
        return [], f"missing day coverage (need day 1 and day {days})"

    # Ensure we have the right number of dishes
    if len(dishes) < needed_dishes:
        base = dishes[:]
        while len(dishes) < needed_dishes:
            src = base[len(dishes) % len(base)]
            dishes.append(
                DishRecommendation(
                    day=(len(dishes) // req.meals_per_day) + 1,
                    name=src.name,
                    reason=src.reason,
                    ingredients_used=src.ingredients_used,
                    ingredients=src.ingredients,
                    steps=src.steps,
                    calories=src.calories,
                    protein_g=src.protein_g,
                    carbs_g=src.carbs_g,
                    fat_g=src.fat_g,
                    estimated_cost=src.estimated_cost,
                )
            )
    if len(dishes) > needed_dishes:
        dishes = dishes[:needed_dishes]
        
    # Ensure correct day numbering
    for idx, dish in enumerate(dishes):
        dish.day = (idx // req.meals_per_day) + 1
        
    return dishes, None


def _default_hf_model(req_model: Optional[str] = None) -> Optional[str]:
    """
    Choose an HF model in priority order:
    1) Request override
    2) First entry in config/models.json[huggingface]
    3) HF_MODEL_ID env (legacy/default)
    """
    if req_model:
        return req_model
    catalog = _load_model_catalog()
    if catalog.get("huggingface"):
        return catalog["huggingface"][0]
    env_model = os.getenv("HF_MODEL_ID")
    if env_model:
        return env_model
    return None


def _default_openai_model(req_model: Optional[str] = None) -> Optional[str]:
    """
    Choose an OpenAI model in priority order:
    1) Request override
    2) First entry in config/models.json[openai]
    3) OPENAI_MODEL env (or default gpt-4o-mini)
    """
    if req_model:
        return req_model
    catalog = _load_model_catalog()
    if catalog.get("openai"):
        return catalog["openai"][0]
    return os.getenv("OPENAI_MODEL") or "gpt-4o-mini"


def _recipe_model_name(override: Optional[str] = None) -> Optional[str]:
    return override or os.getenv("OLLAMA_RECIPE_MODEL") or _default_ollama_model(None)


def _default_ollama_model(req_model: Optional[str] = None) -> Optional[str]:
    """
    Choose an Ollama model in priority order:
    1) Request override
    2) First entry in config/models.json[ollama]
    3) OLLAMA_MODEL env
    """
    if req_model:
        return req_model
    catalog = _load_model_catalog()
    if catalog.get("ollama"):
        return catalog["ollama"][0]
    env_model = os.getenv("OLLAMA_MODEL")
    if env_model:
        return env_model
    return None


def _generate_recipe_from_caption(
    caption: str, hf_model_id: Optional[str], ollama_model: Optional[str], openai_model: Optional[str] = None
) -> Tuple[Optional[dict], str, Optional[str]]:
    """
    Use selected model to turn an image caption into a recipe JSON with macros.
    Returns (data dict, provider label, error string).
    """
    rag_ctx = ""
    rag_service: Optional[RagService] = getattr(app.state, "rag", None)  # type: ignore[attr-defined]
    if rag_service:
        rag_ctx, _ = rag_service.search_context(f"food description: {caption}", k=4)

    prompt = _build_image_recipe_prompt(caption, rag_ctx)

    # OpenAI default if selected or no other override.
    if openai_model or (not hf_model_id and not ollama_model):
        client = OpenAIRecipeClient(model_id=_default_openai_model(openai_model))
        raw, err = client.generate(prompt)
        if raw:
            data = _parse_json_block(_sanitize_jsonish(raw))
            if data:
                return data, f"openai:{client.model_id}", None
        return None, "", err or "OpenAI generation failed"

    # If Ollama requested, try it first.
    if ollama_model:
        raw = _ollama_generate(prompt, model=_recipe_model_name(ollama_model) or ollama_model)
        if raw:
            data = _parse_json_block(_sanitize_jsonish(raw))
            if data:
                return data, f"ollama:{ollama_model}", None
        return None, "", "Ollama generation failed or invalid JSON"

    # Otherwise try HF (default/hf_model_id).
    model_id = _default_hf_model(hf_model_id)
    raw_list = generate_hf_recipes([prompt], model_id=model_id, **_hf_generation_settings())
    if raw_list:
        raw_text = raw_list[0]
        _log_model_output(f"huggingface:{model_id}", raw_text)
        data = _parse_json_block(_sanitize_jsonish(raw_text))
        if data:
            return data, f"huggingface:{model_id}", None
        return None, "", "HF parse failed"

    return None, "", get_last_hf_error() or "HF generation failed"


def _fallback_recommendations(req: RecommendRequest) -> List[DishRecommendation]:
    """
    Deterministic, allergy-aware suggestions when both Hugging Face and Ollama are unavailable.
    Keeps the endpoint responsive for demo purposes.
    """
    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = days * req.meals_per_day
    staples = {
        DietaryPreference.omnivore: ["chicken", "rice", "beans", "eggs", "spinach"],
        DietaryPreference.vegetarian: ["tofu", "quinoa", "beans", "eggs", "spinach"],
        DietaryPreference.vegan: ["tofu", "quinoa", "beans", "lentils", "spinach"],
        DietaryPreference.pescatarian: ["salmon", "rice", "beans", "spinach", "eggs"],
        DietaryPreference.keto: ["chicken", "cauliflower", "eggs", "avocado", "spinach"],
        DietaryPreference.paleo: ["chicken", "sweet potato", "eggs", "spinach", "nuts"],
    }

    def _safe(x: str) -> bool:
        return not any(a.lower() in x.lower() for a in req.allergies)

    pantry = [i for i in req.ingredients if _safe(i)] or [
        i for i in staples.get(req.dietary_preference, staples[DietaryPreference.omnivore]) if _safe(i)
    ]
    styles = ["bowl", "skillet", "salad", "wrap", "scramble", "soup", "bake"]

    dishes: List[DishRecommendation] = []
    for idx in range(needed_dishes):
        ing = pantry[idx % len(pantry)]
        style = styles[idx % len(styles)]
        dishes.append(
            DishRecommendation(
                day=(idx // req.meals_per_day) + 1,
                name=f"{ing.title()} {style.title()}",
                reason=f"Uses available ingredient '{ing}' and fits {req.dietary_preference.value} diet.",
                ingredients_used=[ing],
                steps=[
                    f"Prep {ing} with salt, pepper, and any preferred herbs.",
                    f"Cook the {ing} in a pan or oven until done.",
                    "Add a quick veggie side or greens if available.",
                    "Serve immediately; pack leftovers for the next meal.",
                ],
                calories=None,
                protein_g=None,
                carbs_g=None,
                fat_g=None,
            )
        )
    return dishes


def _hf_generation_settings() -> dict:
    """
    Default generation knobs aligned with the model card for flax-community/t5-recipe-generation.
    Keep these modest to avoid runaway responses when running on CPU.
    """
    return {
        "max_length": 512,
        "min_length": 64,
        "no_repeat_ngram_size": 3,
        "do_sample": True,
        "top_k": 60,
        "top_p": 0.95,
    }


def _hf_clean_recipe_text(text: str) -> str:
    """
    Apply the token replacements described in the model card to produce readable text.
    """
    replacements = {"<sep>": "--", "<section>": "\n", "</s>": "", "<pad>": ""}
    cleaned = text
    for token, val in replacements.items():
        cleaned = cleaned.replace(token, val)
    return cleaned.strip()


def _parse_hf_recipe(text: str) -> Tuple[str, List[str], List[str]]:
    """
    Parse the text output from the HF recipe model into title, ingredients, and steps.
    Ingredients and steps are delimited by '--' after cleaning.
    """
    cleaned = _hf_clean_recipe_text(text)
    title = "Recipe"
    ingredients: List[str] = []
    steps: List[str] = []
    for line in cleaned.splitlines():
        if not line.strip():
            continue
        lowered = line.lower()
        if lowered.startswith("title:"):
            title = line.split(":", 1)[1].strip().title() or title
        elif lowered.startswith("ingredients:"):
            raw = line.split(":", 1)[1]
            ingredients.extend([p.strip(" -") for p in raw.split("--") if p.strip()])
        elif lowered.startswith("directions:") or lowered.startswith("steps:"):
            raw = line.split(":", 1)[1]
            steps.extend([p.strip(" -") for p in raw.split("--") if p.strip()])
    # Fallback: if steps were not captured but the text has '--', treat them as steps.
    if not steps and "--" in cleaned:
        steps = [p.strip(" -") for p in cleaned.split("--") if p.strip()]
    return title, ingredients, steps


def _ingredient_pool(req: RecommendRequest) -> List[str]:
    """
    Build a pool of safe ingredients honoring dietary preference and allergies.
    """
    staples = {
        DietaryPreference.omnivore: ["chicken", "rice", "beans", "eggs", "spinach"],
        DietaryPreference.vegetarian: ["tofu", "quinoa", "beans", "eggs", "spinach"],
        DietaryPreference.vegan: ["tofu", "quinoa", "beans", "lentils", "spinach"],
        DietaryPreference.pescatarian: ["salmon", "rice", "beans", "spinach", "eggs"],
        DietaryPreference.keto: ["chicken", "cauliflower", "eggs", "avocado", "spinach"],
        DietaryPreference.paleo: ["chicken", "sweet potato", "eggs", "spinach", "nuts"],
    }

    def _safe(x: str) -> bool:
        return not any(a.lower() in x.lower() for a in req.allergies)

    pool = [i for i in req.ingredients if _safe(i)]
    if pool:
        return pool
    return [i for i in staples.get(req.dietary_preference, staples[DietaryPreference.omnivore]) if _safe(i)]


def _run_prompt_with_hf(
    req: RecommendRequest, prompt: str
) -> Tuple[Optional[List[DishRecommendation]], Optional[int], Optional[str], Optional[List[ShoppingItem]], Optional[str]]:
    model_id = _default_hf_model(req.hf_model_id)
    if not model_id:
        return None, None, None, None, "HF_MODEL_ID not set"
    raw = generate_hf_recipes([prompt], model_id=model_id, **_hf_generation_settings())
    if not raw:
        return None, None, model_id, None, get_last_hf_error() or "generation returned empty"
    raw_text = raw[0]
    _log_model_output(f"huggingface:{model_id}", raw_text)
    data = _parse_json_block(_sanitize_jsonish(raw_text))
    shopping_list = _extract_shopping_items(data) if data else []
    if not data:
        return None, None, model_id, shopping_list, "HF parse failed"
    dishes, validation_err = _validate_and_normalize_dishes(data, req)
    if not dishes:
        return None, None, model_id, shopping_list, validation_err or "HF returned no dishes"
    _, _, _, param_count = get_hf_pipeline(model_id_override=model_id)
    return dishes, param_count, model_id, shopping_list, None


def _run_prompt_with_ollama(
    req: RecommendRequest, prompt: str
) -> Tuple[Optional[List[DishRecommendation]], Optional[str], Optional[List[ShoppingItem]], Optional[str]]:
    model = _recipe_model_name(req.ollama_model)
    if not model:
        return None, None, None, "OLLAMA_MODEL not set"
    raw = _ollama_generate(prompt, model=model)
    if not raw:
        return None, None, None, _get_last_ollama_error() or "Ollama call failed"
    cleaned = raw.replace("```json", "").replace("```", "")
    data = _parse_json_block(_sanitize_jsonish(cleaned))
    if not data:
        return None, None, None, "Ollama parse failed"
    shopping_list = _extract_shopping_items(data)
    dishes, validation_err = _validate_and_normalize_dishes(data, req)
    if not dishes:
        return None, None, shopping_list, validation_err or "Ollama returned no dishes"
    return dishes, f"ollama:{model}", shopping_list, None


def _generate_recommendations_with_openai(
    req: RecommendRequest, prompt: str
) -> Tuple[Optional[List[DishRecommendation]], str, Optional[int], Optional[List[ShoppingItem]], Optional[str]]:
    model_id = _default_openai_model(req.openai_model)
    client = OpenAIRecipeClient(model_id=model_id)
    raw, err = client.generate(prompt)
    if not raw:
        return None, '', None, None, err or 'OpenAI generation failed'
    data = _parse_json_block(_sanitize_jsonish(raw))
    if not data:
        return None, '', None, None, 'OpenAI parse failed'
    shopping_list = _extract_shopping_items(data)
    dishes, validation_err = _validate_and_normalize_dishes(data, req)
    if not dishes:
        return None, '', None, shopping_list, validation_err or 'OpenAI returned no dishes'
    return dishes, f"openai:{model_id}", None, shopping_list, None


def generate_recommendations_with_llm(
    req: RecommendRequest,
) -> Tuple[Optional[List[DishRecommendation]], str, Optional[int], Optional[List[ShoppingItem]], Optional[str]]:
    """
    Honor user selection strictly: use the chosen model end-to-end.
    If none chosen, try OpenAI (default), then HF default, then Ollama default.
    """
    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = req.meals_per_day * days
    ingredients_txt = ", ".join(req.ingredients) if req.ingredients else "none specified—use pantry-friendly staples that fit the diet"
    allergies_txt = ", ".join(req.allergies) if req.allergies else "none"
    budget_txt = f"${req.budget_amount:.2f} total" if req.budget_amount else "no budget provided"
    budget_per_day = (
        f"~${(req.budget_amount or 0)/days:.2f} per day" if req.budget_amount else "flexible per-day budget"
    )
    profile_txt = "no profile provided"
    targets_txt = "keep meals balanced and macro-aware"
    if req.profile:
        targets = compute_recommendation_targets(req.profile, req.macro_split, req.target_calories, req.goal)
        profile_txt = (
            f"{req.profile.sex}, {req.profile.age}y, {req.profile.height_cm}cm, "
            f"{req.profile.weight_kg}kg, activity {req.profile.activity_level.value}"
        )
        targets_txt = (
            f"calories ~{targets['calories']} kcal/day, macros P:{targets['protein_g']}g "
            f"C:{targets['carbs_g']}g F:{targets['fat_g']}g (from split "
            f"{req.macro_split.protein_pct}/{req.macro_split.carbs_pct}/{req.macro_split.fat_pct})"
        )

    rag_ctx = ''
    rag_service: Optional[RagService] = getattr(app.state, "rag", None)  # type: ignore[attr-defined]
    if rag_service:
        query = (
            f"Ingredients: {ingredients_txt}. Diet: {req.dietary_preference.value}. "
            f"Goal: {req.goal}. Profile: {profile_txt}. Targets: {targets_txt}."
        )
        rag_ctx, _ = rag_service.search_context(query, k=6)

    prompt = _build_recommendation_prompt(
        req, ingredients_txt, allergies_txt, profile_txt, targets_txt, budget_txt, budget_per_day, rag_ctx
    )

    # Strict selection: only the chosen model runs.
    if req.openai_model:
        return _generate_recommendations_with_openai(req, prompt)
    if req.hf_model_id:
        hf_dishes, hf_params, hf_model_id, hf_shopping, hf_error = _run_prompt_with_hf(req, prompt)
        if hf_dishes:
            return hf_dishes, f"huggingface:{hf_model_id}", hf_params, hf_shopping, None
        return None, '', None, hf_shopping, hf_error or 'HF generation failed'
    if req.ollama_model:
        ollama_dishes, provider, shopping, err = _run_prompt_with_ollama(req, prompt)
        if ollama_dishes:
            return ollama_dishes, provider or f"ollama:{req.ollama_model}", None, shopping, None
        return None, '', None, shopping, err or 'Ollama call failed'

    # Defaults: OpenAI -> HF default -> Ollama default.
    oa_dishes, oa_provider, _, oa_shopping, oa_err = _generate_recommendations_with_openai(req, prompt)
    if oa_dishes:
        return oa_dishes, oa_provider, None, oa_shopping, None

    hf_dishes, hf_params, hf_model_id, hf_shopping, hf_error = _run_prompt_with_hf(req, prompt)
    if hf_dishes:
        return hf_dishes, f"huggingface:{hf_model_id}", hf_params, hf_shopping, None

    ollama_dishes, provider, shopping, ollama_err = _run_prompt_with_ollama(req, prompt)
    if ollama_dishes:
        return ollama_dishes, provider or "", None, shopping, None
    return None, '', None, shopping, oa_err or hf_error or ollama_err or 'No model configured'


def refine_recommendations_with_llm(
    feedback: str, dishes: List[DishRecommendation], req: RecommendRequest
) -> Tuple[Optional[List[DishRecommendation]], str, Optional[int], Optional[List[ShoppingItem]], Optional[str]]:
    profile_txt = "no profile provided"
    targets_txt = "keep meals balanced and macro-aware"
    if req.profile:
        targets = compute_recommendation_targets(req.profile, req.macro_split, req.target_calories, req.goal)
        profile_txt = (
            f"{req.profile.sex}, {req.profile.age}y, {req.profile.height_cm}cm, "
            f"{req.profile.weight_kg}kg, activity {req.profile.activity_level.value}"
        )
        targets_txt = (
            f"calories ~{targets['calories']} kcal/day, macros P:{targets['protein_g']}g "
            f"C:{targets['carbs_g']}g F:{targets['fat_g']}g (split "
            f"{req.macro_split.protein_pct}/{req.macro_split.carbs_pct}/{req.macro_split.fat_pct})"
        )

    rag_ctx = ""
    rag_service: Optional[RagService] = getattr(app.state, "rag", None)  # type: ignore[attr-defined]
    if rag_service:
        query = (
            f"Refine dishes with feedback: {feedback}. Diet: {req.dietary_preference.value}. "
            f"Goal: {req.goal}. Profile: {profile_txt}. Targets: {targets_txt}."
        )
        rag_ctx, _ = rag_service.search_context(query, k=6)

    prompt = _build_refine_prompt(req, feedback, dishes, targets_txt, rag_ctx)

    if req.openai_model:
        oa_dishes, provider, _, shopping, err = _generate_recommendations_with_openai(req, prompt)
        if oa_dishes:
            return oa_dishes, provider, None, shopping, None
        return None, "", None, shopping, err
    if req.hf_model_id:
        hf_dishes, params, model_id, shopping, err = _run_prompt_with_hf(req, prompt)
        if hf_dishes:
            return hf_dishes, f"huggingface:{model_id}", params, shopping, None
        return None, "", None, shopping, err
    if req.ollama_model:
        ollama_dishes, provider, shopping, err = _run_prompt_with_ollama(req, prompt)
        if ollama_dishes:
            return ollama_dishes, provider or "", None, shopping, None
        return None, "", None, shopping, err

    shopping = shopping_hf = shopping_ollama = None
    oa_dishes, provider, _, shopping, oa_err = _generate_recommendations_with_openai(req, prompt)
    if oa_dishes:
        return oa_dishes, provider, None, shopping, None
    hf_dishes, params, model_id, shopping_hf, hf_err = _run_prompt_with_hf(req, prompt)
    if hf_dishes:
        return hf_dishes, f"huggingface:{model_id}", params, shopping_hf, None
    ollama_dishes, provider, shopping_ollama, ollama_err = _run_prompt_with_ollama(req, prompt)
    if ollama_dishes:
        return ollama_dishes, provider or "", None, shopping_ollama, None
    return None, "", None, shopping or shopping_hf or shopping_ollama, oa_err or hf_err or ollama_err or "No model configured for refine"


def generate_plan_with_llm(req: PlanRequest, targets: dict) -> Optional[List[DayPlan]]:
    """
    Attempt to create a plan via a local LLM (ollama). Expects JSON output matching the DayPlan schema.
    Returns None on any failure; caller raises.
    """
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return None

    days = 7
    prompt = f"""
You are a nutrition planner. Given calories and macro targets, dietary preferences, allergies, meals per day,
and budget period, produce a JSON object with a list of days. Each day has: day (int), meals (list of meals),
and totals (total_calories, total_protein_g, total_carbs_g, total_fat_g).
Each meal has: name, calories, protein_g, carbs_g, fat_g, and ingredients (list of objects with name, quantity).
Use simple grocery ingredients (not recipes) and keep ingredient names terse to match catalog pricing later.

Targets:
- calories: {targets['calories']}
- protein_g: {targets['protein_g']}
- carbs_g: {targets['carbs_g']}
- fat_g: {targets['fat_g']}

Constraints:
- dietary preference: {req.dietary_preference.value}
- allergies: {', '.join(req.allergies) if req.allergies else 'none'}
- meals per day: {req.meals_per_day}
- days: {days}
- keep ingredient names simple (e.g., "chicken breast", "rice", "broccoli")
- each day must have exactly {req.meals_per_day} meals; add or trim to hit this count
- respond with JSON only, no markdown code fences or extra text

Return only JSON like:
{{
  "days": [
    {{
      "day": 1,
      "meals": [
        {{
          "name": "Breakfast",
          "calories": 500,
          "protein_g": 35,
          "carbs_g": 50,
          "fat_g": 15,
          "ingredients": [{{"name": "oats", "quantity": "1 cup"}}, ...]
        }}
      ],
      "total_calories": 2000,
      "total_protein_g": 140,
      "total_carbs_g": 180,
      "total_fat_g": 70
    }}
  ]
}}
"""
    raw = _ollama_generate(prompt, model=model)
    if not raw:
        return None
    try:
        data = _parse_json_block(raw)
        if not data:
            return None
        parsed_days: List[DayPlan] = []
        for day in data.get("days", []):
            meals: List[Meal] = []
            for meal in day.get("meals", []):
                ing_objs = [
                    Ingredient(name=i.get("name", ""), quantity=i.get("quantity", "1"))
                    for i in meal.get("ingredients", [])
                    if i.get("name")
                ]
                meals.append(
                    Meal(
                        name=meal.get("name", "Meal"),
                        calories=int(meal.get("calories", 0)),
                        protein_g=int(meal.get("protein_g", 0)),
                        carbs_g=int(meal.get("carbs_g", 0)),
                        fat_g=int(meal.get("fat_g", 0)),
                        ingredients=ing_objs,
                    )
                )
            if len(meals) < req.meals_per_day:
                return None
            if len(meals) > req.meals_per_day:
                meals = meals[: req.meals_per_day]
            total_calories = sum(m.calories for m in meals)
            total_protein_g = sum(m.protein_g for m in meals)
            total_carbs_g = sum(m.carbs_g for m in meals)
            total_fat_g = sum(m.fat_g for m in meals)
            parsed_days.append(
                DayPlan(
                    day=int(day.get("day", 0)),
                    meals=meals,
                    total_calories=total_calories,
                    total_protein_g=total_protein_g,
                    total_carbs_g=total_carbs_g,
                    total_fat_g=total_fat_g,
                )
            )
        return parsed_days if parsed_days else None
    except Exception:
        return None


def load_dummy_catalog() -> Dict[str, float]:
    """
    Load a simple item->price map from data/dummy_catalog.csv (name,price).
    Case-insensitive exact matches only; extend as needed.
    """
    catalog_path = ROOT_DIR / "data" / "dummy_catalog.csv"
    if not catalog_path.exists():
        return {}

    catalog: Dict[str, float] = {}
    with open(catalog_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("name", "").strip().lower()
            price_str = row.get("price", "").strip()
            if not name or not price_str:
                continue
            try:
                catalog[name] = float(price_str)
            except ValueError:
                continue
    return catalog


DUMMY_CATALOG = load_dummy_catalog()


def estimate_plan_cost(days: List[DayPlan]) -> Tuple[float, List[str]]:
    """
    Estimate total cost of a plan using dummy catalog.
    Each ingredient counts as one unit; quantity text is ignored in this placeholder.
    Returns total and list of ingredients not found.
    """
    total = 0.0
    missing: set[str] = set()
    for day in days:
        for meal in day.meals:
            for ingredient in meal.ingredients:
                key = ingredient.name.strip().lower()
                if key in DUMMY_CATALOG:
                    total += DUMMY_CATALOG[key]
                else:
                    missing.add(ingredient.name)
    return total, sorted(missing)


@app.post("/plan", response_model=PlanResponse)
def generate_plan(req: PlanRequest) -> PlanResponse:
    """
    Generate a meal plan via LLM only.
    """
    targets = compute_targets(req)
    days = generate_plan_with_llm(req, targets)
    if days is None:
        raise HTTPException(
            status_code=503,
            detail="LLM unavailable or returned invalid plan. Ensure OLLAMA_MODEL is configured and reachable.",
        )

    estimated_cost, unknown = estimate_plan_cost(days)

    notes = [
        "Plan generated by LLM (OLLAMA_MODEL).",
        "When product search is available, map each ingredient to real products and prices.",
        "Cost uses data/dummy_catalog.csv; quantities are ignored—replace with real pricing for production.",
    ]

    return PlanResponse(
        target_calories=targets["calories"],
        target_protein_g=targets["protein_g"],
        target_carbs_g=targets["carbs_g"],
        target_fat_g=targets["fat_g"],
        budget_amount=req.budget_amount,
        days=days,
        estimated_cost=round(estimated_cost, 2) if estimated_cost else None,
        unknown_items=unknown,
        notes=notes,
    )

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(req: RecommendRequest) -> RecommendationResponse:
    ingredients = [i.strip().lower() for i in req.ingredients if i.strip()]
    clean_req = RecommendRequest(**req.dict())
    clean_req.ingredients = ingredients
    target_fields = {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None}
    if clean_req.profile:
        targets = compute_recommendation_targets(clean_req.profile, clean_req.macro_split, clean_req.target_calories, clean_req.goal)
        target_fields = targets

    try:
        dishes, provider, param_count, shopping_list, err = _run_with_timeout(
            lambda: generate_recommendations_with_llm(clean_req), timeout_seconds=60
        )
    except TimeoutError:
        raise HTTPException(
            status_code=504,
            detail="LLM recommendation generation timed out after 60s; request canceled server-side.",
        )

    if dishes is None:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable or returned invalid recommendations; deterministic fallback disabled. Last error: {err}",
        )

    notes: List[str] = []
    instacart_links: Optional[dict] = None
    shopping_items_payload: Optional[List[dict]] = None
    
    if shopping_list:
        shopping_items_payload = [item.dict() for item in shopping_list]
    else:
        items_from_dishes = _build_shopping_from_dishes(dishes)
        if items_from_dishes:
            shopping_items_payload = items_from_dishes
        elif clean_req.ingredients:
            shopping_items_payload = [{"name": ing, "quantity": 1.0, "unit": "piece", "cost": None} for ing in clean_req.ingredients]
    
    if provider:
        notes.append(f"Generated by {provider}.")
    
    if shopping_items_payload:
        try:
            instacart_links = create_shopping_list_with_search_links(
                shopping_items_payload, title="LLM Grocery List"
            )
        except Exception as exc:
            notes.append(f"Instacart link generation failed: {exc}")

    used = set()
    for d in dishes:
        used.update([u.lower() for u in d.ingredients_used])
    unknown = [i for i in ingredients if i not in used]

    total_cost = None
    if shopping_items_payload:
        total_cost = sum((item.get("cost") or 0) for item in shopping_items_payload)
        if total_cost:
            total_cost = round(total_cost, 2)

    resp = RecommendationResponse(
        period=req.period,
        dishes=dishes,
        unknown_items=unknown,
        notes=notes,
        target_calories=target_fields.get("calories"),
        target_protein_g=target_fields.get("protein_g"),
        target_carbs_g=target_fields.get("carbs_g"),
        target_fat_g=target_fields.get("fat_g"),
        cuisine=clean_req.cuisine,
        model_provider=provider,
        model_parameters=param_count,
        shopping_list=[ShoppingItem(**item) for item in shopping_items_payload] if shopping_items_payload else None,
        shopping_links=instacart_links,
        total_shopping_cost=total_cost,
    )
    try:
        print("[RECOMMEND RESPONSE]", json.dumps(resp.dict(), indent=2))
    except Exception:
        pass
    return resp


@app.post("/recommend/refine", response_model=RecommendationResponse)
def refine_recommendations(req: RefineRequest) -> RecommendationResponse:
    if not req.request:
        raise HTTPException(status_code=400, detail="Missing original request context for refinement.")
    base_req = RecommendRequest(**req.request.dict())
    base_req.ingredients = [i.strip().lower() for i in base_req.ingredients if i.strip()]

    target_fields = {"calories": None, "protein_g": None, "carbs_g": None, "fat_g": None}
    if base_req.profile:
        targets = compute_recommendation_targets(base_req.profile, base_req.macro_split, base_req.target_calories, base_req.goal)
        target_fields = targets

    dishes, provider, param_count, shopping_list, err = refine_recommendations_with_llm(req.feedback, req.dishes, base_req)
    if dishes is None:
        raise HTTPException(
            status_code=503,
            detail=f"LLM unavailable or returned invalid refined recommendations. Last error: {err}",
        )

    notes: List[str] = []
    instacart_links: Optional[dict] = None
    shopping_items_payload: Optional[List[dict]] = None
    
    if shopping_list:
        shopping_items_payload = [item.dict() for item in shopping_list]
    else:
        items_from_dishes = _build_shopping_from_dishes(dishes)
        if items_from_dishes:
            shopping_items_payload = items_from_dishes
        elif base_req.ingredients:
            shopping_items_payload = [{"name": ing, "quantity": 1.0, "unit": "piece", "cost": None} for ing in base_req.ingredients]
    
    if provider:
        notes.append(f"Refined by {provider}.")
    
    if shopping_items_payload:
        try:
            instacart_links = create_shopping_list_with_search_links(
                shopping_items_payload, title="LLM Grocery List (Refined)"
            )
        except Exception as exc:
            notes.append(f"Instacart link generation failed: {exc}")

    used = set()
    for d in dishes:
        used.update([u.lower() for u in d.ingredients_used])
    unknown = [i for i in base_req.ingredients if i not in used]

    total_cost = None
    if shopping_items_payload:
        total_cost = sum((item.get("cost") or 0) for item in shopping_items_payload)
        if total_cost:
            total_cost = round(total_cost, 2)

    return RecommendationResponse(
        period=base_req.period,
        dishes=dishes,
        unknown_items=unknown,
        notes=notes,
        target_calories=target_fields.get("calories"),
        target_protein_g=target_fields.get("protein_g"),
        target_carbs_g=target_fields.get("carbs_g"),
        target_fat_g=target_fields.get("fat_g"),
        cuisine=base_req.cuisine,
        model_provider=provider,
        model_parameters=param_count,
        shopping_list=[ShoppingItem(**item) for item in shopping_items_payload] if shopping_items_payload else None,
        shopping_links=instacart_links,
        total_shopping_cost=total_cost,
    )


@app.post("/rag/ingest")
def rag_ingest(req: RagIngestRequest) -> dict:
    rag: Optional[RagService] = getattr(app.state, "rag", None)
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable; check dependencies.")
    count = rag.ingest_texts(req.texts, req.metadata)
    return {"chunks_ingested": count}


@app.post("/rag/query")
def rag_query(req: RagQueryRequest) -> dict:
    rag: Optional[RagService] = getattr(app.state, "rag", None)
    if not rag:
        raise HTTPException(status_code=503, detail="RAG service unavailable; check dependencies.")
    prompt, sources = rag.build_prompt(req.question, k=req.k)
    raw = _ollama_generate(prompt, model=_recipe_model_name() or os.getenv("OLLAMA_MODEL") or "")
    if not raw:
        raise HTTPException(status_code=503, detail="LLM unavailable for RAG response.")
    return {"answer": raw, "sources": sources}


@app.post("/image-recipe")
async def image_recipe(
    file: UploadFile = File(...),
    hf_model_id: Optional[str] = Form(None),
    ollama_model: Optional[str] = Form(None),
    openai_model: Optional[str] = Form(None),
) -> dict:
    """
    Caption an image and generate a recipe with macros using the selected model.
    """
    data = await file.read()
    caption, source = describe_image_from_bytes(data)
    recipe, provider, err = _generate_recipe_from_caption(caption, hf_model_id, ollama_model, openai_model)
    if recipe is None:
        raise HTTPException(status_code=502, detail=f"Image recipe generation failed: {err}")
    return {
        "caption": caption,
        "caption_source": source,
        "recipe": recipe,
        "model_provider": provider,
    }


@app.post("/shopping-list/render", response_class=HTMLResponse)
def render_shopping_list(req: ShoppingLinksRequest) -> HTMLResponse:
    """
    Render an Instacart shopping list HTML page from shopping_list and optional shopping_links.
    Cleans up the temporary file after generating the HTML string.
    """
    try:
        links = req.shopping_links or create_shopping_list_with_search_links(
            [item.dict() for item in req.shopping_list], title=req.title
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate Instacart links: {exc}")

    try:
        # Write to a temp file using the existing helper, then read and remove.
        tmp_path = generate_html_page(
            links.get("shopping_list_url"),
            links.get("product_links", []),
            filename="instacart_shopping_links.html",
        )
        html_content = Path(tmp_path).read_text(encoding="utf-8")
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to render shopping list HTML: {exc}")

    return HTMLResponse(content=html_content, media_type="text/html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/models")
def models() -> dict:
    catalog = _load_model_catalog()
    return {
        "ollama_model": _default_ollama_model(None),
        "hf_model_id": _default_hf_model(None),
        "hf_models": catalog.get("huggingface", []),
        "ollama_models": catalog.get("ollama", []),
        "openai_model": os.getenv("OPENAI_MODEL") or _default_openai_model(None),
        "openai_models": catalog.get("openai", []),
        "defaults": {
            "hf": _default_hf_model(None),
            "ollama": _default_ollama_model(None),
            "openai": _default_openai_model(None),
        },
    }
