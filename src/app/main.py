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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import requests

from .providers.hf_recipes import generate_hf_recipes
from .rag_service import RagService, load_seed_documents


ROOT_DIR = Path(__file__).resolve().parents[2]

app = FastAPI(title="Nutrition Planner", version="0.1.0")

# Allow browser demos from static hosts (e.g., GitHub Pages hitting localhost backend).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

class DishRecommendation(BaseModel):
    day: int
    name: str
    reason: str
    ingredients_used: List[str]
    steps: List[str] = Field(default_factory=list)
    calories: Optional[int] = None
    protein_g: Optional[int] = None
    carbs_g: Optional[int] = None
    fat_g: Optional[int] = None

class RecommendRequest(BaseModel):
    ingredients: List[str] = Field(default_factory=list)
    profile: Optional[Profile] = None
    dietary_preference: DietaryPreference = DietaryPreference.omnivore
    goal: Goal = Goal.maintenance
    allergies: List[str] = Field(default_factory=list)
    period: RecommendationPeriod = RecommendationPeriod.one_day
    meals_per_day: int = Field(3, ge=1, le=6)
    macro_split: MacroSplit = MacroSplit()
    target_calories: Optional[int] = Field(None, gt=800, lt=6000)
    budget_amount: Optional[float] = Field(None, gt=0)

class RecommendationResponse(BaseModel):
    period: RecommendationPeriod
    dishes: List[DishRecommendation]
    unknown_items: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


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


def compute_recommendation_targets(
    profile: Profile, macro_split: MacroSplit, target_calories: Optional[int]
) -> dict:
    """
    Compute calorie/macro targets for recommendations using the same logic as plans.
    """
    bmr = mifflin_st_jeor(profile)
    tdee = bmr * profile.activity_level.multiplier
    calories = int(target_calories or round(tdee))

    protein_cal = calories * (macro_split.protein_pct / 100)
    carbs_cal = calories * (macro_split.carbs_pct / 100)
    fat_cal = calories * (macro_split.fat_pct / 100)

    return {
        "calories": calories,
        "protein_g": int(protein_cal / 4),
        "carbs_g": int(carbs_cal / 4),
        "fat_g": int(fat_cal / 9),
    }


def _ollama_generate(text: str, model: str) -> Optional[str]:
    """Call a local Ollama model. Returns raw response text or None on error."""
    url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    try:
        resp = requests.post(
            url,
            json={"model": model, "prompt": text, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response")
    except Exception:
        return None


def _parse_json_block(raw: str) -> Optional[dict]:
    """Attempt to parse JSON; tolerate leading/trailing prose or code fences."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = raw[start : end + 1]
        try:
            return json.loads(snippet)
        except Exception:
            return None
    return None

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
        txt = re.sub(r"[\\s.;,:-]+$", "", s).strip()
        if not txt:
            continue
        if txt[-1] not in ".!?":
            txt = f"{txt}."
        cleaned.append(txt)
    return cleaned[:8] if cleaned else raw_steps
    
def _recipe_model_name() -> Optional[str]:
    return os.getenv("OLLAMA_RECIPE_MODEL") or os.getenv("OLLAMA_MODEL")


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


def _generate_recommendations_with_hf(req: RecommendRequest) -> Optional[List[DishRecommendation]]:
    """
    Generate dishes using the Hugging Face recipe model (HF_MODEL_ID).
    Returns None if the model is not configured or generation fails.
    """
    if not os.getenv("HF_MODEL_ID"):
        return None

    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = req.meals_per_day * days
    pantry = _ingredient_pool(req)

    prompts: List[str] = []
    for idx in range(needed_dishes):
        window = pantry[idx : idx + 3] or pantry
        prompts.append(f"items: {', '.join(window)}")

    raw_recipes = generate_hf_recipes(prompts, **_hf_generation_settings())
    if not raw_recipes:
        return None

    dishes: List[DishRecommendation] = []
    for idx, text in enumerate(raw_recipes[:needed_dishes]):
        title, ingredients, steps = _parse_hf_recipe(text)
        dishes.append(
            DishRecommendation(
                day=(idx // req.meals_per_day) + 1,
                name=title or f"Dish {idx + 1}",
                reason="Generated by Hugging Face model.",
                ingredients_used=ingredients or prompts[idx].replace("items:", "").split(","),
                steps=_normalize_steps(steps or ["Prep ingredients", "Cook until done", "Serve hot"]),
                calories=None,
                protein_g=None,
                carbs_g=None,
                fat_g=None,
            )
        )

    # Ensure we return exactly the requested count.
    while len(dishes) < needed_dishes and dishes:
        src = dishes[len(dishes) % len(dishes)]
        dishes.append(
            DishRecommendation(
                day=(len(dishes) // req.meals_per_day) + 1,
                name=src.name,
                reason=src.reason,
                ingredients_used=src.ingredients_used,
                steps=src.steps,
                calories=src.calories,
                protein_g=src.protein_g,
                carbs_g=src.carbs_g,
                fat_g=src.fat_g,
            )
        )
    if len(dishes) > needed_dishes:
        dishes = dishes[:needed_dishes]
    return dishes


def generate_recommendations_with_llm(req: RecommendRequest) -> Tuple[Optional[List[DishRecommendation]], str]:
    """
    Try Hugging Face first, then Ollama. Returns dishes and a provider label for notes.
    Provider label: "huggingface:<model>" | "ollama:<model>" | "".
    """
    hf_dishes = _generate_recommendations_with_hf(req)
    if hf_dishes:
        return hf_dishes, f"huggingface:{os.getenv('HF_MODEL_ID')}"

    model = _recipe_model_name()
    if not model:
        return None, ""
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
        targets = compute_recommendation_targets(req.profile, req.macro_split, req.target_calories)
        profile_txt = (
            f"{req.profile.sex}, {req.profile.age}y, {req.profile.height_cm}cm, "
            f"{req.profile.weight_kg}kg, activity {req.profile.activity_level.value}"
        )
        targets_txt = (
            f"calories ~{targets['calories']} kcal/day, macros P:{targets['protein_g']}g "
            f"C:{targets['carbs_g']}g F:{targets['fat_g']}g (from split "
            f"{req.macro_split.protein_pct}/{req.macro_split.carbs_pct}/{req.macro_split.fat_pct})"
        )

    rag_ctx = ""
    rag_sources: List[dict] = []
    rag_service: Optional[RagService] = getattr(app.state, "rag", None)  # type: ignore[attr-defined]
    if rag_service:
        query = (
            f"Ingredients: {ingredients_txt}. Diet: {req.dietary_preference.value}. "
            f"Goal: {req.goal}. Profile: {profile_txt}. Targets: {targets_txt}."
        )
        rag_ctx, rag_sources = rag_service.search_context(query, k=6)

    prompt = f"""
You are a nutrition chef. Recommend simple dishes ONLY using these ingredients: {ingredients_txt}.
Diet: {req.dietary_preference.value}. Goal: {req.goal.replace('_',' ')}. Allergies: {allergies_txt}.
User profile: {profile_txt}. Targets: {targets_txt}. Budget: {budget_txt} ({budget_per_day}); favor affordable ingredients.
Use this nutrition context (prioritize these ingredients/macros if relevant):
{rag_ctx if rag_ctx else "No context provided; rely on general nutrition knowledge."}
Return JSON: {{"dishes":[{{"day":1,"name":"...","reason":"...","ingredients_used":["..."],"steps":["..."],"calories":500,"protein_g":40,"carbs_g":50,"fat_g":15}}]}}.
Days: {days}. Give specific dish names (e.g., "Spinach Feta Omelette", "Chipotle Chicken Wrap")—no generic "Morning/Lunch/Evening".
If meals_per_day is 3, treat them as Breakfast, Lunch, Dinner in order; otherwise, rotate meal types and diversify cuisines (no all-breakfast set).
Return exactly {needed_dishes} dishes total ({req.meals_per_day} per day), using day values 1..{days} with {req.meals_per_day} dishes per day.
Respond with JSON only, no markdown code fences or extra text. Each dish must have 3-8 short steps (one action per step).
"""
    raw = _ollama_generate(prompt, model=model)
    if not raw:
        return None, ""
    try:
        data = _parse_json_block(raw)
        if not data:
            return None, ""
        dishes: List[DishRecommendation] = []
        for d in data.get("dishes", []):
            raw_steps = _normalize_steps([str(s) for s in d.get("steps", []) if s])
            dishes.append(
                DishRecommendation(
                    day=int(d.get("day", 1)),
                    name=d.get("name", "Dish"),
                    reason=d.get("reason", ""),
                    ingredients_used=[str(i) for i in d.get("ingredients_used", []) if i],
                    steps=raw_steps,
                    calories=d.get("calories"),
                    protein_g=d.get("protein_g"),
                    carbs_g=d.get("carbs_g"),
                    fat_g=d.get("fat_g"),
                )
            )
        if not dishes:
            return None, ""
        # Normalize to exactly needed_dishes (meals_per_day per day); pad by cycling existing dishes if short.
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
                        steps=src.steps,
                        calories=src.calories,
                        protein_g=src.protein_g,
                        carbs_g=src.carbs_g,
                        fat_g=src.fat_g,
                    )
                )
        if len(dishes) > needed_dishes:
            dishes = dishes[:needed_dishes]
        # Ensure day labels cycle correctly (meals_per_day dishes per day).
        for idx, dish in enumerate(dishes):
            dish.day = (idx // req.meals_per_day) + 1
        return dishes, f"ollama:{model}"
    except Exception:
        return None, ""



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

    dishes, provider = generate_recommendations_with_llm(clean_req)
    fallback_used = False
    if dishes is None:
        dishes = _fallback_recommendations(clean_req)
        fallback_used = True

    notes: List[str] = []
    if provider and not fallback_used:
        notes.append(f"Generated by {provider}.")
    elif _recipe_model_name() and not fallback_used:
        notes.append(f"Generated by model {_recipe_model_name()}.")
    elif os.getenv("HF_MODEL_ID") and not fallback_used:
        notes.append(f"Generated by Hugging Face model {os.getenv('HF_MODEL_ID')}.")
    if fallback_used:
        notes.append("LLM unavailable; served deterministic fallback suggestions.")

    used = set()
    for d in dishes:
        used.update([u.lower() for u in d.ingredients_used])
    unknown = [i for i in ingredients if i not in used]

    return RecommendationResponse(
        period=req.period,
        dishes=dishes,
        unknown_items=unknown,
        notes=notes,
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
    raw = _ollama_generate(prompt, model=_recipe_model_name() or os.getenv("OLLAMA_MODEL"))
    if not raw:
        raise HTTPException(status_code=503, detail="LLM unavailable for RAG response.")
    return {"answer": raw, "sources": sources}



@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/models")
def models() -> dict:
    return {
        "ollama_model": os.getenv("OLLAMA_MODEL"),
        "ollama_recipe_model": os.getenv("OLLAMA_RECIPE_MODEL"),
        "hf_model_id": os.getenv("HF_MODEL_ID"),
    }
