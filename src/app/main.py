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
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import requests


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
    non_vegeterian = "omnivore"  # backward compatibility for existing requests
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
    dietary_preference: DietaryPreference = DietaryPreference.omnivore
    goal: Goal = Goal.maintenance
    allergies: List[str] = Field(default_factory=list)
    period: RecommendationPeriod = RecommendationPeriod.one_day
    meals_per_day: int = Field(3, ge=1, le=6)

class RecommendationResponse(BaseModel):
    period: RecommendationPeriod
    dishes: List[DishRecommendation]
    unknown_items: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)



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
    
def _recipe_model_name() -> Optional[str]:
    return os.getenv("OLLAMA_RECIPE_MODEL") or os.getenv("OLLAMA_MODEL")

def generate_recommendations_with_llm(req: RecommendRequest) -> Optional[List[DishRecommendation]]:
    model = _recipe_model_name()
    if not model:
        return None
    days = 1 if req.period == RecommendationPeriod.one_day else 7
    needed_dishes = req.meals_per_day * days
    ingredients_txt = ", ".join(req.ingredients) if req.ingredients else "none specified—use pantry-friendly staples that fit the diet"
    allergies_txt = ", ".join(req.allergies) if req.allergies else "none"

    prompt = f"""
You are a nutrition chef. Recommend simple dishes ONLY using these ingredients: {ingredients_txt}.
Diet: {req.dietary_preference.value}. Goal: {req.goal.replace('_',' ')}. Allergies: {allergies_txt}.
Return JSON: {{"dishes":[{{"day":1,"name":"...","reason":"...","ingredients_used":["..."],"steps":["..."],"calories":500,"protein_g":40,"carbs_g":50,"fat_g":15}}]}}.
Days: {days}. Keep names short; avoid missing ingredients.
Return exactly {needed_dishes} dishes total ({req.meals_per_day} per day), using day values 1..{days} with {req.meals_per_day} dishes per day.
Respond with JSON only, no markdown code fences or extra text. Each dish must have 3-8 short steps (one action per step).
"""
    raw = _ollama_generate(prompt, model=model)
    if not raw:
        return None
    try:
        data = _parse_json_block(raw)
        if not data:
            return None
        dishes: List[DishRecommendation] = []
        for d in data.get("dishes", []):
            raw_steps = [str(s) for s in d.get("steps", []) if s]
            if len(raw_steps) == 1:
                # Try splitting a single long step into sentences.
                parts = [p.strip() for p in raw_steps[0].replace("\n", " ").split(". ") if p.strip()]
                if len(parts) >= 2:
                    raw_steps = parts[:8]
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
            return None
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
        return dishes
    except Exception:
        return None



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

    dishes = generate_recommendations_with_llm(clean_req)
    if dishes is None:
        raise HTTPException(
            status_code=503,
            detail="LLM unavailable or returned invalid recommendations. Ensure OLLAMA_MODEL/OLLAMA_RECIPE_MODEL is configured and reachable.",
        )

    notes: List[str] = [f"Generated by model {_recipe_model_name()}."]

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



@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

@app.get("/models")
def models() -> dict:
    return {"ollama_model": os.getenv("OLLAMA_MODEL"), "ollama_recipe_model": os.getenv("OLLAMA_RECIPE_MODEL")}
