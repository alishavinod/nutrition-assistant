"""
FastAPI backend for the Nutrition Planner demo.
Features:
- Macro calculation via Mifflin-St Jeor + activity level.
- Optional LLM-generated plan via local Ollama (falls back to stub).
- Dummy pricing catalog for cost estimation (no real product search).
- CORS enabled for static frontend calls.
"""

import json
import os
import csv
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI
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


class BudgetPeriod(str, Enum):
    week = "week"
    month = "month"


class DietaryPreference(str, Enum):
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
    budget_period: BudgetPeriod = BudgetPeriod.week
    meals_per_day: int = Field(3, ge=1, le=6)
    macro_split: MacroSplit = MacroSplit()
    target_calories: Optional[int] = Field(None, gt=800, lt=6000)
    use_llm: bool = Field(
        False,
        description="If true, attempt to generate the plan via a local LLM (ollama) before falling back to the stub.",
    )


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
    budget_period: BudgetPeriod
    days: List[DayPlan]
    estimated_cost: Optional[float] = None
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


def stub_meal_plan(req: PlanRequest, targets: dict) -> List[DayPlan]:
    days = 7 if req.budget_period == BudgetPeriod.week else 30
    meals: List[DayPlan] = []

    protein_per_meal = targets["protein_g"] // req.meals_per_day
    carbs_per_meal = targets["carbs_g"] // req.meals_per_day
    fat_per_meal = targets["fat_g"] // req.meals_per_day
    calories_per_meal = targets["calories"] // req.meals_per_day

    # Ingredient names are aligned to data/dummy_catalog.csv for pricing.
    base_meals = {
        DietaryPreference.omnivore: [
            ("Breakfast", [("greek yogurt", "1 cup"), ("berries", "1 cup"), ("oats", "1/2 cup"), ("almonds", "1 oz")]),
            ("Lunch", [("chicken breast", "150 g"), ("rice", "1 cup"), ("broccoli", "1 cup")]),
            ("Dinner", [("salmon", "150 g"), ("sweet potato", "1 medium"), ("spinach", "1 cup")]),
        ],
        DietaryPreference.vegetarian: [
            ("Breakfast", [("oats", "1 cup"), ("berries", "1 cup"), ("almonds", "1 oz")]),
            ("Lunch", [("lentils", "1 cup"), ("quinoa", "1 cup"), ("broccoli", "1 cup")]),
            ("Dinner", [("tofu", "150 g"), ("brown rice", "1 cup"), ("spinach", "1 cup")]),
        ],
        DietaryPreference.vegan: [
            ("Breakfast", [("oats", "1 cup"), ("berries", "1 cup"), ("almonds", "1 oz")]),
            ("Lunch", [("lentils", "1 cup"), ("quinoa", "1 cup"), ("spinach", "1 cup")]),
            ("Dinner", [("tofu", "150 g"), ("sweet potato", "1 medium"), ("broccoli", "1 cup")]),
        ],
        DietaryPreference.pescatarian: [
            ("Breakfast", [("greek yogurt", "1 cup"), ("berries", "1 cup"), ("oats", "1/2 cup")]),
            ("Lunch", [("salmon", "120 g"), ("rice", "1 cup"), ("mixed veggies", "1 cup")]),
            ("Dinner", [("salmon", "150 g"), ("sweet potato", "1 medium"), ("spinach", "1 cup")]),
        ],
        DietaryPreference.keto: [
            ("Breakfast", [("eggs", "2"), ("avocado", "1/2"), ("almonds", "1 oz")]),
            ("Lunch", [("chicken breast", "150 g"), ("broccoli", "1 cup"), ("avocado", "1/2")]),
            ("Dinner", [("salmon", "150 g"), ("spinach", "1 cup"), ("broccoli", "1 cup")]),
        ],
        DietaryPreference.paleo: [
            ("Breakfast", [("eggs", "2"), ("sweet potato", "1 small"), ("spinach", "1 cup")]),
            ("Lunch", [("chicken breast", "150 g"), ("mixed veggies", "1 cup")]),
            ("Dinner", [("salmon", "150 g"), ("broccoli", "1 cup"), ("sweet potato", "1 medium")]),
        ],
    }[req.dietary_preference]

    for day in range(1, days + 1):
        day_meals: List[Meal] = []
        for idx in range(req.meals_per_day):
            name, ingredients_raw = base_meals[idx % len(base_meals)]
            day_meals.append(
                Meal(
                    name=name,
                    calories=calories_per_meal,
                    protein_g=protein_per_meal,
                    carbs_g=carbs_per_meal,
                    fat_g=fat_per_meal,
                    ingredients=[Ingredient(name=i_name, quantity=i_qty) for i_name, i_qty in ingredients_raw],
                )
            )
        meals.append(
            DayPlan(
                day=day,
                meals=day_meals,
                total_calories=calories_per_meal * req.meals_per_day,
                total_protein_g=protein_per_meal * req.meals_per_day,
                total_carbs_g=carbs_per_meal * req.meals_per_day,
                total_fat_g=fat_per_meal * req.meals_per_day,
            )
        )
    return meals


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


def generate_plan_with_llm(req: PlanRequest, targets: dict) -> Optional[List[DayPlan]]:
    """
    Attempt to create a plan via a local LLM (ollama). Expects JSON output matching the DayPlan schema.
    If anything fails, return None and fall back to stub.
    """
    model = os.getenv("OLLAMA_MODEL")
    if not model:
        return None

    days = 7 if req.budget_period == BudgetPeriod.week else 30
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
        data = json.loads(raw)
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
            parsed_days.append(
                DayPlan(
                    day=int(day.get("day", 0)),
                    meals=meals,
                    total_calories=int(day.get("total_calories", 0)),
                    total_protein_g=int(day.get("total_protein_g", 0)),
                    total_carbs_g=int(day.get("total_carbs_g", 0)),
                    total_fat_g=int(day.get("total_fat_g", 0)),
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
    Generate a meal plan. Tries LLM first (if enabled and configured), else uses the stub.
    """
    targets = compute_targets(req)
    days = generate_plan_with_llm(req, targets) if req.use_llm else None
    if days is None:
        days = stub_meal_plan(req, targets)

    estimated_cost, unknown = estimate_plan_cost(days)

    notes = [
        "If use_llm=true and a local Ollama model is configured (OLLAMA_MODEL), the plan is generated by the model; otherwise it falls back to the static template.",
        "When product search is available, map each ingredient to real products and prices.",
        "Cost uses data/dummy_catalog.csv; quantities are ignored—replace with real pricing for production.",
    ]

    return PlanResponse(
        target_calories=targets["calories"],
        target_protein_g=targets["protein_g"],
        target_carbs_g=targets["carbs_g"],
        target_fat_g=targets["fat_g"],
        budget_amount=req.budget_amount,
        budget_period=req.budget_period,
        days=days,
        estimated_cost=round(estimated_cost, 2) if estimated_cost else None,
        unknown_items=unknown,
        notes=notes,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
