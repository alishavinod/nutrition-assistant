from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict


class Goal(Enum):
    lose = "lose"
    gain = "gain"
    maintenance = "maintenance"


@dataclass
class Profile:
    weight_lb: float                 # body weight in lb
    maintenance_calories: int        # maintenance kcal
    lean_body_mass_lb: Optional[float] = None  # lbm if known



@dataclass
class MacroSplit:
    protein_mode: int        # 1 = 1g/lb, 2 = 1g/lbm, 3 = custom
    custom_protein_g: Optional[float] = None
    fat_percent: float = 20.0
    calorie_adjustment: int = 0  # deficit or surplus


def compute_recommendation_targets(
    profile: Profile,
    macro_split: MacroSplit,
    target_calories: Optional[int] = None,
    goal: Optional[Goal] = Goal.maintenance,
) -> Dict[str, int]:
    """
    Compute daily calorie and macro targets.
    """

    # ---- Calories ----
    if target_calories is None:
        calories = profile.maintenance_calories

        if goal == Goal.lose:
            calories -= macro_split.calorie_adjustment
        elif goal == Goal.gain:
            calories += macro_split.calorie_adjustment
    else:
        calories = target_calories

    if calories <= 0:
        raise ValueError("Target calories must be positive")

    # ---- Protein ----
    if macro_split.protein_mode == 1:
        protein_g = profile.weight_lb

    elif macro_split.protein_mode == 2:
        if profile.lean_body_mass_lb is None:
            raise ValueError("Lean body mass required for protein_mode=2")
        protein_g = profile.lean_body_mass_lb

    elif macro_split.protein_mode == 3:
        if macro_split.custom_protein_g is None:
            raise ValueError("Custom protein required for protein_mode=3")
        protein_g = macro_split.custom_protein_g

    else:
        raise ValueError("Invalid protein_mode")

    # ---- Fat ----
    fat_g = (calories * macro_split.fat_percent / 100.0) / 9.0

    # ---- Carbs ----
    carbs_g = (calories - (protein_g * 4) - (fat_g * 9)) / 4.0

    if carbs_g < 0:
        raise ValueError("Invalid macro split: carbs became negative")

    return {
        "calories": int(calories),
        "protein_g": int(protein_g),
        "carbs_g": int(carbs_g),
        "fat_g": int(fat_g),
    }
