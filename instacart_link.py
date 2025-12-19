"""
Minimal CLI to create an Instacart shopping-list link for one or more items.

Usage:
  export INSTACART_API_KEY="keys.xxx"
  python instacart_link.py "milk" "eggs" --title "Breakfast run"

Notes:
- The script targets the dev host by default; pass --prod to hit connect.instacart.com.
- Only standard library + requests are used; install requests if needed:
    python -m pip install requests
"""

import argparse
import json
import os
from typing import List, Optional

import requests


def build_payload(items: List[str], title: str, expires_in_days: int, linkback: Optional[str]) -> dict:
    """Shape the payload for /idp/v1/products/products_link."""
    line_items = [{"name": item} for item in items]

    payload: dict = {
        "title": title,
        "expires_in": expires_in_days,
        "line_items": line_items,
    }

    if linkback:
        payload["landing_page_configuration"] = {"partner_linkback_url": linkback}

    return payload


def create_products_link(base_url: str, api_key: str, payload: dict) -> dict:
    """Call Instacart Connect to create a products link."""
    url = base_url.rstrip("/") + "/idp/v1/products/products_link"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an Instacart shopping list link for given items.")
    parser.add_argument(
        "items",
        nargs="*",
        help="Item names to search for (e.g., 'milk' 'eggs'). Leave blank to be prompted.",
    )
    parser.add_argument("--title", default="My Instacart list", help="Title for the generated shopping page.")
    parser.add_argument(
        "--expires-in-days",
        type=int,
        default=7,
        help="Link expiry in days (1-365, default: 7).",
    )
    parser.add_argument("--linkback", help="Optional linkback URL shown on the shopping page.")
    parser.add_argument(
        "--prod",
        action="store_true",
        help="Use the production host connect.instacart.com instead of the dev sandbox.",
    )

    args = parser.parse_args()

    items = args.items
    if not items:
        raw = input("Enter items separated by commas (e.g., milk, eggs, bread): ").strip()
        items = [part.strip() for part in raw.split(",") if part.strip()]
        if not items:
            raise SystemExit("No items provided.")

    api_key = os.getenv("INSTACART_API_KEY")
    if not api_key:
        raise SystemExit("INSTACART_API_KEY env var is required.")

    base_url = "https://connect.instacart.com" if args.prod else "https://connect.dev.instacart.tools"

    expires_days = args.expires_in_days
    if not 1 <= expires_days <= 365:
        raise SystemExit("expires-in-days must be between 1 and 365.")

    payload = build_payload(items, args.title, expires_days, args.linkback)

    try:
        result = create_products_link(base_url, api_key, payload)
    except requests.HTTPError as exc:  # pragma: no cover - for runtime clarity
        print(f"Request failed: {exc.response.status_code} {exc.response.text}")
        raise SystemExit(1)

    # API typically returns a link field (name varies slightly across docs).
    link = result.get("products_link") or result.get("link") or result
    print(json.dumps({"link": link, "raw_response": result}, indent=2))


if __name__ == "__main__":
    main()




## export INSTACART_API_KEY="Bearer keys.aey4IdGLsXAYehcNXQWR7IiSrph999Qyo15OEehAWCM"


keys.aey4IdGLsXAYehcNXQWR7IiSrph999Qyo15OEehAWCM