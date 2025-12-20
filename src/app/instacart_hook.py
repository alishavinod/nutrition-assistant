import http.client
import json
import urllib.parse
import os

#Configuration
API_KEY = os.getenv("INSTACART_API_KEY")

API_HOST = "connect.dev.instacart.tools"

def create_shopping_list_with_search_links(ingredients_list, title="My Shopping List"):
    """
    Creates a shopping list and generates direct Instacart search links for each item.
    Since exact product matching may not work, we provide search links that users can click.
    
    Args:
        ingredients_list: List of dictionaries with 'name', 'quantity', 'unit'
        Example: [
            {"name": "milk", "quantity": 1, "unit": "gallon"},
            {"name": "chicken breast", "quantity": 2, "unit": "pound"}
        ]
    
    Returns:
        Dictionary with shopping list URL and individual product search links
    """
    
    # 1. Create the shopping list via API
    conn = http.client.HTTPSConnection(API_HOST)
    
    line_items = []
    for item in ingredients_list:
        qty = item.get("quantity") or 1
        unit = item.get("unit") or "piece"
        line_items.append({
            "name": item["name"],
            "quantity": qty,
            "unit": unit
        })
    
    payload = {
        "title": title,
        "line_items": line_items
    }
    
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    
    shopping_list_url = None
    try:
        conn.request("POST", "/idp/v1/products/products_link", json.dumps(payload), headers)
        response = conn.getresponse()
        data = response.read()
        result = json.loads(data.decode("utf-8"))
        
        if response.status in [200, 201]:
            shopping_list_url = result.get("products_link_url")
            print(f"✓ Shopping list created: {shopping_list_url}\n")
        else:
            print(f"✗ API Error: {result}\n")
    except Exception as e:
        print(f"✗ Error: {str(e)}\n")
    finally:
        conn.close()
    
    # 2. Create individual search links for each product
    product_links = []
    for item in ingredients_list:
        # Build search query
        qty = item.get('quantity') or 1
        unit = item.get('unit') or "piece"
        search_query = item['name']
        if qty:
            search_query += f" {qty}"
        if unit:
            search_query += f" {unit}"
        
        # Create Instacart search link
        encoded_query = urllib.parse.quote(search_query)
        search_url = f"https://www.instacart.com/store/s?k={encoded_query}"
        
        product_links.append({
            "name": item['name'],
            "quantity": qty,
            "unit": unit,
            "search_term": search_query,
            "link": search_url
        })
    
    return {
        "shopping_list_url": shopping_list_url,
        "product_links": product_links
    }

def generate_html_page(shopping_list_url, product_links, filename="instacart_shopping_links.html", template_path=None):
    """
    Render the shopping list HTML from a template file.
    - template_path defaults to src/app/instacart_template.html
    - Falls back to a simple inline template if the file is missing.
    """
    from pathlib import Path

    template_path = Path(template_path or Path(__file__).with_name("instacart_template.html"))

    # Build product list HTML
    product_items_html = ""
    for product in product_links:
        product_items_html += f"""
        <div class="product-item">
            <div class="product-name">{product.get('search_term','')}</div>
            <div class="product-qty">{product.get('quantity','')} {product.get('unit','')}</div>
            <a href="{product.get('link','')}" target="_blank" class="product-link">Search on Instacart →</a>
        </div>
"""

    shopping_section = ""
    if shopping_list_url:
        shopping_section = f"""
        <a href="{shopping_list_url}" target="_blank" class="main-link">
            📋 View Complete Shopping List on Instacart →
        </a>
        <div class="note">
            <strong>Note:</strong> If some items say "not available" on the shopping list page,
            use the individual product search links below to find them.
        </div>
"""
    else:
        shopping_section = """
        <div class="note">
            <strong>No Instacart URL available.</strong> You can still use the search links below to shop items.
        </div>
"""
    try:
        html = template_path.read_text(encoding="utf-8")
        html = html.replace("{{SHOPPING_LIST_SECTION}}", shopping_section)
        html = html.replace("{{PRODUCT_ITEMS}}", product_items_html)
    except Exception:
        html = f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"><title>Instacart Shopping Links</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Arial,sans-serif;background:#0f172a;color:#e5e7eb;padding:24px;}}
.fallback{{max-width:900px;margin:0 auto;background:#0b1220;border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:20px;}}
a{{color:#34d399;}}
</style>
</head>
<body>
<div class="fallback">
<h1>🛒 Your Instacart Shopping List</h1>
{shopping_section}
<h2>Individual Product Search Links</h2>
{product_items_html}
</div>
</body></html>"""

    Path(filename).write_text(html, encoding="utf-8")
    return filename

ingredients_from_source = []  # Replace with your actual data
# ============================================


print("=" * 70)
print("CREATING INSTACART SHOPPING EXPERIENCE")
print("=" * 70)
print()

# Create shopping list and get links
result = create_shopping_list_with_search_links(
    ingredients_from_source,
    title="Weekly Grocery Shopping"
)

# Display results
if result["shopping_list_url"]:
    print("🎉 SHOPPING LIST URL:")
    print(f"   {result['shopping_list_url']}")
    print()

print("📦 INDIVIDUAL PRODUCT SEARCH LINKS:")
print()
for i, product in enumerate(result["product_links"], 1):
    print(f"{i}. {product['search_term']}")
    print(f"   {product['link']}")
    print()

# Generate HTML file
html_file = generate_html_page(
    result["shopping_list_url"],
    result["product_links"]
)

print("=" * 70)
print(f"✓ HTML file created: {html_file}")
print(f"  Open this file in your browser for clickable links!")
print("=" * 70)
