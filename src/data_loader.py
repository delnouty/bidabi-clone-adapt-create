import csv
import time
import requests
import os

BASE_URL = "https://world.openfoodfacts.org/category/{category}.json"
HEADERS = {"User-Agent": "MyAwesomeApp/1.0"}

TARGET_COUNT = 280          # nombre de produits valides à collecter
PAGE_SIZE = 100             # nombre de produits par page
MAX_PAGES = 50              # limite de pages à parcourir
CATEGORY = "champagnes"     # catégorie cible


def fetch_page(category, page, page_size):
    """
    Télécharge une page de produits depuis l'API OpenFoodFacts.
    """
    url = BASE_URL.format(category=category)
    params = {"page": page, "page_size": page_size, "json": 1}

    try:
        response = requests.get(
            url,
            params=params,
            headers=HEADERS,
            timeout=60
        )
        response.raise_for_status()
        return response.json().get("products", [])
    except requests.exceptions.RequestException as error:
        print(f"Erreur API sur la page {page} :", error)
        return []


def is_valid_product(product):
    """
    Vérifie si un produit contient les champs nécessaires.
    """
    required_fields = ["_id", "product_name", "categories_tags"]

    for field in required_fields:
        if not product.get(field):
            return False

    image = product.get("image_url") or product.get("image_front_url")
    return bool(image)


def extract_product_info(product):
    """
    Extrait les informations utiles d'un produit.
    """
    return [
        product.get("_id"),
        product.get("product_name"),
        ", ".join(product.get("categories_tags", [])),
        product.get("ingredients_text", ""),
        product.get("image_url") or product.get("image_front_url")
    ]


def save_to_csv(filename, rows):
    """
    Sauvegarde les produits valides dans un fichier CSV.
    """
    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["foodId", "label", "category", "foodContentsLabel", "image"]
        )
        writer.writerows(rows)


def download_image(image_url, image_id, folder="images"):
    """
    Télécharge l'image d'un produit dans un dossier local.
    """
    os.makedirs(folder, exist_ok=True)

    ext = image_url.split(".")[-1].split("?")[0]
    filename = os.path.join(folder, f"{image_id}.{ext}")

    try:
        response = requests.get(image_url, headers=HEADERS, timeout=60)
        response.raise_for_status()

        with open(filename, "wb") as f:
            f.write(response.content)

    except requests.exceptions.RequestException as error:
        print(f"⚠ Impossible de télécharger l'image {image_url} :", error)


def main():
    """
    Pipeline principal :
    - télécharge les pages une par une
    - filtre les produits valides
    - télécharge les images
    - s'arrête quand TARGET_COUNT est atteint
    - sauvegarde dans un CSV
    """
    valid_products = []
    page = 1

    while len(valid_products) < TARGET_COUNT and page <= MAX_PAGES:
        print(f"→ Téléchargement page {page}…")

        products = fetch_page(CATEGORY, page, PAGE_SIZE)

        if not products:
            print("Aucun produit trouvé sur cette page.")
            break

        for product in products:
            if is_valid_product(product):
                info = extract_product_info(product)
                valid_products.append(info)

                # téléchargement de l'image
                image_url = info[-1]
                image_id = info[0]
                download_image(image_url, image_id)

            if len(valid_products) == TARGET_COUNT:
                break

        page += 1
        time.sleep(0.5)

    output_file = f"{CATEGORY}_{TARGET_COUNT}.csv"
    save_to_csv(output_file, valid_products)

    print(
        f"✔ Fichier {output_file} créé. "
        f"Produits valides collectés : {len(valid_products)}"
    )


if __name__ == "__main__":
    main()
