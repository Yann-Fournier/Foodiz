# Foodiz

Une IA qui aide les utilisateurs à suivre leur régime alimentaire.

## Données

Nous avons télécharger le fichier suivant [en.openfoodfacts.org.products.csv](https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz) depuis le site [Open Food Facts](https://world.openfoodfacts.org/data).

Ce fichier est composé de 4 532 765 lignes et de 211 colonnes.

Le fichier est à télécharger dans le dossier `Data`.

## Base de données

Nous avons choisi de stocker nos données dans ***MongoDB***.

**1. Schema très creux (beaucoup de NULLs)**

L'EDA montre que la majorité des colonnes ont >50% de valeurs manquantes. En MongoDB, un champ absent n'existe tout simplement pas dans le document — pas de stockage gaspillé. En MySQL, chaque ligne réserve de l'espace pour les 33 colonnes, même si 20 sont NULL.

**2. Champs multi-valués (listes)**

Plusieurs colonnes contiennent des listes séparées par des virgules :

allergens : "en:milk,en:gluten"
labels_en : "Organic,Vegan,Gluten-free"
categories_en, ingredients_tags, additives_tags...

En MongoDB, ça se stocke directement en arrays et on peut requêter dessus nativement. En MySQL, il faudrait normaliser avec des tables de jointure (product_allergens, product_labels...), ce qui complexifie fortement les requêtes et le pipeline ETL.

**3. Requêtes de recommandation**

Le cas d'usage principal est : "trouve-moi des produits compatibles avec mon régime". C'est du filtrage par document (chercher des produits qui matchent des critères), ce qui est le point fort de MongoDB.

**4. Flexibilité pour le feature engineering**

On va ajouter des champs dérivés (is_vegan, is_keto, is_gluten_free...). En MongoDB, on met à jour les documents directement — pas de ALTER TABLE sur 4.5M lignes.