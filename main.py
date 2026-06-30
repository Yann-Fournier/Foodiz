"""Foodiz — Interface de demonstration Streamlit.

Lance avec : streamlit run main.py
Pre-requis : MongoDB doit tourner en local (localhost:27017, base 'foodiz')
et les modeles entraines doivent exister dans Models/ (notebooks 3 et 4).
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "foodiz"
COLLECTION_NAME = "products"
MODELS_DIR = "Models"

GRADE_ORDER = ["a", "b", "c", "d", "e"]
GRADE_COLORS = {"A": "#2d8c2d", "B": "#8bc34a", "C": "#ffc107", "D": "#ff9800", "E": "#f44336"}


# ---------------------------------------------------------------------------
# Connexion MongoDB et chargement des modeles (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_collection():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    return client[DB_NAME][COLLECTION_NAME]


@st.cache_resource
def load_models():
    paths = {
        "classifier": "regime_classifier_rf.joblib",
        "classifier_features": "regime_classifier_features.joblib",
        "classifier_labels": "regime_classifier_labels.joblib",
        "cluster_preprocessor": "cluster_preprocessor.joblib",
        "cluster_features": "cluster_features.joblib",
    }
    models = {}
    for key, filename in paths.items():
        full_path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(full_path):
            return None
        models[key] = joblib.load(full_path)
    return models


# ---------------------------------------------------------------------------
# Helpers de donnees
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_overview_stats(_collection):
    total = _collection.count_documents({})

    nutriscore_counts = list(_collection.aggregate([
        {"$match": {"quality.nutriscore_grade": {"$exists": True}}},
        {"$group": {"_id": "$quality.nutriscore_grade", "count": {"$sum": 1}}},
    ]))
    nova_counts = list(_collection.aggregate([
        {"$match": {"quality.nova_group": {"$exists": True}}},
        {"$group": {"_id": "$quality.nova_group", "count": {"$sum": 1}}},
    ]))
    category_counts = list(_collection.aggregate([
        {"$match": {"main_category": {"$exists": True}}},
        {"$group": {"_id": "$main_category", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15},
    ]))

    regime_fields = [
        "is_vegan", "is_vegetarian", "is_gluten_free", "is_lactose_free",
        "is_keto", "is_low_carb", "is_high_protein", "is_low_sugar", "is_organic",
    ]
    regime_counts = {f: _collection.count_documents({f"regime.{f}": True}) for f in regime_fields}

    return {
        "total": total,
        "nutriscore_counts": nutriscore_counts,
        "nova_counts": nova_counts,
        "category_counts": category_counts,
        "regime_counts": regime_counts,
    }


def search_products(_collection, query_text="", require_cluster=False, limit=20):
    mongo_query = {}
    if query_text:
        mongo_query["product_name"] = {"$regex": query_text, "$options": "i"}
    if require_cluster:
        mongo_query["cluster"] = {"$exists": True}
    return list(_collection.find(mongo_query, {"_id": 0}).limit(limit))


def doc_to_feature_row(doc, feature_cols):
    sections = {
        "nutrition": doc.get("nutrition", {}) or {},
        "quality": doc.get("quality", {}) or {},
        "meta": doc.get("meta", {}) or {},
    }
    flat = {col: sections[col.split(".")[0]].get(col.split(".")[1]) for col in feature_cols}
    return pd.DataFrame([flat])[feature_cols]


def find_alternatives(collection, models, product, n=5, only_healthier=True, candidate_limit=200):
    if "cluster" not in product:
        return pd.DataFrame()

    query = {"cluster": product["cluster"], "code": {"$ne": product["code"]}}
    if product.get("main_category"):
        query["main_category"] = product["main_category"]

    current_grade = (product.get("quality", {}) or {}).get("nutriscore_grade")
    if only_healthier and current_grade in GRADE_ORDER:
        query["quality.nutriscore_grade"] = {"$in": GRADE_ORDER[:GRADE_ORDER.index(current_grade) + 1]}

    candidates = list(collection.find(query, {"_id": 0}).limit(candidate_limit))
    if not candidates:
        return pd.DataFrame()

    feature_cols = models["cluster_features"]
    product_vector = models["cluster_preprocessor"].transform(doc_to_feature_row(product, feature_cols))[0]
    candidate_rows = pd.concat([doc_to_feature_row(c, feature_cols) for c in candidates], ignore_index=True)
    candidate_vectors = models["cluster_preprocessor"].transform(candidate_rows)
    distances = np.linalg.norm(candidate_vectors - product_vector, axis=1)

    result = pd.DataFrame([{
        "Produit": c.get("product_name"),
        "Marque": c.get("brands"),
        "Nutri-Score": ((c.get("quality", {}) or {}).get("nutriscore_grade") or "?").upper(),
        "Energie (kcal/100g)": (c.get("nutrition", {}) or {}).get("energy_kcal"),
        "_distance": d,
    } for c, d in zip(candidates, distances)])

    return result.sort_values("_distance").drop(columns="_distance").head(n).reset_index(drop=True)


def predict_regime(models, feature_values):
    feature_cols = models["classifier_features"]
    row = pd.DataFrame([feature_values])[feature_cols]
    proba_list = models["classifier"].predict_proba(row)

    results = []
    for label, proba in zip(models["classifier_labels"], proba_list):
        if proba.shape[1] == 2:
            p_positive = proba[0][1]
        else:
            p_positive = 1.0 if models["classifier"].named_steps["clf"].estimators_[
                models["classifier_labels"].index(label)
            ].classes_[0] == 1 else 0.0
        results.append({"label": label, "proba": p_positive, "compatible": p_positive >= 0.5})
    return results


def render_regime_badges(results):
    cols = st.columns(3)
    for i, r in enumerate(results):
        label_clean = r["label"].replace("is_", "").replace("_", " ").title()
        col = cols[i % 3]
        col.metric(
            label_clean,
            "Compatible" if r["compatible"] else "Non compatible",
            f"{r['proba']:.0%} confiance",
        )


def render_product_summary(product):
    quality = product.get("quality", {}) or {}
    nutrition = product.get("nutrition", {}) or {}

    st.subheader(product.get("product_name", "Produit sans nom"))
    st.caption(f"{product.get('brands', 'Marque inconnue')} — {product.get('main_category', 'Categorie inconnue')}")

    cols = st.columns(4)
    cols[0].metric("Nutri-Score", (quality.get("nutriscore_grade") or "?").upper())
    cols[1].metric("Groupe NOVA", quality.get("nova_group", "?"))
    cols[2].metric("Energie", f"{nutrition.get('energy_kcal', '?')} kcal/100g")
    cols[3].metric("Cluster", product.get("cluster", "Non assigne"))

    regime = product.get("regime", {}) or {}
    compatible = [k.replace("is_", "").replace("_", " ").title() for k, v in regime.items() if v]
    if compatible:
        st.write("**Compatible avec :** " + ", ".join(compatible))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview(collection):
    st.header("Vue d'ensemble du dataset")

    t0 = time.time()
    stats = get_overview_stats(collection)
    duree = (time.time() - t0) * 1000
    st.caption(f"Statistiques calculees en {duree:.0f} ms (cache 10 min)")

    c1, c2, c3 = st.columns(3)
    c1.metric("Produits references", f"{stats['total']:,}")
    c2.metric("Avec Nutri-Score", f"{sum(d['count'] for d in stats['nutriscore_counts']):,}")
    c3.metric("Avec NOVA", f"{sum(d['count'] for d in stats['nova_counts']):,}")

    col1, col2 = st.columns(2)

    nutriscore_df = pd.DataFrame(stats["nutriscore_counts"])
    if not nutriscore_df.empty:
        nutriscore_df["grade"] = nutriscore_df["_id"].str.upper()
        nutriscore_df["grade"] = pd.Categorical(nutriscore_df["grade"], categories=list(GRADE_COLORS), ordered=True)
        nutriscore_df = nutriscore_df.sort_values("grade")
        fig = px.bar(
            nutriscore_df, x="grade", y="count", color="grade",
            color_discrete_map=GRADE_COLORS, title="Distribution du Nutri-Score",
        )
        fig.update_layout(showlegend=False)
        col1.plotly_chart(fig, use_container_width=True)

    nova_df = pd.DataFrame(stats["nova_counts"])
    if not nova_df.empty:
        nova_df = nova_df.rename(columns={"_id": "nova"}).sort_values("nova")
        fig = px.bar(nova_df, x="nova", y="count", title="Distribution du groupe NOVA")
        col2.plotly_chart(fig, use_container_width=True)

    cat_df = pd.DataFrame(stats["category_counts"]).rename(columns={"_id": "categorie"})
    if not cat_df.empty:
        fig = px.bar(
            cat_df.sort_values("count"), x="count", y="categorie", orientation="h",
            title="Top 15 categories de produits",
        )
        st.plotly_chart(fig, use_container_width=True)

    regime_df = pd.DataFrame([
        {"regime": k.replace("is_", "").replace("_", " ").title(), "count": v}
        for k, v in stats["regime_counts"].items()
    ])
    fig = px.bar(
        regime_df.sort_values("count"), x="count", y="regime", orientation="h",
        title="Nombre de produits par regime",
    )
    st.plotly_chart(fig, use_container_width=True)


def page_explorer(collection):
    st.header("Explorer les produits")

    query_text = st.text_input("Rechercher un produit par nom")

    t0 = time.time()
    results = search_products(collection, query_text, limit=30)
    duree = (time.time() - t0) * 1000

    if not results:
        st.info("Aucun produit trouve. Essayez un autre mot-cle.")
        return

    st.caption(f"{len(results)} resultat(s) en {duree:.0f} ms")

    table = pd.DataFrame([{
        "Produit": r.get("product_name"),
        "Marque": r.get("brands"),
        "Categorie": r.get("main_category"),
        "Nutri-Score": ((r.get("quality", {}) or {}).get("nutriscore_grade") or "?").upper(),
        "Energie (kcal/100g)": (r.get("nutrition", {}) or {}).get("energy_kcal"),
    } for r in results])
    st.dataframe(table, hide_index=True, use_container_width=True)

    names = [r.get("product_name", "N/A") for r in results]
    selected_name = st.selectbox("Voir le detail d'un produit", names)
    selected = next(r for r in results if r.get("product_name") == selected_name)

    st.divider()
    render_product_summary(selected)


def page_predict(collection, models):
    st.header("Predire la compatibilite avec un regime")
    st.caption("Classifieur multi-label (Random Forest) — predit 9 regimes simultanement a partir du profil nutritionnel.")

    mode = st.radio("Mode", ["A partir d'un produit existant", "Saisie manuelle"], horizontal=True)

    if mode == "A partir d'un produit existant":
        query_text = st.text_input("Rechercher un produit")
        if not query_text:
            return
        results = search_products(collection, query_text, limit=15)
        if not results:
            st.info("Aucun produit trouve.")
            return

        names = [r.get("product_name", "N/A") for r in results]
        selected_name = st.selectbox("Choisir un produit", names)
        product = next(r for r in results if r.get("product_name") == selected_name)

        render_product_summary(product)

        if st.button("Predire la compatibilite regime"):
            feature_values = {
                col: doc_to_feature_row(product, models["classifier_features"]).iloc[0][col]
                for col in models["classifier_features"]
            }
            t0 = time.time()
            results = predict_regime(models, feature_values)
            duree = (time.time() - t0) * 1000

            st.caption(f"Prediction en {duree:.0f} ms")
            render_regime_badges(results)

            regime_reel = product.get("regime", {}) or {}
            if regime_reel:
                st.divider()
                st.write("**Flags reels (calcules par regles dans le pipeline ETL) pour comparaison :**")
                comp_df = pd.DataFrame([{
                    "Regime": r["label"].replace("is_", "").replace("_", " ").title(),
                    "Predit": "Oui" if r["compatible"] else "Non",
                    "Reel": "Oui" if regime_reel.get(r["label"]) else "Non",
                    "Coherent": "Oui" if r["compatible"] == bool(regime_reel.get(r["label"])) else "Non",
                } for r in results])
                st.dataframe(comp_df, hide_index=True, use_container_width=True)

    else:
        with st.form("manual_form"):
            col1, col2, col3 = st.columns(3)
            energy = col1.number_input("Energie (kcal/100g)", 0.0, 900.0, 250.0)
            fat = col1.number_input("Matieres grasses (g/100g)", 0.0, 100.0, 10.0)
            saturated_fat = col1.number_input("dont satures (g/100g)", 0.0, 100.0, 3.0)
            carbs = col2.number_input("Glucides (g/100g)", 0.0, 100.0, 30.0)
            sugars = col2.number_input("dont sucres (g/100g)", 0.0, 100.0, 10.0)
            fiber = col2.number_input("Fibres (g/100g)", 0.0, 100.0, 3.0)
            proteins = col3.number_input("Proteines (g/100g)", 0.0, 100.0, 8.0)
            salt = col3.number_input("Sel (g/100g)", 0.0, 100.0, 1.0)
            sodium = col3.number_input("Sodium (g/100g)", 0.0, 40.0, 0.4)
            nova = st.select_slider(
                "Groupe NOVA (niveau de transformation)", options=[1, 2, 3, 4], value=3,
                help="1 = non transforme, 4 = ultra-transforme",
            )
            additives = st.number_input("Nombre d'additifs", 0, 30, 0)

            submitted = st.form_submit_button("Predire la compatibilite regime")

        if submitted:
            feature_values = {
                "nutrition.energy_kcal": energy,
                "nutrition.fat": fat,
                "nutrition.saturated_fat": saturated_fat,
                "nutrition.carbohydrates": carbs,
                "nutrition.sugars": sugars,
                "nutrition.fiber": fiber,
                "nutrition.proteins": proteins,
                "nutrition.salt": salt,
                "nutrition.sodium": sodium,
                "quality.nutriscore_score": np.nan,
                "quality.nova_group": nova,
                "meta.additives_count": additives,
            }
            t0 = time.time()
            results = predict_regime(models, feature_values)
            duree = (time.time() - t0) * 1000

            st.caption(f"Prediction en {duree:.0f} ms")
            render_regime_badges(results)


def page_alternatives(collection, models):
    st.header("Trouver des alternatives similaires")
    st.caption(
        "Clustering K-Means — cherche des produits du meme cluster nutritionnel, "
        "de la meme categorie, avec un Nutri-Score egal ou meilleur."
    )

    query_text = st.text_input("Rechercher un produit dont vous voulez une alternative")
    if not query_text:
        return

    results = search_products(collection, query_text, require_cluster=True, limit=15)
    if not results:
        st.info("Aucun produit clusterise trouve pour cette recherche. Avez-vous execute le notebook 4 ?")
        return

    names = [r.get("product_name", "N/A") for r in results]
    selected_name = st.selectbox("Choisir un produit", names)
    product = next(r for r in results if r.get("product_name") == selected_name)

    render_product_summary(product)

    only_healthier = st.checkbox("Uniquement des alternatives avec un Nutri-Score egal ou meilleur", value=True)
    n_results = st.slider("Nombre d'alternatives", 1, 10, 5)

    if st.button("Trouver des alternatives"):
        t0 = time.time()
        alternatives = find_alternatives(collection, models, product, n=n_results, only_healthier=only_healthier)
        duree = (time.time() - t0) * 1000

        st.caption(f"Recherche en {duree:.0f} ms")
        if alternatives.empty:
            st.warning("Aucune alternative trouvee avec ces criteres.")
        else:
            st.dataframe(alternatives, hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Application principale
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Foodiz", layout="wide")

    st.sidebar.title("Foodiz")
    st.sidebar.caption("Suivi de regime alimentaire — Open Food Facts")

    try:
        collection = get_collection()
    except ServerSelectionTimeoutError:
        st.error(
            "Impossible de se connecter a MongoDB sur "
            f"{MONGO_URI}. Verifiez que le serveur tourne et que les notebooks ETL ont ete executes."
        )
        st.stop()

    models = load_models()
    if models is None:
        st.sidebar.warning(
            "Modeles introuvables dans Models/. Executez les notebooks 3 et 4 avant d'utiliser "
            "les pages de prediction et de recommandation."
        )

    pages = {
        "Vue d'ensemble": "overview",
        "Explorer les produits": "explorer",
        "Predire un regime": "predict",
        "Trouver des alternatives": "alternatives",
    }
    page = st.sidebar.radio("Navigation", list(pages))

    if pages[page] == "overview":
        page_overview(collection)
    elif pages[page] == "explorer":
        page_explorer(collection)
    elif pages[page] == "predict":
        if models is None:
            st.warning("Le classifieur de regime n'est pas disponible. Executez le notebook 3.")
        else:
            page_predict(collection, models)
    elif pages[page] == "alternatives":
        if models is None:
            st.warning("Le modele de clustering n'est pas disponible. Executez le notebook 4.")
        else:
            page_alternatives(collection, models)


if __name__ == "__main__":
    main()
