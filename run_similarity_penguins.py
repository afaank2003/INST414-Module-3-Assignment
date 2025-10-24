#!/usr/bin/env python3
# Robust MVP for Penguins LTER or tidy penguins CSV.
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_PATH = "penguins_lter.csv"   # or a tidy penguins CSV
ROOT = "mvp_penguins_similarity"

def load_and_prepare(path):
    raw = pd.read_csv(path)
    colmap_lter = {
        "Species": "species",
        "Culmen Length (mm)": "bill_length_mm",
        "Culmen Depth (mm)": "bill_depth_mm",
        "Flipper Length (mm)": "flipper_length_mm",
        "Body Mass (g)": "body_mass_g",
    }
    tidy = {"species","bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"}
    if tidy.issubset(raw.columns):
        df = raw[list(tidy)].copy()
    else:
        if not set(colmap_lter.keys()).issubset(raw.columns):
            raise ValueError("CSV does not have expected LTER or tidy column names.")
        df = raw[list(colmap_lter.keys())].rename(columns=colmap_lter).copy()
    df = df.dropna(subset=list(tidy)).reset_index(drop=True)
    df["id"] = [f"Penguin_{i}" for i in range(len(df))]
    df["class_name"] = df["species"].astype(str)
    return df

def main():
    os.makedirs(f"{ROOT}/outputs", exist_ok=True)
    os.makedirs(f"{ROOT}/data", exist_ok=True)

    df = load_and_prepare(DATA_PATH)
    df.to_csv(f"{ROOT}/data/penguins_clean.csv", index=False)

    FEATURES = ["bill_length_mm","bill_depth_mm","flipper_length_mm","body_mass_g"]
    X = df[FEATURES].values
    Xz = StandardScaler().fit_transform(X)
    Xn = Xz / np.clip(np.linalg.norm(Xz, axis=1, keepdims=True), 1e-12, None)
    S = Xn @ Xn.T

    queries = []
    for s in df["class_name"].unique():
        idxs = df.index[df["class_name"] == s]
        if len(idxs) > 0:
            queries.append(int(idxs[0]))
        if len(queries) == 3:
            break
    if len(queries) < 3:
        for i in range(len(df)):
            if i not in queries:
                queries.append(i)
            if len(queries) == 3:
                break

    def top_k(i, k=10):
        sims = S[i]
        order = np.argsort(-sims)
        keep = [j for j in order if j != i][:k]
        out = df.iloc[keep][["id", "class_name"]].copy()
        out["similarity"] = sims[keep]
        out = out.reset_index().rename(columns={"index": "row_index"})
        return out

    tops = []
    for qi in queries:
        qrow = df.loc[qi, ["id", "class_name"]]
        sub = top_k(qi, k=10)
        sub.insert(0, "query_id", qrow["id"])
        sub.insert(1, "query_class", qrow["class_name"])
        tops.append(sub)

    combined = pd.concat(tops, ignore_index=True)
    combined["similarity"] = combined["similarity"].round(4)
    combined.to_csv(f"{ROOT}/outputs/top10_all_queries.csv", index=False)
    for qi in queries:
        qid = df.loc[qi, "id"]
        combined[combined["query_id"] == qid].to_csv(f"{ROOT}/outputs/top10_{qid}.csv", index=False)

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(Xz)
    df["pc1"], df["pc2"] = X2[:, 0], X2[:, 1]

    plt.figure()
    plt.scatter(df["pc1"], df["pc2"], alpha=0.6, s=20)
    qmask = df.index.isin(queries)
    plt.scatter(df.loc[qmask, "pc1"], df.loc[qmask, "pc2"], s=100, marker='*')
    for _, r in df.loc[qmask, ["pc1", "pc2", "id"]].iterrows():
        plt.text(r["pc1"], r["pc2"], r["id"], fontsize=8)
    plt.title("PCA of Penguin Feature Space")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.tight_layout(); plt.savefig(f"{ROOT}/outputs/pca_scatter.png", dpi=200); plt.close()

    first_q = queries[0]
    sims = S[first_q]
    neighbor_idx = np.argsort(-sims)[1]
    pair = pd.DataFrame({ "feature": FEATURES,
                          "query": df.loc[first_q, FEATURES].values,
                          "neighbor": df.loc[neighbor_idx, FEATURES].values }).set_index("feature")
    pair.plot(kind="bar", figsize=(10,4))
    plt.title(f"Feature Profile: {df.loc[first_q,'id']} vs Nearest Neighbor ({df.loc[neighbor_idx,'id']})")
    plt.tight_layout(); plt.savefig(f"{ROOT}/outputs/feature_bars.png", dpi=200); plt.close()

    val_rows = []
    for qi in queries:
        q_class = df.loc[qi, "class_name"]
        sub = combined[combined["query_id"] == df.loc[qi, "id"]]
        same_share = float((sub["class_name"] == q_class).mean())
        val_rows.append({"query_id": df.loc[qi, "id"],
                         "query_class": q_class,
                         "share_same_class_in_top10": round(same_share, 2)})
    pd.DataFrame(val_rows).to_csv(f"{ROOT}/outputs/validation_summary.csv", index=False)

if __name__ == "__main__":
    main()
