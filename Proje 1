# =========================================================
# Churn-Analysis
# Customer churn analysis & prediction using Python.
# Scope: Data prep → EDA → Feature engineering → Segmentation
#        → Statistical tests → ML modeling/comparison → Prediction form
# Tech stack: Python, (optional: SQL), Streamlit, scikit-learn, XGBoost
# =========================================================

import pandas as pd
import streamlit as st
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product

# Statistical Analysis
import scipy.stats as stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests  # optional (not used below)

# Machine Learning
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Metrics
from sklearn.metrics import (
    precision_recall_curve,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

# =========================================================
# 1) DATA LOADING + BASIC CLEANING
# =========================================================

veri = pd.read_csv("data.csv", sep=';', encoding='latin5')

# Normalize column names
veri.columns = (
    veri.columns.str.strip()
        .str.upper()
        .str.replace(r"\s+", "_", regex=True)
)

# Rename (months -> years) as per your project
veri = veri.rename(columns={'SUBSCRIPTION_MONTHS': 'SUBSCRIPTION_YEARS'})

# Feature engineering (groups)
veri["SUBS_GROUP"] = pd.cut(
    veri["SUBSCRIPTION_YEARS"],
    bins=[0, 1, 3, 5, veri["SUBSCRIPTION_YEARS"].max()],
    labels=["0-1 yıl", "2-3 yıl", "4-5 yıl", "6+ yıl"],
    right=True,
    include_lowest=True
)

veri["AGE_GROUP"] = pd.cut(
    veri["AGE"],
    bins=[18, 25, 35, 45, 60, 120],
    labels=["18-25", "26-35", "36-45", "46-60", "61+"],
    right=True,
    include_lowest=True
)

# Quick look
print(veri.head())
print(veri.describe())
print(veri.info())

# Missing value analysis
print(veri.isnull().sum())
print((veri.isnull().sum() / len(veri)) * 100)

# Validate engineered variable
print(veri[["SUBSCRIPTION_YEARS", "SUBS_GROUP"]].head(15))
print(veri["SUBS_GROUP"])

# =========================================================
# 2) MODULE 2 — CHURN SEGMENTATION (SUMMARY TABLES)
# =========================================================

# Cohort groups (same bins)
veri["COHORT_GRUBU"] = pd.cut(
    veri["SUBSCRIPTION_YEARS"],
    bins=[0, 1, 3, 5, veri["SUBSCRIPTION_YEARS"].max()],
    labels=["0-1 yıl", "2-3 yıl", "4-5 yıl", "6+ yıl"],
    right=True,
    include_lowest=True
)

cohort = (
    veri.groupby("COHORT_GRUBU")["CHURN"]
        .agg(kisi_sayisi="count", churn_sayisi="sum")
        .reset_index()
)
cohort["ikna_sayisi"] = cohort["kisi_sayisi"] - cohort["churn_sayisi"]
cohort["churn_orani_%"] = (cohort["churn_sayisi"] / cohort["kisi_sayisi"] * 100).round(2)
cohort["ikna_orani_%"] = (cohort["ikna_sayisi"] / cohort["kisi_sayisi"] * 100).round(2)
print(cohort)

# CITY
city_summary = (
    veri.groupby("CITY")["CHURN"]
        .agg(kisi_sayisi="count", churn_sayisi="sum")
        .reset_index()
)
city_summary["ikna_sayisi"] = city_summary["kisi_sayisi"] - city_summary["churn_sayisi"]
city_summary["churn_orani_%"] = (city_summary["churn_sayisi"] / city_summary["kisi_sayisi"] * 100).round(2)
city_summary["ikna_orani_%"] = (city_summary["ikna_sayisi"] / city_summary["kisi_sayisi"] * 100).round(2)
print("\n=== CITY ===")
print(city_summary.nlargest(10, "kisi_sayisi"))

# PACKAGE
package_summary = (
    veri.groupby("PACKAGE")["CHURN"]
        .agg(kisi_sayisi="count", churn_sayisi="sum")
        .reset_index()
)
package_summary["ikna_sayisi"] = package_summary["kisi_sayisi"] - package_summary["churn_sayisi"]
package_summary["churn_orani_%"] = (package_summary["churn_sayisi"] / package_summary["kisi_sayisi"] * 100).round(2)
package_summary["ikna_orani_%"] = (package_summary["ikna_sayisi"] / package_summary["kisi_sayisi"] * 100).round(2)
print("\n=== PACKAGE ===")
print(package_summary.nlargest(10, "kisi_sayisi"))

# AGE_GROUP
age_summary = (
    veri.groupby("AGE_GROUP")["CHURN"]
        .agg(kisi_sayisi="count", churn_sayisi="sum")
        .reset_index()
)
age_summary["ikna_sayisi"] = age_summary["kisi_sayisi"] - age_summary["churn_sayisi"]
age_summary["churn_orani_%"] = (age_summary["churn_sayisi"] / age_summary["kisi_sayisi"] * 100).round(2)
age_summary["ikna_orani_%"] = (age_summary["ikna_sayisi"] / age_summary["kisi_sayisi"] * 100).round(2)
print("\n=== AGE_GROUP ===")
print(age_summary.nlargest(10, "kisi_sayisi"))

# GENDER
gender_summary = (
    veri.groupby("GENDER")["CHURN"]
        .agg(kisi_sayisi="count", churn_sayisi="sum")
        .reset_index()
)
gender_summary["ikna_sayisi"] = gender_summary["kisi_sayisi"] - gender_summary["churn_sayisi"]
gender_summary["churn_orani_%"] = (gender_summary["churn_sayisi"] / gender_summary["kisi_sayisi"] * 100).round(2)
gender_summary["ikna_orani_%"] = (gender_summary["ikna_sayisi"] / gender_summary["kisi_sayisi"] * 100).round(2)
print("\n=== GENDER ===")
print(gender_summary.nlargest(10, "kisi_sayisi"))

# REASON_CODE
reason_summary = (
    veri.groupby("REASON_CODE")["CHURN"]
        .agg(kisi_sayisi="count", churn_sayisi="sum")
        .reset_index()
)
reason_summary["ikna_sayisi"] = reason_summary["kisi_sayisi"] - reason_summary["churn_sayisi"]
reason_summary["churn_orani_%"] = (reason_summary["churn_sayisi"] / reason_summary["kisi_sayisi"] * 100).round(2)
reason_summary["ikna_orani_%"] = (reason_summary["ikna_sayisi"] / reason_summary["kisi_sayisi"] * 100).round(2)
print("\n=== REASON_CODE ===")
print(reason_summary.nlargest(10, "kisi_sayisi"))

# =========================================================
# 3) EARLY CHURN ANALYSIS (< 1 YEAR)
# =========================================================

erken_CHURN = veri[(veri["CHURN"] == 1) & (veri["SUBSCRIPTION_YEARS"] < 1)].copy()
print(f"\nErken CHURN sayısı: {len(erken_CHURN)}  | Oran: %{100*len(erken_CHURN)/len(veri):.2f}")

for col in ["SUBS_GROUP", "PACKAGE", "CITY", "AGE_GROUP", "GENDER", "REASON_CODE"]:
    if col in erken_CHURN.columns:
        t = (
            erken_CHURN.groupby(col)["ID"].count()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
                .head(6)
        )
        t["oran"] = 100 * t["count"] / len(erken_CHURN)
        print(f"\nErken CHURN – {col} dağılımı:")
        print(t)

# =========================================================
# 4) VISUALIZATION (STREAMLIT UI)
# =========================================================

kat_kol = ["AGE_GROUP", "SUBS_GROUP", "GENDER", "CITY", "PACKAGE", "INVERTION_APPLIED", "REASON_CODE"]

st.title("CHURN Analiz Görselleştirme")
st.caption("Menüden bir grafik tipi seç → butona bas → grafiği gör.")

plot_type = st.radio(
    "Grafik tipi",
    ["CHURN oranı", "Isı haritası"],
    horizontal=True
)

import os
os.makedirs("fig", exist_ok=True)

# CHURN rate chart
if plot_type == "CHURN oranı":
    col = st.selectbox("Kırılım kolonu", options=kat_kol)
    if st.button("Grafiği Göster"):
        stat = (
            veri.groupby(col)["CHURN"]
                .agg(sum="sum", mean="mean")
                .reset_index()
                .sort_values(by="sum", ascending=False)
                .head(7)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        order = stat[col].tolist()

        sns.barplot(x=col, y="mean", data=stat, order=order, ax=ax)

        for bar, (_, row) in zip(ax.patches, stat.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{int(row['sum'])} kişi\n%{row['mean']*100:.1f}",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_title(f"{col} bazında CHURN oranı")
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=30)
        plt.tight_layout()

        fname = f"fig/churn_orani_{col}.pdf"
        fig.savefig(fname, dpi=300, bbox_inches="tight")

        st.pyplot(fig)
        plt.clf()
        st.success(f"Kaydedildi: {fname}")

# Heatmap
elif plot_type == "Isı haritası":
    col1 = st.selectbox("Satır (index)", options=kat_kol, key="c1")
    col2 = st.selectbox("Sütun (columns)", options=kat_kol, key="c2")
    k = st.number_input("En yoğun k satır/sütun", 3, 20, 8, 1)

    if st.button("Isı Haritasını Göster"):
        seg = (
            veri.groupby([col1, col2])["CHURN"]
                .agg(rate="mean", n="size")
                .reset_index()
        )
        rows = seg.groupby(col1)["n"].sum().nlargest(k).index
        cols = seg.groupby(col2)["n"].sum().nlargest(k).index

        piv = (
            seg[seg[col1].isin(rows) & seg[col2].isin(cols)]
                .pivot(index=col1, columns=col2, values="rate")
        )
        annot = piv.applymap(lambda v: f"{v:.2%}" if pd.notnull(v) else "veri yok")

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.heatmap(
            piv.fillna(0.0),
            annot=annot,
            fmt="",
            vmin=0,
            vmax=1,
            cmap="coolwarm",
            linewidths=0.5,
            linecolor="white",
            ax=ax
        )
        ax.set_title(f"Segment Bazlı CHURN Isı Haritası (En yoğun {k}×{k})")

        fname = f"fig/isi_haritasi_{col1}_x_{col2}.pdf"
        fig.savefig(fname, dpi=300, bbox_inches="tight")

        st.pyplot(fig)
        plt.clf()
        st.success(f"Kaydedildi: {fname}")

# =========================================================
# 5) MODULE 3 — STATISTICAL TESTS
# =========================================================

# SUBSCRIPTION_YEARS: distribution check + Mann–Whitney U
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(veri["SUBSCRIPTION_YEARS"], bins=30, kde=True)
plt.title("Abonelik Süresi Histogram")
plt.subplot(1, 2, 2)
stats.probplot(veri["SUBSCRIPTION_YEARS"], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

print("Subscription skewness:", veri["SUBSCRIPTION_YEARS"].skew())

subs0 = pd.to_numeric(veri.loc[veri["CHURN"] == 0, "SUBSCRIPTION_YEARS"], errors="coerce").dropna().to_numpy(float)
subs1 = pd.to_numeric(veri.loc[veri["CHURN"] == 1, "SUBSCRIPTION_YEARS"], errors="coerce").dropna().to_numpy(float)

n0, n1 = subs0.size, subs1.size
U, p = mannwhitneyu(subs0, subs1, alternative="two-sided", method="asymptotic")
U_g, p_g = mannwhitneyu(subs0, subs1, alternative="greater", method="asymptotic")  # H1: subs0 > subs1
r_signed = 1 - 2 * U_g / (n0 * n1)

print(f"Mann–Whitney : U={U:.0f}, p={p:.3e}")
print(f"Rank-biserial  : r={r_signed:.3f}")
print(f"CHURN=0 Ortalama: {np.mean(subs0):.2f} | Medyan: {np.median(subs0):.2f}")
print(f"CHURN=1 Ortalama: {np.mean(subs1):.2f} | Medyan: {np.median(subs1):.2f}")

# AGE: Welch t-test + Hedges' g
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(veri["AGE"], bins=30, kde=True)
plt.title("Yaş Histogram")
plt.subplot(1, 2, 2)
stats.probplot(veri["AGE"], dist="norm", plot=plt)
plt.title("Q-Q Plot")
plt.show()

age0 = pd.to_numeric(veri.loc[veri["CHURN"] == 0, "AGE"], errors="coerce").dropna().to_numpy(float)
age1 = pd.to_numeric(veri.loc[veri["CHURN"] == 1, "AGE"], errors="coerce").dropna().to_numpy(float)

t_stat, p_val = ttest_ind(age0, age1, equal_var=False)

na, nb = len(age0), len(age1)
sa, sb = np.var(age0, ddof=1), np.var(age1, ddof=1)
sp = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
d = (np.mean(age0) - np.mean(age1)) / sp
J = 1 - 3 / (4 * (na + nb) - 9)
g = d * J

print(f"AGE Welch t-test: T={t_stat:.3f}, p={p_val:.3e}, Hedges' g={g:.3f}")

# Categorical: group rare levels + Chi-square + Cramér’s V
city_counts = veri["CITY"].value_counts()
rare_cities = city_counts[city_counts < 5].index
veri["CITY_GROUPED"] = veri["CITY"].replace(rare_cities, "Other")

reason_counts = veri["REASON_CODE"].value_counts()
rare_reasons = reason_counts[reason_counts < 5].index
veri["REASON_GROUPED"] = veri["REASON_CODE"].replace(rare_reasons, "Other")

kategorik_deg = ["GENDER", "PACKAGE", "CITY_GROUPED", "REASON_GROUPED"]

ozet_kayitlar = []
for col in kategorik_deg:
    ct = pd.crosstab(veri[col], veri["CHURN"])
    chi2, p, dof, expected = chi2_contingency(ct)

    n_tot = ct.values.sum()
    k_min = min(ct.shape)
    V = np.sqrt(chi2 / (n_tot * (k_min - 1))) if k_min > 1 else np.nan

    if V < 0.1:
        etki = "küçük"
    elif V < 0.3:
        etki = "orta"
    else:
        etki = "büyük"

    ozet_kayitlar.append({
        "Değişken": col,
        "Ki-kare": round(chi2, 2),
        "p-değeri": f"{p:.3e}",
        "d.f.": dof,
        "Cramér’s V": round(V, 3),
        "Etki": etki,
        "Yorum": "Anlamlı ilişki var" if p < 0.05 else "İlişki yok"
    })

ozet_df = pd.DataFrame(ozet_kayitlar)
print(ozet_df)

# =========================================================
# 6) MODULE 4 — MODELING & EVALUATION
# =========================================================

num_cols = ["AGE", "SUBSCRIPTION_YEARS"]
cat_cols = ["GENDER", "PACKAGE", "CITY_GROUPED", "REASON_GROUPED"]

X_num = veri[num_cols].copy()
X_cat = pd.get_dummies(veri[cat_cols], drop_first=True)

x = pd.concat([X_num, X_cat], axis=1)
y = veri["CHURN"].astype(int)

x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# 1) Logistic Regression (scaled)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train_raw)
x_test = scaler.transform(x_test_raw)

lr = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42).fit(x_train, y_train)
y_pred_lr = lr.predict(x_test)
y_prob_lr = lr.predict_proba(x_test)[:, 1]

# 2) Decision Tree
dt = DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42).fit(x_train_raw, y_train)
y_pred_dt = dt.predict(x_test_raw)
y_prob_dt = dt.predict_proba(x_test_raw)[:, 1]

# 3) Random Forest
rf = RandomForestClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    class_weight="balanced_subsample"
)
rf.fit(x_train_raw, y_train)
rf_pred = rf.predict(x_test_raw)
rf_proba = rf.predict_proba(x_test_raw)[:, 1]

# 4) XGBoost
pos_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
xgb = XGBClassifier(
    random_state=42,
    eval_metric="aucpr",
    learning_rate=0.08,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=pos_ratio
).fit(x_train_raw, y_train)

xgb_pred = xgb.predict(x_test_raw)
xgb_proba = xgb.predict_proba(x_test_raw)[:, 1]

# Model reports
print("=== Logistic Regression ===")
print("Train Acc:", accuracy_score(y_train, lr.predict(x_train)))
print("Test  Acc:", accuracy_score(y_test, y_pred_lr))
print("Test AUC :", roc_auc_score(y_test, y_prob_lr))
print()

print("=== Decision Tree ===")
print("Train Acc:", accuracy_score(y_train, dt.predict(x_train_raw)))
print("Test  Acc:", accuracy_score(y_test, y_pred_dt))
print("Test AUC :", roc_auc_score(y_test, y_prob_dt))
print()

print("=== Random Forest ===")
print("Train Acc:", accuracy_score(y_train, rf.predict(x_train_raw)))
print("Test  Acc:", accuracy_score(y_test, rf_pred))
print("Test AUC :", roc_auc_score(y_test, rf_proba))
print()

print("=== XGBoost ===")
print("Train Acc:", accuracy_score(y_train, xgb.predict(x_train_raw)))
print("Test  Acc:", accuracy_score(y_test, xgb_pred))
print("Test AUC :", roc_auc_score(y_test, xgb_proba))

# Feature importances
fi_lr = pd.DataFrame({
    "Feature": x.columns,
    "Importance": np.abs(lr.coef_[0])
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_lr, y="Feature", x="Importance", palette="Blues_r")
plt.title("Logistic Regression Feature Importance (Top 20)")
plt.show()

fi_dt = pd.DataFrame({
    "Feature": x.columns,
    "Importance": dt.feature_importances_
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_dt, y="Feature", x="Importance", palette="Greens_r")
plt.title("Decision Tree Feature Importance (Top 20)")
plt.show()

fi_rf = pd.DataFrame({
    "Feature": x.columns,
    "Importance": rf.feature_importances_
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_rf, y="Feature", x="Importance", palette="Purples_r")
plt.title("Random Forest Feature Importance (Top 20)")
plt.show()

fi_xgb = pd.DataFrame({
    "Feature": x.columns,
    "Importance": xgb.feature_importances_
}).sort_values("Importance", ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(data=fi_xgb, y="Feature", x="Importance", palette="Oranges_r")
plt.title("XGBoost Feature Importance (Top 20)")
plt.show()

# Comparison table
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_proba)

cmp = pd.DataFrame({
    "Accuracy":  [accuracy_score(y_test, y_pred_lr), accuracy_score(y_test, y_pred_dt),
                  accuracy_score(y_test, rf_pred),  accuracy_score(y_test, xgb_pred)],
    "Precision": [precision_score(y_test, y_pred_lr, zero_division=0),
                  precision_score(y_test, y_pred_dt, zero_division=0),
                  precision_score(y_test, rf_pred, zero_division=0),
                  precision_score(y_test, xgb_pred, zero_division=0)],
    "Recall":    [recall_score(y_test, y_pred_lr),
                  recall_score(y_test, y_pred_dt),
                  recall_score(y_test, rf_pred),
                  recall_score(y_test, xgb_pred)],
    "F1":        [f1_score(y_test, y_pred_lr),
                  f1_score(y_test, y_pred_dt),
                  f1_score(y_test, rf_pred),
                  f1_score(y_test, xgb_pred)],
    "ROC-AUC":   [auc(fpr_lr, tpr_lr), auc(fpr_dt, tpr_dt), auc(fpr_rf, tpr_rf), auc(fpr_xgb, tpr_xgb)]
}, index=["LogReg", "Decision Tree", "Random Forest", "XGBoost"]).round(3)

print("\n=== Performans Karşılaştırması ===\n", cmp)

# Confusion matrices
cms = [
    ("Logistic Regression", confusion_matrix(y_test, y_pred_lr)),
    ("Decision Tree",        confusion_matrix(y_test, y_pred_dt)),
    ("Random Forest",        confusion_matrix(y_test, rf_pred)),
    ("XGBoost",              confusion_matrix(y_test, xgb_pred)),
]

plt.rcParams.update({"font.size": 12})
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
names = np.array([["TN", "FP"], ["FN", "TP"]])

for ax, (name, cm) in zip(axes.ravel(), cms):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name} — Confusion Matrix", fontsize=13)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Tahmin 0", "Tahmin 1"], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Gerçek 0", "Gerçek 1"], fontsize=11)

    total = cm.sum()
    for (i, j) in product(range(2), range(2)):
        count = cm[i, j]
        pct = 100.0 * count / total if total else 0
        ax.text(
            j, i,
            f"{names[i, j]}\n{count}\n%{pct:.1f}",
            ha="center",
            va="center",
            fontsize=11,
            color="white" if count > cm.max() / 2 else "black"
        )

    ax.set_xlabel("Tahmin", fontsize=11)
    ax.set_ylabel("Gerçek", fontsize=11)

plt.tight_layout()
fig.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()

# ROC curves
plt.figure(figsize=(6, 5))
plt.plot(fpr_lr, tpr_lr, label=f"LR (AUC={auc(fpr_lr, tpr_lr):.3f})")
plt.plot(fpr_dt, tpr_dt, label=f"DT (AUC={auc(fpr_dt, tpr_dt):.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"RF (AUC={auc(fpr_rf, tpr_rf):.3f})")
plt.plot(fpr_xgb, tpr_xgb, label=f"XGB (AUC={auc(fpr_xgb, tpr_xgb):.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.legend()
plt.title("ROC Curves")
plt.show()

# Precision–Recall curves
prec_lr, rec_lr, _ = precision_recall_curve(y_test, y_prob_lr)
prec_dt, rec_dt, _ = precision_recall_curve(y_test, y_prob_dt)
prec_rf, rec_rf, _ = precision_recall_curve(y_test, rf_proba)
prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, xgb_proba)

plt.figure(figsize=(6, 5))
plt.plot(rec_lr, prec_lr, label="LR")
plt.plot(rec_dt, prec_dt, label="DT")
plt.plot(rec_rf, prec_rf, label="RF")
plt.plot(rec_xgb, prec_xgb, label="XGB")
plt.legend()
plt.title("Precision–Recall Curves")
plt.show()

# =========================================================
# 7) MODULE 5 — NEW CUSTOMER CHURN PREDICTION (STREAMLIT FORM)
# =========================================================

st.subheader("Müşteri Churn Tahmini")

with st.form("new_customer_full"):
    age     = st.text_input("Yaş", value="")
    subs    = st.text_input("Abonelik Süresi (yıl)", value="")
    gender  = st.selectbox("Cinsiyet", ["(boş bırak)"] + sorted(veri["GENDER"].dropna().unique().tolist()))
    city    = st.selectbox("Şehir (CITY_GROUPED)", ["(boş bırak)"] + sorted(veri["CITY_GROUPED"].dropna().unique().tolist()))
    package = st.selectbox("Paket", ["(boş bırak)"] + sorted(veri["PACKAGE"].dropna().unique().tolist()))
    reason  = st.selectbox("Reason (REASON_GROUPED)", ["(boş bırak)"] + sorted(veri["REASON_GROUPED"].dropna().unique().tolist()))
    submit  = st.form_submit_button("Tahmin Et")

if submit:
    def to_float(s):
        s = str(s).strip().replace(",", ".")
        try:
            return float(s) if s != "" else np.nan
        except:
            return np.nan

    age_val  = to_float(age)
    subs_val = to_float(subs)

    gender_val  = np.nan if gender  == "(boş bırak)" else gender
    city_val    = np.nan if city    == "(boş bırak)" else city
    package_val = np.nan if package == "(boş bırak)" else package
    reason_val  = np.nan if reason  == "(boş bırak)" else reason

    new_cust = pd.DataFrame({
        "AGE": [age_val],
        "SUBSCRIPTION_YEARS": [subs_val],
        "GENDER": [gender_val],
        "CITY_GROUPED": [city_val],
        "PACKAGE": [package_val],
        "REASON_GROUPED": [reason_val],
    })

    age_mu  = float(veri["AGE"].mean(skipna=True))
    subs_mu = float(veri["SUBSCRIPTION_YEARS"].mean(skipna=True))
    new_cust["AGE"] = new_cust["AGE"].fillna(age_mu)
    new_cust["SUBSCRIPTION_YEARS"] = new_cust["SUBSCRIPTION_YEARS"].fillna(subs_mu)

    new_enc = pd.get_dummies(new_cust, drop_first=True)
    new_enc = new_enc.reindex(columns=x.columns, fill_value=0)
    new_enc = new_enc.astype(float)

    p_lr  = lr.predict_proba(scaler.transform(new_enc))[:, 1][0]
    p_dt  = dt.predict_proba(new_enc)[:, 1][0]
    p_rf  = rf.predict_proba(new_enc)[:, 1][0]
    p_xgb = xgb.predict_proba(new_enc)[:, 1][0]

    st.table(pd.DataFrame([
        ["LogReg",       int(p_lr  >= 0.5), round(p_lr,  3)],
        ["DecisionTree", int(p_dt  >= 0.5), round(p_dt,  3)],
        ["RandomForest", int(p_rf  >= 0.5), round(p_rf,  3)],
        ["XGBoost",      int(p_xgb >= 0.5), round(p_xgb, 3)],
    ], columns=["Model", "Tahmin (0=Yok,1=Var)", "Churn Olasılığı"]))
