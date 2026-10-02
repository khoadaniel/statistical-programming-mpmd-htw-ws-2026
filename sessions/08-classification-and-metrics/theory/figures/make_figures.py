"""Make the figures of the Session 8 theory pages.

Run from the repository root:
    uv run python sessions/08-classification-and-metrics/theory/figures/make_figures.py
Deterministic: fixed seeds, the IBM Telco churn sample (downloaded) and the EBTI case-study
sample in case-study/data/ (coverage_accuracy.png).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

OUT = Path(__file__).resolve().parent
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TELCO = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


telco = pd.read_csv(TELCO)
telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce")
y = (telco["Churn"] == "Yes").astype(int)
X = telco.drop(columns=["customerID", "Churn"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
numeric = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
categorical = [c for c in X.columns if c not in numeric]
pre = ColumnTransformer([
    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
])
logreg = make_pipeline(pre, LogisticRegression(max_iter=1000)).fit(X_train, y_train)
knn = make_pipeline(pre, KNeighborsClassifier(n_neighbors=50)).fit(X_train, y_train)


def confusion():
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_estimator(logreg, X_test, y_test, display_labels=["stays (0)", "churns (1)"],
                                          cmap="Blues", colorbar=False, ax=ax)
    ax.grid(False)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Logistic regression, threshold 0.5\n1,761 test customers", color=INK, loc="left")
    for t, lab in zip(ax.texts, ["TN", "FP", "FN", "TP"]):
        t.set_text(f"{lab}\n{t.get_text()}")
    save(fig, "confusion_matrix.png")


def roc_pr():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    base = y_test.mean()
    for name, m, c in [("logistic regression", logreg, BLUE), ("50-nearest neighbours", knn, ORANGE)]:
        p = m.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        axes[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {roc_auc_score(y_test, p):.3f})")
        prec, rec, _ = precision_recall_curve(y_test, p)
        axes[1].plot(rec, prec, color=c, lw=2, label=f"{name} (AP {average_precision_score(y_test, p):.3f})")
    p = logreg.predict_proba(X_test)[:, 1]
    pred = p >= 0.5
    tpr05 = (pred & (y_test == 1)).sum() / (y_test == 1).sum()
    fpr05 = (pred & (y_test == 0)).sum() / (y_test == 0).sum()
    prec05 = (pred & (y_test == 1)).sum() / pred.sum()
    axes[0].plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="random ranking (AUC 0.5)")
    axes[0].plot(fpr05, tpr05, "o", ms=8, color=INK)
    axes[0].annotate("threshold 0.5", (fpr05, tpr05), (fpr05 + 0.08, tpr05 - 0.12), color=INK,
                     arrowprops={"arrowstyle": "-", "color": MUTED})
    axes[1].axhline(base, color=MUTED, ls="--", lw=1, label=f"random ranking (churn rate {base:.3f})")
    axes[1].plot(tpr05, prec05, "o", ms=8, color=INK)
    axes[1].annotate("threshold 0.5", (tpr05, prec05), (tpr05 + 0.05, prec05 + 0.15), color=INK,
                     arrowprops={"arrowstyle": "-", "color": MUTED})
    axes[0].set(xlabel="false positive rate  FP / (FP + TN)", ylabel="true positive rate (recall)",
                xlim=(0, 1), ylim=(0, 1.01))
    axes[1].set(xlabel="recall  TP / (TP + FN)", ylabel="precision  TP / (TP + FP)", xlim=(0, 1), ylim=(0, 1.01))
    axes[0].set_title("ROC curve", color=INK, loc="left")
    axes[1].set_title("Precision–recall curve", color=INK, loc="left")
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    save(fig, "roc_pr_curves.png")


def knn_boundaries():
    cols = ["tenure", "MonthlyCharges"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_train), 600, replace=False)
    Xs, ys = X_train[cols].iloc[idx], y_train.iloc[idx]
    xx, yy = np.meshgrid(np.linspace(0, 73, 300), np.linspace(15, 122, 300))
    grid = pd.DataFrame({"tenure": xx.ravel(), "MonthlyCharges": yy.ravel()})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    cmap = ListedColormap(["#d6e6f8", "#fbdccd"])
    for ax, k in zip(axes, [1, 15, 150]):
        m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(Xs, ys)
        z = m.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], cmap=cmap)
        for cls, c, lab in [(0, BLUE, "stays"), (1, ORANGE, "churns")]:
            s = ys == cls
            ax.scatter(Xs["tenure"][s], Xs["MonthlyCharges"][s], s=10, color=c, edgecolor=SURFACE,
                       linewidth=0.4, label=lab)
        acc_tr = m.score(Xs, ys)
        acc_te = m.score(X_test[cols], y_test)
        ax.set_title(f"k = {k}: train acc. {acc_tr:.2f}, test acc. {acc_te:.2f}", color=INK, loc="left")
        ax.set_xlabel("tenure (months)")
        ax.grid(False)
    axes[0].set_ylabel("monthly charges ($)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"true class: {lab}" for lab in labels], loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.14), title="dots: true class; shaded areas: predicted class", markerscale=2)
    fig.suptitle("k-nearest neighbours on two Telco features (600 training customers)", x=0.06, y=1.03,
                 ha="left", color=INK)
    save(fig, "knn_boundaries.png")


def calibration():
    nb = make_pipeline(pre, GaussianNB()).fit(X_train, y_train)
    nb_cal = CalibratedClassifierCV(make_pipeline(pre, GaussianNB()), method="isotonic", cv=5).fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1, label="perfectly calibrated")
    for name, m, c in [("logistic regression", logreg, BLUE), ("naive Bayes", nb, ORANGE),
                       ("naive Bayes, recalibrated", nb_cal, AQUA)]:
        frac, mean_p = calibration_curve(y_test, m.predict_proba(X_test)[:, 1], n_bins=10, strategy="quantile")
        ax.plot(mean_p, frac, marker="o", ms=5, lw=2, color=c, label=name)
    ax.set(xlabel="mean predicted probability of churn (per bin)", ylabel="observed share of churners",
           xlim=(0, 1), ylim=(0, 1))
    ax.set_title("Calibration curves, Telco test customers (10 bins)", color=INK, loc="left")
    ax.legend(loc="upper left")
    save(fig, "calibration_curve.png")


def coverage_accuracy():
    """Abstention on the EBTI decisions: accept a predicted heading only above a confidence threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier

    decisions = pd.read_parquet(Path("case-study/data/train_sample.parquet"))
    past = decisions["start_date"].dt.year <= 2021
    clf = make_pipeline(TfidfVectorizer(min_df=2, sublinear_tf=True),
                        SGDClassifier(loss="log_loss", alpha=1e-6, random_state=0, n_jobs=-1))
    clf.fit(decisions.loc[past, "description"], decisions.loc[past, "heading"])
    proba = clf.predict_proba(decisions.loc[~past, "description"])
    correct = clf.classes_[proba.argmax(axis=1)] == decisions.loc[~past, "heading"].to_numpy()
    conf = proba.max(axis=1)
    ts = np.linspace(0, 0.95, 96)              # above 0.95 almost no case is left
    cov = np.array([(conf >= t).mean() for t in ts])
    acc = np.array([correct[conf >= t].mean() for t in ts])
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(cov, acc, color=BLUE, lw=2.5)
    for t in (0.0, 0.3, 0.5, 0.7, 0.9):
        k = conf >= t
        ax.plot(k.mean(), correct[k].mean(), "o", color=ORANGE, ms=6)
        ax.annotate(f"t = {t:.1f}", (k.mean(), correct[k].mean()), (8, -14), textcoords="offset points",
                    color=INK, fontsize=9)
    ax.set(xlabel="coverage: share of decisions the model decides itself",
           ylabel="accuracy on the decided cases", xlim=(0, 1.02), ylim=(0.7, 1.01))
    ax.set_title("Abstention: TF-IDF heading classifier, EBTI 2022–2023\n"
                 "(trained on 2017–2021 of the 50,000-decision sample)", color=INK, loc="left")
    save(fig, "coverage_accuracy.png")


if __name__ == "__main__":
    coverage_accuracy()
    confusion()
    roc_pr()
    knn_boundaries()
    calibration()
