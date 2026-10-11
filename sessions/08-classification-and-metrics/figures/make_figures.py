"""Make the figures of the Session 8 lecture notebooks.

Run from the repository root:
    uv run python sessions/08-classification-and-metrics/figures/make_figures.py
Deterministic: fixed seeds and the IBM Telco churn sample (downloaded). The resampling figure needs imbalanced-learn.
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


def cost_threshold():
    """Total cost of a retention campaign against the decision threshold (out-of-fold, training customers)."""
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    offer = 100
    oof = cross_val_predict(make_pipeline(pre, LogisticRegression(max_iter=1000)), X_train, y_train,
                            method="predict_proba", cv=StratifiedKFold(5, shuffle=True, random_state=0))[:, 1]
    ts = np.linspace(0.02, 0.9, 89)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for lost, c in [(200, AQUA), (400, BLUE), (800, ORANGE)]:
        costs = np.array([offer * (oof >= t).sum() + lost * ((oof < t) & (y_train == 1)).sum() for t in ts])
        costs = costs / len(y_train)
        k = costs.argmin()
        ax.plot(ts, costs, color=c, lw=2.2, label=f"lost customer €{lost}: best t = {ts[k]:.2f} (c / L = {offer / lost:.3f})")
        ax.plot(ts[k], costs[k], "o", color=c, ms=7)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.set(xlabel="decision threshold t (offer if predicted churn probability ≥ t; dashed: default 0.5)",
           ylabel="cost per customer (€)")
    ax.set_title("Retention offer €100: cost against threshold, logistic regression\n"
                 "(out-of-fold probabilities, 5,282 Telco training customers)", color=INK, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "cost_threshold.png")


# The two imbalance figures moved here from Session 9 together with the topic; they keep their own style.
YELLOW = "#eda100"
IMB_STYLE = {
    "font.size": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": False, "legend.frameon": False,
}


def smote_illustration():
    rng = np.random.default_rng(3)
    majority = rng.normal([0, 0], [1.3, 1.3], size=(160, 2))
    minority = rng.normal([4.2, 3.8], [1.0, 1.0], size=(14, 2))
    i = int(np.argmin(np.linalg.norm(minority - minority.mean(axis=0), axis=1)))
    d = np.linalg.norm(minority - minority[i], axis=1)
    nn = np.argsort(d)[1:6]
    synth = []
    for j in nn:
        lam = rng.uniform()
        synth.append(minority[i] + lam * (minority[j] - minority[i]))
    synth = np.array(synth)

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(*majority.T, s=22, color=GRID, edgecolor=MUTED, lw=0.4, label="majority class")
    ax.scatter(*minority.T, s=60, color=BLUE, edgecolor="white", lw=1.5, label="minority class", zorder=3)
    for j in nn:
        ax.plot(*np.c_[minority[i], minority[j]], color=BLUE, lw=1.2, ls="--", zorder=2)
    ax.scatter(*minority[i], s=150, color=BLUE, edgecolor=INK, lw=1.5, zorder=4)
    ax.annotate("x: chosen minority row", minority[i], xytext=(minority[i][0] - 4.5, minority[i][1] + 1.8),
                arrowprops={"arrowstyle": "->", "color": MUTED}, color=INK)
    ax.scatter(*synth.T, s=90, marker="D", color=ORANGE, edgecolor="white", lw=1.5, zorder=5,
               label=r"synthetic rows $x + \lambda\,(x_{nn} - x)$")
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_title("SMOTE: new rows on the lines to the 5 nearest minority neighbours", fontsize=11,
                 color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "smote_illustration.png", dpi=130)
    plt.close(fig)


def resampling_precision_recall():
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.pipeline import make_pipeline
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    telco = pd.read_csv(TELCO)
    telco["TotalCharges"] = pd.to_numeric(telco["TotalCharges"], errors="coerce").fillna(0)
    y = (telco["Churn"] == "Yes").astype(int)
    X = telco.drop(columns=["customerID", "Churn"])
    cat_cols = X.select_dtypes("object").columns.tolist()
    prep = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
                                   remainder=StandardScaler())
    cv = StratifiedKFold(5, shuffle=True, random_state=0)

    def lr(cw=None):
        return LogisticRegression(max_iter=2000, class_weight=cw)

    candidates = {
        "no correction": make_pipeline(prep, lr()),
        "undersampling": make_pipeline(prep, RandomUnderSampler(random_state=0), lr()),
        "oversampling": make_pipeline(prep, RandomOverSampler(random_state=0), lr()),
        "SMOTE": make_pipeline(prep, SMOTE(random_state=0), lr()),
        "class weights": make_pipeline(prep, lr("balanced")),
    }
    rows = []
    for name, model in candidates.items():
        proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        if name == "no correction":
            plain = proba
        pred = (proba >= 0.5).astype(int)
        rows.append((name, accuracy_score(y, pred), precision_score(y, pred), recall_score(y, pred), f1_score(y, pred)))
    pred = (plain >= 0.3).astype(int)
    rows.append(("no correction,\nthreshold 0.3", accuracy_score(y, pred), precision_score(y, pred),
                 recall_score(y, pred), f1_score(y, pred)))
    res = pd.DataFrame(rows, columns=["method", "accuracy", "precision", "recall", "f1"])

    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(res))
    width = 0.2
    series = [("accuracy", "accuracy", BLUE), ("precision", "precision (churn)", AQUA),
              ("recall", "recall (churn)", ORANGE), ("f1", "F1 (churn)", YELLOW)]
    for k, (col, label, color) in enumerate(series):
        bars = ax.bar(x + (k - 1.5) * width, res[col], width - 0.02, color=color, label=label)
        for bar, v in zip(bars, res[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7, color=MUTED)
    ax.set_xticks(x, res["method"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (5-fold cross-validation)")
    ax.set_title("Telco churn: corrections trade precision for recall, like a lower threshold",
                 fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(OUT / "resampling_precision_recall.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    cost_threshold()
    confusion()
    roc_pr()
    knn_boundaries()
    calibration()
    with plt.rc_context(IMB_STYLE):
        smote_illustration()
        resampling_precision_recall()
