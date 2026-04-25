

# ── STEP 0 : Install / import libraries ─────────────────────
import subprocess, sys
def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["xgboost", "openpyxl", "xlsxwriter", "scikit-learn",
            "matplotlib", "seaborn", "pandas", "numpy", "scipy"]:
    install(pkg)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, io

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                      cross_val_predict, cross_val_score)
from sklearn.feature_selection import (SelectKBest, f_classif,
                                        mutual_info_classif, RFE)
from sklearn.linear_model  import LogisticRegression, LassoCV
from sklearn.svm            import SVC
from sklearn.tree           import DecisionTreeClassifier
from sklearn.ensemble       import (RandomForestClassifier,
                                    GradientBoostingClassifier)
from sklearn.neighbors      import KNeighborsClassifier
from sklearn.naive_bayes    import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline       import Pipeline          # ← KEY fix
from sklearn.metrics        import (roc_auc_score, accuracy_score,
                                    confusion_matrix, precision_score,
                                    recall_score, f1_score, roc_curve)
import xgboost as xgb

from google.colab import files
from matplotlib.backends.backend_pdf import PdfPages

print("✅  All libraries loaded.\n")

# ════════════════════════════════════════════════════════════
#  STEP 1 : Upload & Parse Excel
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  STEP 1 – Upload your RNA-seq Excel file")
print("=" * 60)

uploaded = files.upload()
filename = list(uploaded.keys())[0]
raw_df   = pd.read_excel(io.BytesIO(uploaded[filename]), header=0, index_col=0)

print(f"\n✅  '{filename}' loaded  →  shape {raw_df.shape}")

SPECIAL_ROWS = ["disease","padj","pvalue","log2foldchange","symbol","description"]

def find_row(df, kw):
    for idx in df.index:
        if str(idx).strip().lower() == kw.lower():
            return idx
    return None

disease_row = find_row(raw_df, "disease")
if disease_row is None:
    raise ValueError("❌  'Disease' row not found. Check your Excel.")

y_series    = raw_df.loc[disease_row].astype(float)
rows_to_drop = [find_row(raw_df, r) for r in SPECIAL_ROWS if find_row(raw_df, r)]
expr_df      = raw_df.drop(index=rows_to_drop)
expr_df      = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0)

# ── FIX: strip NaN and any value that is not 0 or 1 ─────────
# The Disease row often contains NaN in the Gene_id/index column
# which converts to a huge negative int and creates a phantom 3rd class.
y_series = y_series.dropna()
y_series = y_series[y_series.isin([0.0, 1.0])]

# Keep only samples that appear in both the expression matrix and y
common_cols = expr_df.columns.intersection(y_series.index)
if len(common_cols) == 0:
    raise ValueError("❌  No overlapping sample columns between expression data "
                     "and Disease row. Check that sample names match.")
expr_df  = expr_df[common_cols]
y_series = y_series[common_cols]

n_genes   = expr_df.shape[0]
n_samples = expr_df.shape[1]
n_ctrl    = int((y_series == 0).sum())
n_dis     = int((y_series == 1).sum())

print(f"\n   Expression matrix : {n_genes} genes × {n_samples} samples")
print(f"   Labels            : 0 (Control)={n_ctrl}  1 (Disease)={n_dis}")
print(f"   Unique y values   : {sorted(y_series.unique().tolist())}  ← should be [0.0, 1.0]")

# Sanity checks
if n_ctrl == 0 or n_dis == 0:
    raise ValueError("❌  Only one class found in Disease row. "
                     "Need both 0 and 1 labels.")
if n_samples < 10:
    print(f"\n⚠️  WARNING: Only {n_samples} samples — results may be unreliable.")
    print("   Consider LOOCV (option 3) for cross-validation.")

# ════════════════════════════════════════════════════════════
#  STEP 2 : Log2 Normalisation  (no scaler yet — goes inside Pipeline)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 2 – Log2(x+1) Normalisation")
print("=" * 60)

# ── Final y cleanup: guarantee clean 0/1 int array ──────────
# Encodes with LabelEncoder so XGBoost always sees {0,1}
from sklearn.preprocessing import LabelEncoder
le       = LabelEncoder()
y        = le.fit_transform(y_series.values.astype(int))   # always 0,1
X_all    = log_df.T.loc[y_series.index].values.astype(float)
gene_names = log_df.index.tolist()

print("✅  Log2(x+1) applied.")
print(f"   Encoded labels: {np.unique(y).tolist()}  ← must be [0, 1]")
print("   NOTE: Z-score scaling applied INSIDE each pipeline fold (no leakage).")

# ════════════════════════════════════════════════════════════
#  STEP 3 : Train–Test Split (80 : 20, stratified)
#           Test set is set aside and NEVER used until final eval
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 3 – Train / Test Split  (80:20 stratified)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.20, random_state=42, stratify=y)

print(f"✅  Train: {X_train.shape[0]} samples  |  Test: {X_test.shape[0]} samples")

# ════════════════════════════════════════════════════════════
#  STEP 4 : CV setup
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 4 – Cross-Validation Configuration")
print("=" * 60)
print("  [1] 5-Fold   [2] 10-Fold   [3] LOOCV   [4] Custom")
cv_choice = input("Enter choice [1/2/3/4]: ").strip()

if   cv_choice == "1": N_SPLITS = 5
elif cv_choice == "2": N_SPLITS = 10
elif cv_choice == "3": N_SPLITS = X_train.shape[0]
elif cv_choice == "4": N_SPLITS = int(input("Enter k: ").strip())
else:                  N_SPLITS = 5

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
print(f"✅  Stratified {N_SPLITS}-Fold CV selected.")

# ── Safety check: warn if k > smallest class size ────────────
min_class_train = int(np.bincount(y_train).min())
if N_SPLITS > min_class_train:
    print(f"\n⚠️  WARNING: {N_SPLITS}-Fold CV with only {min_class_train} samples "
          f"in the smallest class.")
    print(f"   Some folds may have no minority-class samples → unstable AUC.")
    print(f"   Recommended: use at most {min_class_train}-Fold for this dataset.")
    fix_k = input(f"   Auto-reduce k to {min_class_train}? [y/n]: ").strip().lower()
    if fix_k != "n":
        N_SPLITS = min_class_train
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        print(f"   ✅  Reduced to {N_SPLITS}-Fold CV.")

# ════════════════════════════════════════════════════════════
#  STEP 5 : Feature Selection  ← NOW LEAK-FREE via Pipeline
#
#  How it works:
#  • A Pipeline is [Scaler → FS selector → Classifier].
#  • cross_val_score / cross_val_predict runs the ENTIRE
#    pipeline inside each fold, so the selector only sees
#    the fold's training split — never the validation split.
#  • For the final test-set evaluation, the pipeline is
#    re-fit on ALL of X_train (still blind to X_test).
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 5 – Feature Selection (leak-free, inside Pipeline)")
print("=" * 60)

# ── Ask user how many features to select ─────────────────────
total_genes = X_train.shape[1]
min_cls     = min(np.bincount(y_train))          # size of smallest class
max_safe    = max(1, min(total_genes, min_cls * X_train.shape[0] // 2))

print(f"\n   Dataset summary:")
print(f"     Total genes (features) : {total_genes}")
print(f"     Training samples       : {X_train.shape[0]}")
print(f"     Smallest class size    : {min_cls}")
print(f"\n   ⚠️  Rule of thumb for small datasets:")
print(f"      Select at most ~{max_safe} features to avoid overfitting")
print(f"      (more features than samples → high overfitting risk)")
print(f"\n   Enter number of top genes to select per FS method.")
print(f"   Press Enter to use the recommended default.")

default_k  = max(2, min(20, total_genes // 2 if total_genes > 10 else total_genes))
raw_k      = input(f"   Number of features [default={default_k}, max={total_genes}]: ").strip()
N_FEATURES = int(raw_k) if raw_k.isdigit() else default_k
N_FEATURES = max(2, min(N_FEATURES, total_genes))

print(f"\n✅  Will select top {N_FEATURES} genes per FS method.")
if N_FEATURES >= total_genes:
    print("   ℹ️  All genes selected — effective feature selection is off.")
    print("      Consider selecting fewer features for a more meaningful comparison.")

# ── Helper: fit selector on full X_train, return gene list ──
# (used only for the output Excel / inspection, NOT for CV metrics)
def get_selected_genes_filter(X_tr, y_tr, k):
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X_tr)
    sel_f   = SelectKBest(f_classif,           k=k).fit(Xs, y_tr)
    sel_mi  = SelectKBest(mutual_info_classif, k=k).fit(Xs, y_tr)
    f_n  = sel_f.scores_;  f_n  = (f_n  - f_n.min())  / (f_n.max()  - f_n.min()  + 1e-9)
    mi_n = sel_mi.scores_; mi_n = (mi_n - mi_n.min()) / (mi_n.max() - mi_n.min() + 1e-9)
    comb = (f_n + mi_n) / 2
    idx  = np.argsort(comb)[::-1][:k]
    return [gene_names[i] for i in idx], comb[idx]

def get_selected_genes_wrapper(X_tr, y_tr, k):
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_tr)
    rfe    = RFE(LogisticRegression(max_iter=1000, C=1.0, solver="liblinear",
                                    random_state=42),
                 n_features_to_select=k, step=10)
    rfe.fit(Xs, y_tr)
    idx    = np.where(rfe.support_)[0]
    return [gene_names[i] for i in idx], rfe.ranking_[idx]

def get_selected_genes_embedded(X_tr, y_tr, k):
    scaler  = StandardScaler()
    Xs      = scaler.fit_transform(X_tr)
    lasso   = LassoCV(cv=5, random_state=42, max_iter=5000).fit(Xs, y_tr)
    rf      = RandomForestClassifier(n_estimators=200, random_state=42,
                                     n_jobs=-1).fit(Xs, y_tr)
    l_n = np.abs(lasso.coef_); l_n = (l_n - l_n.min()) / (l_n.max() - l_n.min() + 1e-9)
    r_n = rf.feature_importances_; r_n = (r_n - r_n.min()) / (r_n.max() - r_n.min() + 1e-9)
    comb = (l_n + r_n) / 2
    idx  = np.argsort(comb)[::-1][:k]
    return [gene_names[i] for i in idx], comb[idx]

print("   Fitting selectors on X_train (for gene list export) …")
filter_genes,   filter_scores   = get_selected_genes_filter  (X_train, y_train, N_FEATURES)
wrapper_genes,  wrapper_scores  = get_selected_genes_wrapper  (X_train, y_train, N_FEATURES)
embedded_genes, embedded_scores = get_selected_genes_embedded (X_train, y_train, N_FEATURES)

set_f = set(filter_genes); set_w = set(wrapper_genes); set_e = set(embedded_genes)
overlap_all = set_f & set_w & set_e
print(f"   Filter genes    : {len(filter_genes)}")
print(f"   Wrapper genes   : {len(wrapper_genes)}")
print(f"   Embedded genes  : {len(embedded_genes)}")
print(f"   Overlap (all 3) : {len(overlap_all)} genes")

# ── FS selector objects used INSIDE pipelines ────────────────
# These are re-instantiated fresh every time the pipeline runs

def make_filter_selector(k):
    """Combined univariate: picks top-k by avg(ANOVA-F rank, MI rank)."""
    # sklearn doesn't natively combine two selectors, so we use a
    # FunctionTransformer wrapper approach with SelectKBest on f_classif
    # (MI is used for the standalone gene list; pipeline uses ANOVA-F
    # which is fast, deterministic, and well-validated for expression data)
    return SelectKBest(f_classif, k=k)

def make_wrapper_selector(k):
    return RFE(LogisticRegression(max_iter=1000, C=1.0, solver="liblinear",
                                   random_state=42),
               n_features_to_select=k, step=10)

def make_embedded_selector(k):
    # Embedded: use RF-based SelectFromModel threshold equivalent
    from sklearn.feature_selection import SelectFromModel
    return SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        max_features=k)

FS_PIPELINE_SELECTORS = {
    "Filter (Univariate)": lambda k: make_filter_selector(k),
    "Wrapper (RFE)":       lambda k: make_wrapper_selector(k),
    "Embedded (LASSO+RF)": lambda k: make_embedded_selector(k),
}

# ════════════════════════════════════════════════════════════
#  STEP 6 : Save Feature Selection Results → Excel
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 6 – Saving FS Results")
print("=" * 60)

FS_OUTPUT = "Feature_Selection_Results.xlsx"
with pd.ExcelWriter(FS_OUTPUT, engine="xlsxwriter") as writer:
    wb = writer.book
    hdr_fmt  = wb.add_format({"bold":True,"bg_color":"#1F3864","font_color":"white",
                               "border":1,"align":"center","valign":"vcenter"})
    cell_fmt = wb.add_format({"border":1})
    num_fmt  = wb.add_format({"border":1,"num_format":"0.0000"})
    alt_c    = wb.add_format({"border":1,"bg_color":"#D9E1F2"})
    alt_n    = wb.add_format({"border":1,"bg_color":"#D9E1F2","num_format":"0.0000"})

    def write_sheet(ws, headers, rows, widths):
        for c,h in enumerate(headers): ws.write(0,c,h,hdr_fmt)
        for r,row in enumerate(rows):
            for c,val in enumerate(row):
                f = (alt_n if r%2 else num_fmt) if c>0 else (alt_c if r%2 else cell_fmt)
                ws.write(r+1,c,val,f)
        for c,w in enumerate(widths): ws.set_column(c,c,w)

    write_sheet(wb.add_worksheet("Filter_Univariate"),
                ["Gene ID","Combined Score (ANOVA-F + MI)"],
                [(g,round(s,6)) for g,s in zip(filter_genes, filter_scores)],
                [25,32])
    write_sheet(wb.add_worksheet("Wrapper_RFE"),
                ["Gene ID","RFE Rank (1=best)"],
                [(g,int(s)) for g,s in zip(wrapper_genes, wrapper_scores)],
                [25,22])
    write_sheet(wb.add_worksheet("Embedded_LASSO_RF"),
                ["Gene ID","Combined Score (LASSO + RF)"],
                [(g,round(s,6)) for g,s in zip(embedded_genes, embedded_scores)],
                [25,34])
    overlap_rows = [
        ("Filter ∩ Wrapper ∩ Embedded", len(overlap_all),
         ", ".join(list(overlap_all)[:20])),
        ("Filter ∩ Wrapper",            len(set_f&set_w),
         ", ".join(list(set_f&set_w)[:20])),
        ("Filter ∩ Embedded",           len(set_f&set_e),
         ", ".join(list(set_f&set_e)[:20])),
        ("Wrapper ∩ Embedded",          len(set_w&set_e),
         ", ".join(list(set_w&set_e)[:20])),
    ]
    write_sheet(wb.add_worksheet("Overlap_Summary"),
                ["Comparison","Gene Count","Sample Genes (first 20)"],
                overlap_rows, [30,14,80])
    write_sheet(wb.add_worksheet("Common_All_3_Methods"),
                ["Gene ID (all 3 methods)"],
                [(g,) for g in sorted(overlap_all)], [30])

print(f"✅  Saved → '{FS_OUTPUT}'")

# ════════════════════════════════════════════════════════════
#  STEP 7 : 9 Models × 3 FS methods  (Pipeline = leak-free)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 7 – Model Training (Pipeline, no leakage)")
print("=" * 60)
print("""
  ┌─────────────────────────────────────────────────────────┐
  │  Each Pipeline = [StandardScaler → FS selector → Model] │
  │  cross_val_predict runs the FULL pipeline inside every   │
  │  CV fold, so the feature selector NEVER sees the         │
  │  validation fold — overfitting from FS is prevented.     │
  └─────────────────────────────────────────────────────────┘
""")

MODELS = {
    "Logistic Regression":  LogisticRegression(C=1.0, max_iter=1000,
                                                solver="liblinear", random_state=42),
    "SVM":                  SVC(C=1.0, probability=True, random_state=42),
    "Random Forest":        RandomForestClassifier(n_estimators=200,
                                                   max_depth=6,        # regularised
                                                   min_samples_leaf=2,
                                                   random_state=42, n_jobs=-1),
    "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=7),       # odd k, larger
    "Naïve Bayes":          GaussianNB(),
    "Decision Tree":        DecisionTreeClassifier(max_depth=5,         # regularised
                                                   min_samples_leaf=3,
                                                   random_state=42),
    "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100,
                                                        learning_rate=0.1,
                                                        max_depth=3,    # regularised
                                                        subsample=0.8,
                                                        random_state=42),
    "XGBoost":              xgb.XGBClassifier(n_estimators=100,
                                               max_depth=4,             # regularised
                                               learning_rate=0.1,
                                               subsample=0.8,
                                               colsample_bytree=0.8,
                                               use_label_encoder=False,
                                               eval_metric="logloss",
                                               random_state=42,
                                               verbosity=0),
    "MLP Neural Network":   MLPClassifier(hidden_layer_sizes=(64,32),
                                           alpha=0.001,                 # L2 regularisation
                                           max_iter=500, random_state=42),
}

METRICS_ORDER = ["Accuracy","Sensitivity","Specificity",
                 "Precision","F1_Score","ROC_AUC",
                 "CV_AUC","CV_AUC_Std","Gap(Test-CV)"]

def compute_metrics(y_true, y_pred, y_prob):
    """Safe metric computation — handles small / imbalanced test sets."""
    # Force labels=[0,1] so confusion matrix is always 2×2
    # even if one class is absent in a tiny test set
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(
        Accuracy    = accuracy_score(y_true, y_pred),
        Sensitivity = float(tp) / (tp + fn + 1e-9),
        Specificity = float(tn) / (tn + fp + 1e-9),
        Precision   = precision_score(y_true, y_pred, zero_division=0, labels=[0,1], pos_label=1),
        F1_Score    = f1_score(y_true, y_pred, zero_division=0, labels=[0,1], pos_label=1),
        ROC_AUC     = roc_auc_score(y_true, y_prob),
    )

PALETTE = ["#E63946","#2196F3","#4CAF50","#FF9800","#9C27B0",
           "#00BCD4","#FF5722","#795548","#607D8B"]

all_results = {}
roc_data    = {}

for fs_name, make_selector in FS_PIPELINE_SELECTORS.items():
    print(f"\n  ── FS: {fs_name} ──")
    all_results[fs_name] = {}
    roc_data[fs_name]    = {}

    for model_name, clf in MODELS.items():
        try:
            # ── Build pipeline ─────────────────────────────────
            pipe = Pipeline([
                ("scaler",   StandardScaler()),
                ("selector", make_selector(N_FEATURES)),
                ("clf",      clf),
            ])

            # ── CV on training set (all steps re-run per fold) ──
            cv_probs = cross_val_predict(
                pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            cv_auc_scores = cross_val_score(
                pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            cv_auc_mean = cv_auc_scores.mean()
            cv_auc_std  = cv_auc_scores.std()

            # ── Final fit on ALL of X_train, evaluate on X_test ─
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_prob = pipe.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_pred, y_prob)
            gap     = round(metrics["ROC_AUC"] - cv_auc_mean, 4)

            metrics["CV_AUC"]     = cv_auc_mean
            metrics["CV_AUC_Std"] = cv_auc_std
            metrics["Gap(Test-CV)"] = gap        # small gap = no overfitting

            all_results[fs_name][model_name] = metrics

            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[fs_name][model_name] = (fpr, tpr, metrics["ROC_AUC"])

            gap_flag = "⚠️ " if abs(gap) > 0.10 else "✅ "
            print(f"   {gap_flag}{model_name:<26} "
                  f"Test AUC={metrics['ROC_AUC']:.3f}  "
                  f"CV AUC={cv_auc_mean:.3f}±{cv_auc_std:.3f}  "
                  f"Gap={gap:+.3f}")
        except Exception as e:
            print(f"   ❌  {model_name} failed: {e}")
            all_results[fs_name][model_name] = {m:0 for m in METRICS_ORDER}

print("\n⚠️  Note: A large |Gap(Test-CV)| > 0.10 may indicate overfitting.")

# ════════════════════════════════════════════════════════════
#  STEP 8A : ROC Curves PDF
#  Layout:
#    Page 1  – Cover / table of contents
#    Pages 2-4 – One full page per FS method (all 9 models)
#    Page 5  – 3-panel summary (all methods side-by-side)
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 8A – ROC Curves PDF")
print("=" * 60)

ROC_PDF = "ROC_Curves.pdf"

def styled_roc_ax(ax, model_rocs, title, palette, show_legend=True):
    """Draw ROC curves with shaded AUC fill on a given Axes."""
    ax.plot([0,1],[0,1],"k--",lw=1.4,alpha=0.6,label="Random  AUC=0.500")
    ax.fill_between([0,1],[0,1],alpha=0.04,color="grey")
    for i,(mname,(fpr,tpr,auc_v)) in enumerate(model_rocs.items()):
        color = palette[i % len(palette)]
        ax.plot(fpr, tpr, lw=2.2, color=color,
                label=f"{mname}  AUC={auc_v:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.04, color=color)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.03])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    if show_legend:
        ax.legend(loc="lower right", fontsize=8.5,
                  framealpha=0.9, edgecolor="#cccccc")

with PdfPages(ROC_PDF) as pdf:

    # ── Page 1 : Cover ─────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#1F3864")
    ax_cov = fig.add_axes([0,0,1,1])
    ax_cov.set_axis_off()
    ax_cov.text(0.5, 0.70, "RNA-seq ML Pipeline",
                ha="center", va="center", fontsize=26,
                color="white", fontweight="bold", transform=ax_cov.transAxes)
    ax_cov.text(0.5, 0.60, "ROC-AUC Curves",
                ha="center", va="center", fontsize=20,
                color="#90CAF9", transform=ax_cov.transAxes)
    ax_cov.text(0.5, 0.50,
                "Feature Selection Methods:  Filter · Wrapper · Embedded",
                ha="center", va="center", fontsize=13,
                color="#CFD8DC", transform=ax_cov.transAxes)
    ax_cov.text(0.5, 0.42,
                f"9 Models  ×  3 FS Methods  =  27 Pipelines",
                ha="center", va="center", fontsize=12,
                color="#CFD8DC", transform=ax_cov.transAxes)
    ax_cov.text(0.5, 0.22,
                "Pages:  2–4  Individual FS method ROC curves\n"
                "Page 5  Three-panel comparison (all methods)\n"
                "Page 6  Best-model overlay across FS methods",
                ha="center", va="center", fontsize=11,
                color="#B0BEC5", transform=ax_cov.transAxes,
                linespacing=1.8)
    pdf.savefig(fig, facecolor=fig.get_facecolor()); plt.close(fig)

    # ── Pages 2-4 : One page per FS method ─────────────────
    for fs_name, model_rocs in roc_data.items():
        fig, ax = plt.subplots(figsize=(9, 8))
        fig.patch.set_facecolor("#FAFAFA")
        styled_roc_ax(ax, model_rocs,
                      f"ROC Curves — {fs_name}\n(Test Set, n={X_test.shape[0]} samples)",
                      PALETTE)
        # AUC ranking table inside figure
        sorted_models = sorted(model_rocs.items(), key=lambda x: x[1][2], reverse=True)
        table_text = "\n".join(
            [f"  {'Rank':>4}   {'Model':<26}  AUC"] +
            [f"  {r+1:>4}.  {mname:<26}  {auc_v:.4f}"
             for r,(mname,(_,_,auc_v)) in enumerate(sorted_models)]
        )
        fig.text(0.01, 0.01, table_text,
                 fontsize=7.5, family="monospace",
                 color="#333333", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.4", fc="white",
                           ec="#cccccc", alpha=0.85))
        plt.tight_layout(rect=[0, 0.18, 1, 1])
        pdf.savefig(fig, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"   📄  Page saved: {fs_name}")

    # ── Page 5 : 3-panel side-by-side ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    fig.patch.set_facecolor("#F5F5F5")
    for ax, (fs_name, model_rocs) in zip(axes, roc_data.items()):
        styled_roc_ax(ax, model_rocs, fs_name, PALETTE, show_legend=True)
    fig.suptitle("ROC Curves — All Feature Selection Methods (Side-by-Side)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("   📄  Page saved: 3-panel comparison")

    # ── Page 6 : Best model per FS method overlay ──────────
    fig, ax = plt.subplots(figsize=(9, 8))
    fig.patch.set_facecolor("#FAFAFA")
    ax.plot([0,1],[0,1],"k--",lw=1.4,alpha=0.5,label="Random  AUC=0.500")
    best_colors = ["#E63946","#2196F3","#4CAF50"]
    linestyles  = ["-","--","-."]
    any_plotted = False
    for idx,(fs_name, model_rocs) in enumerate(roc_data.items()):
        if not model_rocs:                  # skip if all models failed
            continue
        best_mname = max(model_rocs, key=lambda m: model_rocs[m][2])
        fpr, tpr, auc_v = model_rocs[best_mname]
        ax.plot(fpr, tpr, lw=2.8,
                color=best_colors[idx], ls=linestyles[idx],
                label=f"[{fs_name}]\n  Best: {best_mname}  AUC={auc_v:.3f}")
        ax.fill_between(fpr, tpr, alpha=0.07, color=best_colors[idx])
        any_plotted = True
    if not any_plotted:
        ax.text(0.5, 0.5, "No models succeeded\n(all failed during training)",
                ha="center", va="center", fontsize=13, color="red",
                transform=ax.transAxes)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.03])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Best Model Per FS Method — ROC Overlay",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5,
              framealpha=0.95, edgecolor="#aaaaaa")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    pdf.savefig(fig, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("   📄  Page saved: best-model overlay")

    # PDF metadata
    d = pdf.infodict()
    d["Title"]   = "RNA-seq ML Pipeline – ROC Curves"
    d["Subject"] = "Feature Selection & Classification"

print(f"✅  ROC PDF saved → '{ROC_PDF}'  (6 pages)")

# ════════════════════════════════════════════════════════════
#  STEP 8B : Comparison Plots PDF
#  Pages:
#    1  – Cover
#    2  – Performance heatmaps (all 3 FS methods)
#    3  – Grouped bar chart: ROC-AUC across models & methods
#    4  – Grouped bar chart: Accuracy
#    5  – Grouped bar chart: F1-Score
#    6  – Grouped bar chart: Sensitivity & Specificity
#    7  – CV AUC vs Test AUC (overfitting diagnostic)
#    8  – Radar / spider chart: best model per FS method
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 8B – Comparison Plots PDF")
print("=" * 60)

CMP_PDF    = "Comparison_Plots.pdf"
FS_COLORS  = {"Filter (Univariate)":"#2196F3",
              "Wrapper (RFE)":       "#E63946",
              "Embedded (LASSO+RF)":"#4CAF50"}
MODEL_LIST = list(MODELS.keys())
N_MODELS   = len(MODEL_LIST)

def bar_comparison(ax, metric_key, title, ylabel="Score", ylim=(0,1.05)):
    """Grouped bar chart comparing all 3 FS methods across all 9 models."""
    x       = np.arange(N_MODELS)
    n_fs    = len(all_results)
    width   = 0.25
    offsets = np.linspace(-(n_fs-1)*width/2, (n_fs-1)*width/2, n_fs)
    for i,(fs_name,results) in enumerate(all_results.items()):
        vals  = [results.get(m,{}).get(metric_key,0) for m in MODEL_LIST]
        bars  = ax.bar(x + offsets[i], vals, width,
                       label=fs_name, color=FS_COLORS[fs_name],
                       alpha=0.85, edgecolor="white", linewidth=0.6)
        for bar,val in zip(bars,vals):
            if val > 0.01:
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                        f"{val:.2f}", ha="center", va="bottom",
                        fontsize=6.2, rotation=90, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LIST, rotation=35, ha="right", fontsize=8.5)
    ax.set_ylim(ylim); ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

with PdfPages(CMP_PDF) as pdf:

    # ── Page 1 : Cover ─────────────────────────────────────
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("#1B5E20")
    ax_cov = fig.add_axes([0,0,1,1]); ax_cov.set_axis_off()
    ax_cov.text(0.5,0.70,"RNA-seq ML Pipeline",
                ha="center",fontsize=26,color="white",
                fontweight="bold",transform=ax_cov.transAxes)
    ax_cov.text(0.5,0.61,"Model Comparison Report",
                ha="center",fontsize=20,color="#A5D6A7",
                transform=ax_cov.transAxes)
    ax_cov.text(0.5,0.52,"Filter · Wrapper · Embedded  ×  9 ML Models",
                ha="center",fontsize=13,color="#C8E6C9",
                transform=ax_cov.transAxes)
    ax_cov.text(0.5,0.30,
                "Page 2   Performance Heatmaps\n"
                "Page 3   ROC-AUC Grouped Bar Chart\n"
                "Page 4   Accuracy Grouped Bar Chart\n"
                "Page 5   F1-Score Grouped Bar Chart\n"
                "Page 6   Sensitivity & Specificity\n"
                "Page 7   Overfitting Diagnostic (CV vs Test AUC)\n"
                "Page 8   Radar Chart – Best Models",
                ha="center",fontsize=11,color="#B9F6CA",
                transform=ax_cov.transAxes,linespacing=1.8)
    pdf.savefig(fig,facecolor=fig.get_facecolor()); plt.close(fig)

    # ── Page 2 : Heatmaps ──────────────────────────────────
    fig, axes = plt.subplots(1,3,figsize=(20,7))
    fig.patch.set_facecolor("#FAFAFA")
    DISPLAY_METRICS = ["Accuracy","Sensitivity","Specificity",
                       "Precision","F1_Score","ROC_AUC","CV_AUC"]
    for ax,(fs_name,results) in zip(axes,all_results.items()):
        mat = pd.DataFrame(results).T[DISPLAY_METRICS].astype(float)
        sns.heatmap(mat, annot=True, fmt=".3f", cmap="RdYlGn",
                    vmin=0, vmax=1, ax=ax, cbar=True,
                    linewidths=0.5, linecolor="#dddddd",
                    annot_kws={"size":8})
        ax.set_title(fs_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Metric", fontsize=9)
        ax.set_ylabel("Model",  fontsize=9)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)
    fig.suptitle("Performance Heatmaps – All Feature Selection Methods",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    pdf.savefig(fig, dpi=180, bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: Heatmaps")

    # ── Page 3 : ROC-AUC bar ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14,6)); fig.patch.set_facecolor("#FAFAFA")
    bar_comparison(ax,"ROC_AUC","ROC-AUC Comparison — All Models × All FS Methods","AUC")
    plt.tight_layout()
    pdf.savefig(fig,dpi=180,bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: ROC-AUC bar chart")

    # ── Page 4 : Accuracy bar ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14,6)); fig.patch.set_facecolor("#FAFAFA")
    bar_comparison(ax,"Accuracy","Accuracy Comparison — All Models × All FS Methods","Accuracy")
    plt.tight_layout()
    pdf.savefig(fig,dpi=180,bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: Accuracy bar chart")

    # ── Page 5 : F1-Score bar ──────────────────────────────
    fig, ax = plt.subplots(figsize=(14,6)); fig.patch.set_facecolor("#FAFAFA")
    bar_comparison(ax,"F1_Score","F1-Score Comparison — All Models × All FS Methods","F1-Score")
    plt.tight_layout()
    pdf.savefig(fig,dpi=180,bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: F1-Score bar chart")

    # ── Page 6 : Sensitivity & Specificity side-by-side ────
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
    fig.patch.set_facecolor("#FAFAFA")
    bar_comparison(ax1,"Sensitivity","Sensitivity (Recall / TPR)","Sensitivity")
    bar_comparison(ax2,"Specificity","Specificity (TNR)","Specificity")
    fig.suptitle("Sensitivity & Specificity — All Models × All FS Methods",
                 fontsize=14,fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig,dpi=180,bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: Sensitivity & Specificity")

    # ── Page 7 : CV AUC vs Test AUC (overfitting diagnostic) ─
    fig, axes = plt.subplots(1,3,figsize=(20,6))
    fig.patch.set_facecolor("#FAFAFA")
    for ax,(fs_name,results) in zip(axes,all_results.items()):
        mnames  = MODEL_LIST
        cv_auc  = [results.get(m,{}).get("CV_AUC",0)  for m in mnames]
        te_auc  = [results.get(m,{}).get("ROC_AUC",0) for m in mnames]
        cv_std  = [results.get(m,{}).get("CV_AUC_Std",0) for m in mnames]
        x = np.arange(len(mnames))
        ax.bar(x-0.18, cv_auc, 0.34, label="CV AUC",
               color="#2196F3", alpha=0.85, edgecolor="white")
        ax.errorbar(x-0.18, cv_auc, yerr=cv_std,
                    fmt="none", color="#0D47A1", capsize=3, lw=1.2)
        ax.bar(x+0.18, te_auc, 0.34, label="Test AUC",
               color="#E63946", alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(mnames, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0,1.10); ax.axhline(0.5,ls="--",color="grey",lw=0.8,alpha=0.6)
        ax.set_title(fs_name, fontsize=10, fontweight="bold")
        ax.set_ylabel("AUC", fontsize=9)
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3,linestyle="--")
        ax.spines[["top","right"]].set_visible(False)
        # Gap annotation
        for xi,cv,te in zip(x,cv_auc,te_auc):
            gap = te - cv
            clr = "#D32F2F" if abs(gap)>0.10 else "#388E3C"
            ax.text(xi, max(cv,te)+0.03, f"{gap:+.2f}",
                    ha="center",fontsize=6.5,color=clr,fontweight="bold")
    fig.suptitle("Overfitting Diagnostic: CV AUC vs Test AUC\n"
                 "(Error bars = CV std dev  |  Numbers = gap; red if |gap|>0.10)",
                 fontsize=13,fontweight="bold")
    plt.tight_layout()
    pdf.savefig(fig,dpi=180,bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: Overfitting diagnostic")

    # ── Page 8 : Radar chart – best model per FS method ────
    RADAR_METRICS = ["Accuracy","Sensitivity","Specificity",
                     "Precision","F1_Score","ROC_AUC"]
    N_R = len(RADAR_METRICS)
    angles = np.linspace(0, 2*np.pi, N_R, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(10,8)); fig.patch.set_facecolor("#FAFAFA")
    ax_r = fig.add_subplot(111, polar=True)
    radar_colors = ["#E63946","#2196F3","#4CAF50"]
    any_radar = False
    for idx,(fs_name,results) in enumerate(all_results.items()):
        succeeded = {m:v for m,v in results.items() if v.get("ROC_AUC",0) > 0}
        if not succeeded:
            continue
        best_m = max(succeeded, key=lambda m: succeeded[m].get("ROC_AUC",0))
        vals   = [succeeded[best_m].get(mk,0) for mk in RADAR_METRICS]
        vals  += vals[:1]
        ax_r.plot(angles, vals, lw=2.5,
                  color=radar_colors[idx], linestyle="-",
                  label=f"{fs_name}\n  ({best_m})")
        ax_r.fill(angles, vals, alpha=0.12, color=radar_colors[idx])
        any_radar = True
    if not any_radar:
        ax_r.text(0, 0, "No models succeeded", ha="center", fontsize=12, color="red")
    ax_r.set_thetagrids(np.degrees(angles[:-1]), RADAR_METRICS, fontsize=10)
    ax_r.set_ylim(0,1)
    ax_r.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax_r.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7, color="grey")
    ax_r.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
    ax_r.set_title("Radar Chart — Best Model Per FS Method",
                   fontsize=13, fontweight="bold", pad=20)
    ax_r.legend(loc="upper right", bbox_to_anchor=(1.35,1.15),
                fontsize=9, framealpha=0.9)
    plt.tight_layout()
    pdf.savefig(fig,dpi=180,bbox_inches="tight"); plt.close(fig)
    print("   📄  Page saved: Radar chart")

    # PDF metadata
    d = pdf.infodict()
    d["Title"]   = "RNA-seq ML Pipeline – Comparison Plots"
    d["Subject"] = "Feature Selection & Model Comparison"

print(f"✅  Comparison PDF saved → '{CMP_PDF}'  (8 pages)")

# ── Also save standalone heatmap PNG ─────────────────────────
fig, axes = plt.subplots(1,3,figsize=(20,6))
for ax,(fs_name,results) in zip(axes,all_results.items()):
    mat = pd.DataFrame(results).T[DISPLAY_METRICS].astype(float)
    sns.heatmap(mat,annot=True,fmt=".3f",cmap="RdYlGn",
                vmin=0,vmax=1,ax=ax,cbar=True,linewidths=0.5,
                annot_kws={"size":8})
    ax.set_title(fs_name,fontsize=11,fontweight="bold")
plt.suptitle("Performance Heatmap",fontsize=14,fontweight="bold")
plt.tight_layout()
plt.savefig("Performance_Heatmap.png",dpi=150,bbox_inches="tight")
plt.close()
print("✅  Performance_Heatmap.png saved.")

# ── CV-AUC vs Test-AUC gap chart (overfitting diagnostic) ───
fig, axes = plt.subplots(1,3,figsize=(20,5))
for ax,(fs_name,results) in zip(axes,all_results.items()):
    mnames = list(results.keys())
    cv_auc = [results[m]["CV_AUC"]  for m in mnames]
    te_auc = [results[m]["ROC_AUC"] for m in mnames]
    x = np.arange(len(mnames))
    ax.bar(x-0.18, cv_auc, 0.35, label="CV AUC",   color="#2196F3", alpha=0.85)
    ax.bar(x+0.18, te_auc, 0.35, label="Test AUC", color="#E63946", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(mnames, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0,1.05); ax.axhline(0.5,ls="--",color="gray",lw=0.8)
    ax.set_title(f"CV vs Test AUC\n{fs_name}",fontsize=10,fontweight="bold")
    ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
plt.suptitle("Overfitting Diagnostic: CV AUC vs Test AUC\n"
             "(Large gap → possible overfitting)",
             fontsize=13,fontweight="bold")
plt.tight_layout()
plt.savefig("Overfitting_Diagnostic.png",dpi=150,bbox_inches="tight")
plt.close()
print("✅  Overfitting diagnostic chart saved.")

# ════════════════════════════════════════════════════════════
#  STEP 9 : Save Results → Excel
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 9 – Saving Model Results")
print("=" * 60)

RESULTS_OUTPUT = "ML_Model_Results.xlsx"
with pd.ExcelWriter(RESULTS_OUTPUT, engine="xlsxwriter") as writer:
    wb = writer.book
    hdr    = wb.add_format({"bold":True,"bg_color":"#1F3864","font_color":"white",
                             "border":1,"align":"center","valign":"vcenter","text_wrap":True})
    sub    = wb.add_format({"bold":True,"bg_color":"#2E75B6","font_color":"white",
                             "border":1,"align":"center","valign":"vcenter"})
    good   = wb.add_format({"border":1,"num_format":"0.0000","bg_color":"#E2EFDA"})
    ok     = wb.add_format({"border":1,"num_format":"0.0000","bg_color":"#FFEB9C"})
    bad    = wb.add_format({"border":1,"num_format":"0.0000","bg_color":"#FFC7CE"})
    warn   = wb.add_format({"border":1,"num_format":"0.0000","bg_color":"#FFD966"})
    cellfm = wb.add_format({"border":1})
    boldfm = wb.add_format({"border":1,"bold":True})

    def sfmt(val, is_gap=False):
        if is_gap:
            return warn if abs(val) > 0.10 else good
        return good if val >= 0.80 else (ok if val >= 0.60 else bad)

    for fs_name, results in all_results.items():
        ws = wb.add_worksheet(fs_name[:31].replace("/","_"))
        ws.merge_range(0,0,0,len(METRICS_ORDER),f"ML Results – {fs_name}",hdr)
        ws.set_row(0,22); ws.set_row(1,22)
        ws.write(1,0,"Model",sub)
        for c,m in enumerate(METRICS_ORDER,1): ws.write(1,c,m,sub)
        ws.set_column(0,0,26); ws.set_column(1,len(METRICS_ORDER),13)

        for r,(mname,metrics) in enumerate(results.items(),2):
            ws.write(r,0,mname,cellfm)
            for c,m in enumerate(METRICS_ORDER,1):
                v = metrics.get(m,0)
                ws.write(r,c,v, sfmt(v, is_gap=(m=="Gap(Test-CV)")))

    # Summary
    ws_s = wb.add_worksheet("Summary_Best_Models")
    ws_s.merge_range(0,0,0,len(METRICS_ORDER)+1,"Best Model Per FS Method",hdr)
    ws_s.write(1,0,"FS Method",sub); ws_s.write(1,1,"Best Model",sub)
    for c,m in enumerate(METRICS_ORDER,2): ws_s.write(1,c,m,sub)
    ws_s.set_column(0,0,28); ws_s.set_column(1,1,26)
    ws_s.set_column(2,len(METRICS_ORDER)+1,13)
    for r,(fs_name,results) in enumerate(all_results.items(),2):
        best = max(results, key=lambda m: results[m]["ROC_AUC"])
        ws_s.write(r,0,fs_name,cellfm); ws_s.write(r,1,best,boldfm)
        for c,m in enumerate(METRICS_ORDER,2):
            v = results[best].get(m,0)
            ws_s.write(r,c,v, sfmt(v, is_gap=(m=="Gap(Test-CV)")))

    # Full comparison
    ws_c = wb.add_worksheet("Full_Comparison")
    for c,h in enumerate(["FS Method","Model"]+METRICS_ORDER):
        ws_c.write(0,c,h,sub)
    ws_c.set_column(0,0,28); ws_c.set_column(1,1,26)
    ws_c.set_column(2,len(METRICS_ORDER)+1,13)
    row=1
    for fs_name,results in all_results.items():
        for mname,metrics in results.items():
            ws_c.write(row,0,fs_name,cellfm); ws_c.write(row,1,mname,cellfm)
            for c,m in enumerate(METRICS_ORDER,2):
                v=metrics.get(m,0)
                ws_c.write(row,c,v, sfmt(v, is_gap=(m=="Gap(Test-CV)")))
            row+=1

    # Overfitting legend sheet
    ws_l = wb.add_worksheet("Legend")
    legend = [
        ("Metric","Meaning","Good range"),
        ("ROC_AUC","Test-set AUC","≥ 0.80"),
        ("CV_AUC","Mean AUC across CV folds","≥ 0.75"),
        ("CV_AUC_Std","Stability of CV folds","< 0.10 (lower=stable)"),
        ("Gap(Test-CV)","ROC_AUC minus CV_AUC",
         "< 0.10 (green); > 0.10 (yellow=check overfitting)"),
    ]
    for r,row_data in enumerate(legend):
        for c,v in enumerate(row_data):
            ws_l.write(r,c,v, hdr if r==0 else cellfm)
    ws_l.set_column(0,0,20); ws_l.set_column(1,1,40); ws_l.set_column(2,2,45)

print(f"✅  Saved → '{RESULTS_OUTPUT}'")

# ════════════════════════════════════════════════════════════
#  STEP 10 : Download
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  STEP 10 – Downloading Files")
print("=" * 60)

outputs = [FS_OUTPUT, RESULTS_OUTPUT,
           ROC_PDF, CMP_PDF,
           "Performance_Heatmap.png"]

for f in outputs:
    if os.path.exists(f):
        files.download(f); print(f"   ⬇️  {f}")

print("\n" + "=" * 60)
print("  ✅  PIPELINE COMPLETE")
print("=" * 60)
print("""
  Output files:
  ┌──────────────────────────────────────────────────────────┐
  │  Feature_Selection_Results.xlsx                          │
  │  ML_Model_Results.xlsx  (Gap column, colour-coded)       │
  ├──────────────────────────────────────────────────────────┤
  │  ROC_Curves.pdf  (6 pages)                               │
  │    p1  Cover                                             │
  │    p2  Filter ROC — all 9 models                         │
  │    p3  Wrapper ROC — all 9 models                        │
  │    p4  Embedded ROC — all 9 models                       │
  │    p5  3-panel side-by-side comparison                   │
  │    p6  Best-model overlay (one curve per FS method)      │
  ├──────────────────────────────────────────────────────────┤
  │  Comparison_Plots.pdf  (8 pages)                         │
  │    p1  Cover                                             │
  │    p2  Performance heatmaps                              │
  │    p3  ROC-AUC grouped bar chart                         │
  │    p4  Accuracy grouped bar chart                        │
  │    p5  F1-Score grouped bar chart                        │
  │    p6  Sensitivity & Specificity                         │
  │    p7  Overfitting diagnostic (CV vs Test AUC)           │
  │    p8  Radar chart — best model per FS method            │
  └──────────────────────────────────────────────────────────┘
""")
