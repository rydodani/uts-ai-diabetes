import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

FEATURE_NAMES = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]
TARGET_NAME = 'Outcome'
ALL_COLUMNS = FEATURE_NAMES + [TARGET_NAME]

FEATURE_LABELS_ID = {
    'Pregnancies': 'Jumlah Kehamilan',
    'Glucose': 'Kadar Glukosa (mg/dL)',
    'BloodPressure': 'Tekanan Darah Diastolik (mm Hg)',
    'SkinThickness': 'Ketebalan Lipatan Kulit Triceps (mm)',
    'Insulin': 'Kadar Insulin Serum 2 Jam (mu U/ml)',
    'BMI': 'Indeks Massa Tubuh (BMI, kg/m²)',
    'DiabetesPedigreeFunction': 'Fungsi Riwayat Keturunan Diabetes (DPF)',
    'Age': 'Usia (Tahun)'
}

def train_and_evaluate():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'diabetes.csv')
    model_dir = os.path.join(base_dir, 'model')
    img_dir = os.path.join(base_dir, 'static', 'img')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path, names=ALL_COLUMNS)

    # 1. Dataset Characteristics & Summary
    total_samples = len(df)
    class_counts = df[TARGET_NAME].value_counts().to_dict()
    non_diabetic_count = int(class_counts.get(0, 0))
    diabetic_count = int(class_counts.get(1, 0))

    zero_counts = {}
    physio_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in FEATURE_NAMES:
        zc = int((df[col] == 0).sum())
        zero_counts[col] = {
            'count': zc,
            'percentage': round((zc / total_samples) * 100, 2),
            'is_physiologically_abnormal': col in physio_zero_cols
        }

    feature_stats = {}
    for col in FEATURE_NAMES:
        feature_stats[col] = {
            'label_id': FEATURE_LABELS_ID[col],
            'mean': round(float(df[col].mean()), 2),
            'std': round(float(df[col].std()), 2),
            'min': round(float(df[col].min()), 2),
            'median': round(float(df[col].median()), 2),
            'max': round(float(df[col].max()), 2)
        }

    X = df[FEATURE_NAMES]
    y = df[TARGET_NAME]

    # 2. Train/Test Split (Hold-out 80:20 - random_state=42 as in Journal)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ==========================================
    # MODEL 1: BASELINE (SESUAI JURNAL JDMIS)
    # ==========================================
    clf_baseline = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        random_state=42
    )
    clf_baseline.fit(X_train_raw, y_train)

    y_pred_baseline = clf_baseline.predict(X_test_raw)
    y_prob_baseline = clf_baseline.predict_proba(X_test_raw)[:, 1]

    cm_base = confusion_matrix(y_test, y_pred_baseline)
    tn_b, fp_b, fn_b, tp_b = int(cm_base[0,0]), int(cm_base[0,1]), int(cm_base[1,0]), int(cm_base[1,1])

    acc_b = accuracy_score(y_test, y_pred_baseline)
    prec_b1 = precision_score(y_test, y_pred_baseline, pos_label=1)
    rec_b1 = recall_score(y_test, y_pred_baseline, pos_label=1)
    f1_b1 = f1_score(y_test, y_pred_baseline, pos_label=1)

    prec_b0 = precision_score(y_test, y_pred_baseline, pos_label=0)
    rec_b0 = recall_score(y_test, y_pred_baseline, pos_label=0)
    f1_b0 = f1_score(y_test, y_pred_baseline, pos_label=0)
    spec_b = tn_b / (tn_b + fp_b) if (tn_b + fp_b) > 0 else 0

    importances_base = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURE_NAMES, clf_baseline.feature_importances_)
    }

    rules_base = export_text(clf_baseline, feature_names=FEATURE_NAMES)

    # Save Baseline Model
    joblib.dump(clf_baseline, os.path.join(model_dir, 'decision_tree_baseline.joblib'))

    # Visualisasi Baseline Decision Tree
    plt.figure(figsize=(18, 9), dpi=300)
    plot_tree(
        clf_baseline,
        feature_names=FEATURE_NAMES,
        class_names=['Sehat (0)', 'Diabetes (1)'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Visualisasi Model Decision Tree Baseline (Sesuai Jurnal JDMIS 2026 - max_depth=3, Entropy)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'decision_tree_baseline.png'), dpi=300)
    plt.savefig(os.path.join(img_dir, 'decision_tree_baseline.svg'))
    plt.close()

    # ==========================================
    # MODEL 2: OPTIMIZED (WITH PREPROCESSING IMPUTATION)
    # ==========================================
    # Sesuai saran pengembangan di Bab 4 Jurnal:
    # Mengatasi nilai 0 pada parameter medis fisiologis (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
    # menggunakan Imputasi Median yang dihitung HANYA dari data latih (mencegah data leakage).
    X_train_clean = X_train_raw.copy()
    X_test_clean = X_test_raw.copy()

    for col in physio_zero_cols:
        X_train_clean[col] = X_train_clean[col].replace(0, np.nan)
        X_test_clean[col] = X_test_clean[col].replace(0, np.nan)

    imputer = SimpleImputer(strategy='median')
    X_train_clean = pd.DataFrame(
        imputer.fit_transform(X_train_clean),
        columns=FEATURE_NAMES,
        index=X_train_raw.index
    )
    X_test_clean = pd.DataFrame(
        imputer.transform(X_test_clean),
        columns=FEATURE_NAMES,
        index=X_test_raw.index
    )

    imputed_medians = {
        col: round(float(val), 2)
        for col, val in zip(FEATURE_NAMES, imputer.statistics_)
    }

    # Model Decision Tree dengan Imputasi Median
    clf_opt = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=3,
        random_state=42
    )
    clf_opt.fit(X_train_clean, y_train)

    y_pred_opt = clf_opt.predict(X_test_clean)
    y_prob_opt = clf_opt.predict_proba(X_test_clean)[:, 1]

    cm_opt = confusion_matrix(y_test, y_pred_opt)
    tn_o, fp_o, fn_o, tp_o = int(cm_opt[0,0]), int(cm_opt[0,1]), int(cm_opt[1,0]), int(cm_opt[1,1])

    acc_o = accuracy_score(y_test, y_pred_opt)
    prec_o1 = precision_score(y_test, y_pred_opt, pos_label=1)
    rec_o1 = recall_score(y_test, y_pred_opt, pos_label=1)
    f1_o1 = f1_score(y_test, y_pred_opt, pos_label=1)

    prec_o0 = precision_score(y_test, y_pred_opt, pos_label=0)
    rec_o0 = recall_score(y_test, y_pred_opt, pos_label=0)
    f1_o0 = f1_score(y_test, y_pred_opt, pos_label=0)
    spec_o = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0

    importances_opt = {
        feat: round(float(imp), 4)
        for feat, imp in zip(FEATURE_NAMES, clf_opt.feature_importances_)
    }

    rules_opt = export_text(clf_opt, feature_names=FEATURE_NAMES)

    # Save Optimized Model & Imputer
    joblib.dump(clf_opt, os.path.join(model_dir, 'decision_tree_optimized.joblib'))
    joblib.dump(imputer, os.path.join(model_dir, 'median_imputer.joblib'))

    # Visualisasi Optimized Decision Tree
    plt.figure(figsize=(18, 9), dpi=300)
    plot_tree(
        clf_opt,
        feature_names=FEATURE_NAMES,
        class_names=['Sehat (0)', 'Diabetes (1)'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title("Visualisasi Model Decision Tree Teroptimasi Preprocessing (Median Imputation, max_depth=3, Entropy)", fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, 'decision_tree_optimized.png'), dpi=300)
    plt.savefig(os.path.join(img_dir, 'decision_tree_optimized.svg'))
    plt.close()

    # ==========================================
    # METRICS JSON EXPORT (FACTUAL & OBJECTIVE)
    # ==========================================
    metrics_data = {
        'dataset_summary': {
            'total_rows': total_samples,
            'total_cols': len(ALL_COLUMNS),
            'feature_names': FEATURE_NAMES,
            'feature_labels_id': FEATURE_LABELS_ID,
            'target_name': TARGET_NAME,
            'target_distribution': {
                '0': {'label': 'Non-Diabetes (Sehat)', 'count': non_diabetic_count, 'pct': round(non_diabetic_count / total_samples * 100, 2)},
                '1': {'label': 'Diabetes', 'count': diabetic_count, 'pct': round(diabetic_count / total_samples * 100, 2)}
            },
            'zero_counts': zero_counts,
            'feature_stats': feature_stats
        },
        'splitting': {
            'method': 'Hold-out Validation',
            'train_ratio': 0.80,
            'test_ratio': 0.20,
            'train_count': len(X_train_raw),
            'test_count': len(X_test_raw),
            'random_state': 42,
            'test_target_distribution': {
                '0': int((y_test == 0).sum()),
                '1': int((y_test == 1).sum())
            }
        },
        'model_baseline': {
            'name': 'Decision Tree Baseline (Jurnal JDMIS)',
            'criterion': 'entropy',
            'max_depth': 3,
            'random_state': 42,
            'confusion_matrix': {
                'tn': tn_b,
                'fp': fp_b,
                'fn': fn_b,
                'tp': tp_b,
                'matrix': [[tn_b, fp_b], [fn_b, tp_b]]
            },
            'metrics': {
                'accuracy': round(acc_b, 4),
                'accuracy_pct': round(acc_b * 100, 2),
                'precision_diabetes': round(prec_b1, 4),
                'precision_diabetes_pct': round(prec_b1 * 100, 2),
                'recall_diabetes': round(rec_b1, 4),
                'recall_diabetes_pct': round(rec_b1 * 100, 2),
                'f1_diabetes': round(f1_b1, 4),
                'precision_sehat': round(prec_b0, 4),
                'recall_sehat': round(rec_b0, 4),
                'specificity': round(spec_b, 4)
            },
            'feature_importances': importances_base,
            'rules_text': rules_base
        },
        'model_optimized': {
            'name': 'Decision Tree Preprocessed (Median Imputation)',
            'criterion': 'entropy',
            'max_depth': 3,
            'random_state': 42,
            'imputation': imputed_medians,
            'confusion_matrix': {
                'tn': tn_o,
                'fp': fp_o,
                'fn': fn_o,
                'tp': tp_o,
                'matrix': [[tn_o, fp_o], [fn_o, tp_o]]
            },
            'metrics': {
                'accuracy': round(acc_o, 4),
                'accuracy_pct': round(acc_o * 100, 2),
                'precision_diabetes': round(prec_o1, 4),
                'precision_diabetes_pct': round(prec_o1 * 100, 2),
                'recall_diabetes': round(rec_o1, 4),
                'recall_diabetes_pct': round(rec_o1 * 100, 2),
                'f1_diabetes': round(f1_o1, 4),
                'precision_sehat': round(prec_o0, 4),
                'recall_sehat': round(rec_o0, 4),
                'specificity': round(spec_o, 4)
            },
            'feature_importances': importances_opt,
            'rules_text': rules_opt
        },
        'journal_reference': {
            'title': 'Prediksi Diabetes Berbasis Decision Tree Dengan Menggunakan Dataset Pima Indians Diabetes',
            'authors': 'Yustri Insani, Marcel Filemon Naibaho, Sardo Pardingotan Sipayung',
            'institution': 'Universitas Katolik Santo Thomas, Medan',
            'journal': 'JDMIS: Journal of Data Mining and Information Systems',
            'volume_issue': 'Vol. 4, No. 1, 2026, pp. 40-45',
            'doi': '10.54259/jdmis.v4i1.7107',
            'reported_metrics': {
                'accuracy_pct': 76.62,
                'precision_diabetes': 0.69,
                'recall_diabetes': 0.64,
                'f1_diabetes': 0.66,
                'precision_sehat': 0.81,
                'tn': 83,
                'fp': 16,
                'fn': 20,
                'tp': 35
            }
        }
    }

    metrics_file = os.path.join(model_dir, 'metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)

    print(f"Metrics and model data successfully exported to: {metrics_file}")
    print("\n--- BASELINE METRICS (Jurnal Match) ---")
    print(f"Accuracy: {acc_b*100:.2f}% | Precision (1): {prec_b1:.4f} | Recall (1): {rec_b1:.4f} | F1: {f1_b1:.4f}")
    print(f"CM: TN={tn_b}, FP={fp_b}, FN={fn_b}, TP={tp_b}")

    print("\n--- OPTIMIZED METRICS (With Imputation) ---")
    print(f"Accuracy: {acc_o*100:.2f}% | Precision (1): {prec_o1:.4f} | Recall (1): {rec_o1:.4f} | F1: {f1_o1:.4f}")
    print(f"CM: TN={tn_o}, FP={fp_o}, FN={fn_o}, TP={tp_o}")

if __name__ == '__main__':
    train_and_evaluate()
