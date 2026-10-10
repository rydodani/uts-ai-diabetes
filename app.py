import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
METRICS_PATH = os.path.join(MODEL_DIR, 'metrics.json')

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

ALL_COLUMNS = FEATURE_NAMES + ['Outcome']

FEATURE_INFO = {
    'Pregnancies': {
        'name': 'Jumlah Kehamilan',
        'unit': 'kali',
        'desc': 'Berapa kali pasien pernah hamil',
        'min': 0, 'max': 17, 'step': 1, 'default': 1,
        'normal_hint': '0 - 4 kali'
    },
    'Glucose': {
        'name': 'Kadar Glukosa Darah',
        'unit': 'mg/dL',
        'desc': 'Konsentrasi glukosa plasma 2 jam dalam tes toleransi glukosa oral',
        'min': 0, 'max': 250, 'step': 1, 'default': 117,
        'normal_hint': 'Normal puasa < 100 mg/dL, 2 jam oral < 140 mg/dL'
    },
    'BloodPressure': {
        'name': 'Tekanan Darah Diastolik',
        'unit': 'mm Hg',
        'desc': 'Tekanan darah diastolik pasien',
        'min': 0, 'max': 150, 'step': 1, 'default': 72,
        'normal_hint': 'Normal < 80 mm Hg'
    },
    'SkinThickness': {
        'name': 'Ketebalan Lipatan Kulit',
        'unit': 'mm',
        'desc': 'Ketebalan lipatan kulit otot triceps',
        'min': 0, 'max': 99, 'step': 1, 'default': 23,
        'normal_hint': 'Normal 10 - 30 mm'
    },
    'Insulin': {
        'name': 'Kadar Insulin Serum',
        'unit': 'mu U/ml',
        'desc': 'Kadar insulin serum setelah 2 jam',
        'min': 0, 'max': 850, 'step': 1, 'default': 30,
        'normal_hint': 'Normal puasa 16 - 166 mu U/ml'
    },
    'BMI': {
        'name': 'Indeks Massa Tubuh (BMI)',
        'unit': 'kg/m²',
        'desc': 'Berat badan dalam kg dibagi kuadrat tinggi dalam meter',
        'min': 0.0, 'max': 70.0, 'step': 0.1, 'default': 32.0,
        'normal_hint': 'Normal 18.5 - 24.9, Overweight 25 - 29.9, Obesitas ≥ 30'
    },
    'DiabetesPedigreeFunction': {
        'name': 'Riwayat Silsilah Diabetes (DPF)',
        'unit': 'skor',
        'desc': 'Fungsi skor riwayat keturunan genetik diabetes pada keluarga',
        'min': 0.05, 'max': 2.5, 'step': 0.01, 'default': 0.37,
        'normal_hint': 'Rata-rata populasi ~0.47'
    },
    'Age': {
        'name': 'Usia Pasien',
        'unit': 'tahun',
        'desc': 'Usia pasien saat pemeriksaan',
        'min': 15, 'max': 100, 'step': 1, 'default': 29,
        'normal_hint': 'Rentang dewasa 21 - 81 tahun'
    }
}

# Load Models, Imputer & Metrics
model_baseline = None
model_optimized = None
imputer = None
metrics_data = {}

def load_artifacts():
    global model_baseline, model_optimized, imputer, metrics_data
    try:
        model_baseline = joblib.load(os.path.join(MODEL_DIR, 'decision_tree_baseline.joblib'))
        model_optimized = joblib.load(os.path.join(MODEL_DIR, 'decision_tree_optimized.joblib'))
        imputer = joblib.load(os.path.join(MODEL_DIR, 'median_imputer.joblib'))
    except Exception as e:
        print(f"Warning: Models not loaded yet ({e}). Run train_model.py first.")

    try:
        with open(METRICS_PATH, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
    except Exception as e:
        print(f"Warning: metrics.json not found ({e}).")

load_artifacts()

# Helper: Load Dataset DataFrame
def get_dataset():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH, names=ALL_COLUMNS)
    return pd.DataFrame()

# ==========================================
# WEB ROUTES
# ==========================================

@app.route('/')
def index():
    """Halaman Dashboard Utama / Ringkasan Proyek UTS"""
    return render_template(
        'index.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Dashboard Utama'
    )

@app.route('/dataset')
def dataset():
    """Halaman Eksplorasi Dataset & Karakteristik Data"""
    df = get_dataset()
    sample_data = df.head(50).to_dict(orient='records') if not df.empty else []
    return render_template(
        'dataset.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        sample_data=sample_data,
        total_rows=len(df),
        page_title='Karakteristik & Sumber Dataset'
    )

@app.route('/preprocessing')
def preprocessing():
    """Halaman Tahapan Preprocessing & Splitting Data"""
    return render_template(
        'preprocessing.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Preprocessing & Data Splitting'
    )

@app.route('/modeling')
def modeling():
    """Halaman Pemodelan Algoritma Decision Tree"""
    return render_template(
        'modeling.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Pemodelan Decision Tree'
    )

@app.route('/evaluation')
def evaluation():
    """Halaman Evaluasi Model (Confusion Matrix, Akurasi, Presisi, Recall)"""
    return render_template(
        'evaluation.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Evaluasi Kinerja Model'
    )

@app.route('/comparison')
def comparison():
    """Halaman Komparasi Hasil Proyek vs Jurnal JDMIS 2026 (Poin 8 UTS)"""
    return render_template(
        'comparison.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Komparasi dengan Jurnal Referensi'
    )

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    """Halaman Form Pengujian Data Pasien Baru"""
    return render_template(
        'predict.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Pengujian Data Baru (Prediksi)'
    )

@app.route('/presentation')
def presentation():
    """Halaman Mode Presentasi 10 Menit UTS (Pitch Deck View)"""
    return render_template(
        'presentation.html',
        metrics=metrics_data,
        feature_info=FEATURE_INFO,
        page_title='Mode Presentasi UTS (10 Menit)'
    )

# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API Prediksi Data Baru Berdasarkan 8 Variabel Medis"""
    try:
        data = request.get_json(force=True)
        model_type = data.get('model_type', 'baseline')  # 'baseline' or 'optimized'

        # Ekstraksi dan konversi 8 fitur
        features_dict = {}
        features_list = []
        for feat in FEATURE_NAMES:
            val = float(data.get(feat, FEATURE_INFO[feat]['default']))
            features_dict[feat] = val
            features_list.append(val)

        input_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)

        # Pilih Model
        if model_type == 'optimized' and model_optimized is not None and imputer is not None:
            # Terapkan imputasi median jika nilai 0 pada fitur fisiologis
            input_clean = input_df.copy()
            physio_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in physio_cols:
                input_clean[col] = input_clean[col].replace(0, np.nan)
            input_imputed = imputer.transform(input_clean)
            model = model_optimized
            pred = int(model.predict(input_imputed)[0])
            prob = model.predict_proba(input_imputed)[0].tolist()
            used_model_name = "Decision Tree Preprocessed (Median Imputation)"
        else:
            model = model_baseline
            pred = int(model.predict(input_df)[0])
            prob = model.predict_proba(input_df)[0].tolist()
            used_model_name = "Decision Tree Baseline (Sesuai Jurnal JDMIS)"

        prob_sehat = round(prob[0] * 100, 2)
        prob_diabetes = round(prob[1] * 100, 2)

        # Analisis Aturan Keputusan yang Dilalui (Decision Path)
        # Menilai kondisi klinis pasien berdasarkan split tree
        glu = features_dict['Glucose']
        bmi = features_dict['BMI']
        age = features_dict['Age']

        decision_steps = []
        if glu <= 127.5:
            decision_steps.append(f"Kadar Glukosa ({glu} mg/dL) ≤ 127.5 mg/dL (Ambang Batas Rendah-Sedang).")
            if age <= 28.5:
                decision_steps.append(f"Usia Pasien ({age} tahun) ≤ 28.5 tahun (Kelompok Usia Muda).")
                decision_steps.append("Kesimpulan Cabang: Probabilitas Sehat Sangat Tinggi.")
            else:
                decision_steps.append(f"Usia Pasien ({age} tahun) > 28.5 tahun.")
                if bmi <= 26.35:
                    decision_steps.append(f"BMI ({bmi} kg/m²) ≤ 26.35 kg/m² (Kategori Berat Badan Ideal).")
                else:
                    decision_steps.append(f"BMI ({bmi} kg/m²) > 26.35 kg/m² (Kategori Berat Badan Berlebih).")
        else:
            decision_steps.append(f"Kadar Glukosa ({glu} mg/dL) > 127.5 mg/dL (Ambang Batas Tinggi - Faktor Risiko Dominan).")
            if bmi <= 29.95:
                decision_steps.append(f"BMI ({bmi} kg/m²) ≤ 29.95 kg/m².")
                if glu <= 145.5:
                    decision_steps.append(f"Glukosa ({glu} mg/dL) ≤ 145.5 mg/dL.")
                else:
                    decision_steps.append(f"Glukosa ({glu} mg/dL) > 145.5 mg/dL (Hiperglikemia Signifikan).")
            else:
                decision_steps.append(f"BMI ({bmi} kg/m²) > 29.95 kg/m² (Kategori Obesitas Tinggi).")
                decision_steps.append("Kesimpulan Cabang: Kombinasi Glukosa Tinggi & Obesitas memicu Diagnosis Diabetes.")

        # Rekomendasi Klinis
        if pred == 1:
            clinical_advice = (
                "⚠️ Pasien terindikasi memiliki risiko tinggi Diabetes Melitus. "
                "Disarankan segera melakukan tes konfirmasi laboratorium lanjutan (HbA1c & TTGO), "
                "mengatur pola makan rendah gula, serta berkonsultasi dengan dokter spesialis penyakit dalam."
            )
            badge_class = "danger"
            status_text = "POSITIF DIABETES (Kelas 1)"
        else:
            clinical_advice = (
                "✅ Pasien terindikasi Non-Diabetes (Kategori Sehat). "
                "Tetap pertahankan pola hidup sehat, olahraga rutin minimal 150 menit per minggu, "
                "dan lakukan pemeriksaan berkala kadar gula darah tahunan."
            )
            badge_class = "success"
            status_text = "NEGATIF DIABETES / SEHAT (Kelas 0)"

        return jsonify({
            'status': 'success',
            'prediction': pred,
            'prediction_label': 'Diabetes' if pred == 1 else 'Sehat (Non-Diabetes)',
            'status_text': status_text,
            'badge_class': badge_class,
            'probabilities': {
                'sehat': prob_sehat,
                'diabetes': prob_diabetes
            },
            'confidence': max(prob_sehat, prob_diabetes),
            'model_used': used_model_name,
            'input_data': features_dict,
            'decision_steps': decision_steps,
            'clinical_advice': clinical_advice
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Mengembalikan metrik kalkulasi riil dalam format JSON"""
    return jsonify(metrics_data)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
