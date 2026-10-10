import requests

base = 'http://127.0.0.1:5000'

# 1. Test Static Files
static_files = [
    '/static/css/style.css',
    '/static/js/main.js',
    '/static/js/presentation.js',
    '/static/img/decision_tree_baseline.png',
    '/static/img/decision_tree_optimized.png'
]
print('=== 1. TESTING STATIC ASSETS ===')
for sf in static_files:
    res = requests.get(base + sf)
    print(f"{sf:40} : Status {res.status_code}, Length {len(res.content)} bytes")
    assert res.status_code == 200

# 2. Test HTML Content Keywords
print('\n=== 2. TESTING HTML CONTENT ===')
pages = {
    '/': ['Sistem Klasifikasi Diabetes', '8 / 8 Kriteria Terpenuhi', '76.62%'],
    '/dataset': ['Pima Indians Diabetes', '500 Pasien', '268 Pasien', 'Glukosa'],
    '/preprocessing': ['Hold-out', '80% data latih', '614 Sampel', '154 Sampel', 'Imputasi Median'],
    '/modeling': ['Entropy', 'max_depth=3', 'Glucose ≤ 127.5', 'Information Gain'],
    '/evaluation': ['Confusion Matrix', 'True Negative (TN)', '83', '16', '20', '35', '76.62%'],
    '/comparison': ['JDMIS 2026', 'Yustri Insani', 'Persamaan Utama', '77.27%'],
    '/predict': ['Form Input Parameter Klinis', 'Pilih Model Klasifikasi', 'Tombol Preset'],
    '/presentation': ['Mode Presentasi 10 Menit', 'SLIDE 1', 'SLIDE 6', 'Catatan Presentasi Mahasiswa']
}

for path, keywords in pages.items():
    res = requests.get(base + path)
    assert res.status_code == 200
    for kw in keywords:
        assert kw in res.text, f"Missing keyword '{kw}' on {path}"
    print(f"{path:16} : OK (All {len(keywords)} keywords verified)")

# 3. Test Predictions
print('\n=== 3. TESTING MULTIPLE PREDICTION SCENARIOS ===')
test_cases = [
    {
        'label': 'High Risk Diabetic Case',
        'data': {'model_type': 'baseline', 'Pregnancies': 8, 'Glucose': 183, 'BloodPressure': 64, 'SkinThickness': 32, 'Insulin': 175, 'BMI': 34.5, 'DiabetesPedigreeFunction': 0.672, 'Age': 48},
        'expected_pred': 1
    },
    {
        'label': 'Low Risk Healthy Case',
        'data': {'model_type': 'baseline', 'Pregnancies': 1, 'Glucose': 85, 'BloodPressure': 66, 'SkinThickness': 29, 'Insulin': 0, 'BMI': 26.6, 'DiabetesPedigreeFunction': 0.351, 'Age': 31},
        'expected_pred': 0
    },
    {
        'label': 'Optimized Model Diabetic Case',
        'data': {'model_type': 'optimized', 'Pregnancies': 6, 'Glucose': 160, 'BloodPressure': 80, 'SkinThickness': 0, 'Insulin': 0, 'BMI': 35.0, 'DiabetesPedigreeFunction': 0.5, 'Age': 45},
        'expected_pred': 1
    }
]

for tc in test_cases:
    res = requests.post(base + '/api/predict', json=tc['data'])
    assert res.status_code == 200
    r_json = res.json()
    assert r_json['prediction'] == tc['expected_pred'], f"Prediction mismatch for {tc['label']}"
    print(f"{tc['label']:32} -> Pred: {r_json['prediction']} ({r_json['prediction_label']}), Confidence: {r_json['confidence']}%, Decision Steps: {len(r_json['decision_steps'])}")

print('\n=== ALL VERIFICATIONS PASSED 100%! APPLICATION IS FULLY FUNCTIONAL! ===')
