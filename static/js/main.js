// Main Application Script - UTS AI Diabetes Prediction
document.addEventListener('DOMContentLoaded', () => {
  initPredictionForm();
  initPresetButtons();
});

// Preset Data Pasien untuk Demo & Uji Coba Cepat
const PRESETS = {
  sehat: {
    name: "Pasien Sehat (Risiko Sangat Rendah)",
    data: {
      Pregnancies: 1,
      Glucose: 85,
      BloodPressure: 66,
      SkinThickness: 29,
      Insulin: 0,
      BMI: 26.6,
      DiabetesPedigreeFunction: 0.351,
      Age: 31
    }
  },
  diabetes: {
    name: "Pasien Diabetes (Risiko Tinggi - Hiperglikemia & Obesitas)",
    data: {
      Pregnancies: 8,
      Glucose: 183,
      BloodPressure: 64,
      SkinThickness: 32,
      Insulin: 175,
      BMI: 34.5,
      DiabetesPedigreeFunction: 0.672,
      Age: 48
    }
  },
  borderline: {
    name: "Pasien Ambang Batas (Borderline Glukosa ~130 mg/dL)",
    data: {
      Pregnancies: 3,
      Glucose: 128,
      BloodPressure: 76,
      SkinThickness: 25,
      Insulin: 110,
      BMI: 29.5,
      DiabetesPedigreeFunction: 0.420,
      Age: 35
    }
  },
  muda_sehat: {
    name: "Pasien Usia Muda (Glukosa Rendah & Usia < 28 Tahun)",
    data: {
      Pregnancies: 1,
      Glucose: 89,
      BloodPressure: 66,
      SkinThickness: 23,
      Insulin: 94,
      BMI: 24.1,
      DiabetesPedigreeFunction: 0.167,
      Age: 21
    }
  }
};

function initPresetButtons() {
  const container = document.getElementById('presetChipsContainer');
  if (!container) return;

  Object.entries(PRESETS).forEach(([key, preset]) => {
    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'preset-chip';
    chip.innerHTML = `⚡ <strong>${preset.name}</strong>`;
    chip.addEventListener('click', () => loadPreset(preset.data));
    container.appendChild(chip);
  });
}

function loadPreset(data) {
  Object.entries(data).forEach(([key, val]) => {
    const input = document.getElementById(`input_${key}`);
    if (input) {
      input.value = val;
      // Trigger animation effect
      input.style.borderColor = 'var(--primary)';
      setTimeout(() => { input.style.borderColor = ''; }, 600);
    }
  });

  const notify = document.getElementById('presetNotification');
  if (notify) {
    notify.innerText = "Data preset berhasil dimuat ke dalam form!";
    notify.style.display = 'block';
    setTimeout(() => { notify.style.display = 'none'; }, 3000);
  }
}

function initPredictionForm() {
  const form = document.getElementById('predictionForm');
  if (!form) return;

  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const submitBtn = document.getElementById('btnSubmitPredict');
    const resultContainer = document.getElementById('predictionResultContainer');
    const loadingIndicator = document.getElementById('predictLoading');

    if (submitBtn) submitBtn.disabled = true;
    if (loadingIndicator) loadingIndicator.style.display = 'block';
    if (resultContainer) resultContainer.style.display = 'none';

    // Ambil nilai dari form
    const formData = new FormData(form);
    const payload = {};
    formData.forEach((value, key) => {
      payload[key] = parseFloat(value);
    });

    const modelType = document.querySelector('input[name="model_type"]:checked')?.value || 'baseline';
    payload['model_type'] = modelType;

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Content': 'application/json', 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const res = await response.json();

      if (res.status === 'success') {
        renderPredictionResult(res);
      } else {
        alert('Gagal melakukan prediksi: ' + (res.message || 'Terjadi kesalahan.'));
      }
    } catch (err) {
      console.error(err);
      alert('Terjadi kesalahan jaringan saat memanggil API prediksi.');
    } finally {
      if (submitBtn) submitBtn.disabled = false;
      if (loadingIndicator) loadingIndicator.style.display = 'none';
    }
  });
}

function renderPredictionResult(data) {
  const container = document.getElementById('predictionResultContainer');
  if (!container) return;

  const isDiabetic = data.prediction === 1;
  const cardClass = isDiabetic ? 'danger' : 'success';

  const probSehat = data.probabilities.sehat;
  const probDiabetes = data.probabilities.diabetes;

  let stepsHtml = '';
  if (data.decision_steps && data.decision_steps.length > 0) {
    stepsHtml = data.decision_steps.map(step => `
      <li style="margin-bottom: 0.4rem; color: var(--text-secondary);">
        <span style="color: var(--primary);">▸</span> ${step}
      </li>
    `).join('');
  }

  container.innerHTML = `
    <div class="result-card ${cardClass}">
      <div class="result-badge">
        ${data.status_text}
      </div>

      <h3 style="font-size: 1.5rem; margin-bottom: 0.5rem; color: #fff;">
        Hasil Diagnosis: <strong>${data.prediction_label}</strong>
      </h3>

      <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 1.5rem;">
        Model yang digunakan: <strong>${data.model_used}</strong> | Tingkat Keyakinan: <strong>${data.confidence}%</strong>
      </p>

      <div style="background: rgba(0,0,0,0.3); padding: 1.25rem; border-radius: var(--radius-sm); margin-bottom: 1.5rem;">
        <div style="display: flex; justify-content: space-between; font-weight: 700; font-size: 0.9rem; margin-bottom: 0.4rem;">
          <span style="color: var(--success);">Probabilitas Sehat: ${probSehat}%</span>
          <span style="color: var(--danger);">Probabilitas Diabetes: ${probDiabetes}%</span>
        </div>
        <div class="probability-bar-wrapper">
          <div class="prob-fill-sehat" style="width: ${probSehat}%;"></div>
          <div class="prob-fill-diabetes" style="width: ${probDiabetes}%;"></div>
        </div>
      </div>

      <div style="text-align: left; background: rgba(15, 23, 42, 0.7); padding: 1.25rem; border-radius: var(--radius-sm); border: 1px solid var(--border-light); margin-bottom: 1.5rem;">
        <h4 style="color: #fff; font-size: 0.95rem; margin-bottom: 0.75rem; display: flex; align-items: center; gap: 0.5rem;">
          <span>🌳</span> Jalur Penelusuran Pohon Keputusan (Decision Path):
        </h4>
        <ul style="list-style: none; padding-left: 0; font-size: 0.875rem;">
          ${stepsHtml}
        </ul>
      </div>

      <div style="text-align: left; background: rgba(255, 255, 255, 0.04); padding: 1.25rem; border-radius: var(--radius-sm); border-left: 4px solid ${isDiabetic ? 'var(--danger)' : 'var(--success)'};">
        <h4 style="color: #fff; font-size: 0.95rem; margin-bottom: 0.4rem;">
          📋 Rekomendasi Klinis & Interpretasi Medis:
        </h4>
        <p style="font-size: 0.9rem; color: var(--text-secondary); line-height: 1.5;">
          ${data.clinical_advice}
        </p>
      </div>
    </div>
  `;

  container.style.display = 'block';
  container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}
