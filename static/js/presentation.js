// Presentation Mode Logic for UTS Kecerdasan Buatan (10 Minutes Timer & Slides)

let currentSlide = 1;
const totalSlides = 6;
let timerSeconds = 600; // 10 minutes (600 seconds)
let timerInterval = null;
let isRunning = false;

document.addEventListener('DOMContentLoaded', () => {
  initPresentation();
});

function initPresentation() {
  updateSlideView();
  updateTimerDisplay();

  // Keyboard Navigation
  document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === 'PageDown') {
      nextSlide();
    } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
      prevSlide();
    }
  });

  const btnNext = document.getElementById('btnNextSlide');
  const btnPrev = document.getElementById('btnPrevSlide');
  const btnToggleTimer = document.getElementById('btnToggleTimer');
  const btnResetTimer = document.getElementById('btnResetTimer');
  const btnToggleNotes = document.getElementById('btnToggleNotes');

  if (btnNext) btnNext.addEventListener('click', nextSlide);
  if (btnPrev) btnPrev.addEventListener('click', prevSlide);
  if (btnToggleTimer) btnToggleTimer.addEventListener('click', toggleTimer);
  if (btnResetTimer) btnResetTimer.addEventListener('click', resetTimer);
  if (btnToggleNotes) btnToggleNotes.addEventListener('click', toggleNotes);

  // Slide Jump Buttons
  document.querySelectorAll('.slide-pill').forEach((pill) => {
    pill.addEventListener('click', () => {
      const slideNum = parseInt(pill.getAttribute('data-slide'), 10);
      if (slideNum) goToSlide(slideNum);
    });
  });
}

function updateSlideView() {
  document.querySelectorAll('.slide').forEach((slide) => {
    slide.classList.remove('active');
  });

  const activeSlide = document.getElementById(`slide-${currentSlide}`);
  if (activeSlide) activeSlide.classList.add('active');

  const counter = document.getElementById('slideCounter');
  if (counter) counter.innerText = `${currentSlide} / ${totalSlides}`;

  // Update pills
  document.querySelectorAll('.slide-pill').forEach((pill) => {
    const slideNum = parseInt(pill.getAttribute('data-slide'), 10);
    if (slideNum === currentSlide) {
      pill.style.background = 'var(--primary)';
      pill.style.color = '#fff';
    } else {
      pill.style.background = 'rgba(255, 255, 255, 0.05)';
      pill.style.color = 'var(--text-secondary)';
    }
  });

  const prevBtn = document.getElementById('btnPrevSlide');
  const nextBtn = document.getElementById('btnNextSlide');
  if (prevBtn) prevBtn.disabled = currentSlide === 1;
  if (nextBtn) nextBtn.disabled = currentSlide === totalSlides;
}

function nextSlide() {
  if (currentSlide < totalSlides) {
    currentSlide++;
    updateSlideView();
  }
}

function prevSlide() {
  if (currentSlide > 1) {
    currentSlide--;
    updateSlideView();
  }
}

function goToSlide(n) {
  if (n >= 1 && n <= totalSlides) {
    currentSlide = n;
    updateSlideView();
  }
}

function toggleTimer() {
  const btn = document.getElementById('btnToggleTimer');
  if (isRunning) {
    clearInterval(timerInterval);
    isRunning = false;
    if (btn) btn.innerHTML = '▶ Mulai Timer';
  } else {
    isRunning = true;
    if (btn) btn.innerHTML = '⏸ Jeda Timer';
    timerInterval = setInterval(() => {
      if (timerSeconds > 0) {
        timerSeconds--;
        updateTimerDisplay();
      } else {
        clearInterval(timerInterval);
        isRunning = false;
        alert('⏱ Waktu presentasi 10 menit telah berakhir!');
      }
    }, 1000);
  }
}

function resetTimer() {
  clearInterval(timerInterval);
  isRunning = false;
  timerSeconds = 600;
  updateTimerDisplay();
  const btn = document.getElementById('btnToggleTimer');
  if (btn) btn.innerHTML = '▶ Mulai Timer';
}

function updateTimerDisplay() {
  const display = document.getElementById('timerDisplay');
  if (!display) return;

  const mins = Math.floor(timerSeconds / 60);
  const secs = timerSeconds % 60;
  display.innerText = `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;

  if (timerSeconds < 120) {
    display.style.color = 'var(--danger)';
  } else if (timerSeconds < 300) {
    display.style.color = 'var(--warning)';
  } else {
    display.style.color = 'var(--primary)';
  }
}

function toggleNotes() {
  const notes = document.querySelectorAll('.speaker-notes');
  notes.forEach((el) => {
    el.style.display = el.style.display === 'none' ? 'block' : 'none';
  });
}
