// ============================================
// Learn Logic Programming - Site Interactivity
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  const sections = document.querySelectorAll('section');
  const navLinks = document.querySelectorAll('#sidebar nav a[data-section]');
  const progressFill = document.getElementById('progress-fill');
  const menuToggle = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('sidebar');

  // Track completed sections
  const completed = new Set(JSON.parse(localStorage.getItem('lp-completed') || '[]'));

  // ---- Section Navigation ----
  function showSection(id) {
    sections.forEach(s => s.classList.remove('active'));
    navLinks.forEach(a => a.classList.remove('active'));

    const target = document.getElementById(id);
    if (target) {
      target.classList.add('active');
      window.scrollTo(0, 0);
    }

    const link = document.querySelector(`a[data-section="${id}"]`);
    if (link) link.classList.add('active');

    // Mark current as completed
    completed.add(id);
    localStorage.setItem('lp-completed', JSON.stringify([...completed]));
    updateProgress();

    // Update nav buttons
    updateNavButtons(id);

    // Close mobile sidebar
    sidebar.classList.remove('open');

    // Update URL hash
    history.replaceState(null, '', '#' + id);
  }

  function updateProgress() {
    const total = sections.length;
    const done = completed.size;
    const pct = Math.round((done / total) * 100);
    progressFill.style.width = pct + '%';
  }

  function updateNavButtons(currentId) {
    const ids = [...sections].map(s => s.id);
    const idx = ids.indexOf(currentId);

    document.querySelectorAll('.nav-prev').forEach(btn => {
      if (idx <= 0) {
        btn.classList.add('disabled');
        btn.onclick = null;
      } else {
        btn.classList.remove('disabled');
        btn.onclick = () => showSection(ids[idx - 1]);
      }
    });

    document.querySelectorAll('.nav-next').forEach(btn => {
      if (idx >= ids.length - 1) {
        btn.classList.add('disabled');
        btn.onclick = null;
      } else {
        btn.classList.remove('disabled');
        btn.onclick = () => showSection(ids[idx + 1]);
      }
    });
  }

  // Bind nav links
  navLinks.forEach(link => {
    link.addEventListener('click', e => {
      e.preventDefault();
      showSection(link.dataset.section);
    });
  });

  // ---- Mobile Menu ----
  menuToggle.addEventListener('click', () => {
    sidebar.classList.toggle('open');
  });

  // ---- Solution Toggles ----
  document.querySelectorAll('.solution-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const solution = btn.nextElementSibling;
      const isOpen = solution.classList.toggle('open');
      btn.textContent = isOpen ? 'Hide Solution' : 'Show Solution';
    });
  });

  // ---- Quizzes ----
  document.querySelectorAll('.quiz').forEach(quiz => {
    const options = quiz.querySelectorAll('.quiz-option');
    const explanation = quiz.querySelector('.quiz-explanation');
    let answered = false;

    options.forEach(opt => {
      opt.addEventListener('click', () => {
        if (answered) return;
        answered = true;

        const isCorrect = opt.dataset.correct === 'true';
        opt.classList.add(isCorrect ? 'correct' : 'wrong');

        if (!isCorrect) {
          options.forEach(o => {
            if (o.dataset.correct === 'true') o.classList.add('correct');
          });
        }

        if (explanation) explanation.style.display = 'block';
      });
    });
  });

  // ---- Init ----
  const hash = window.location.hash.slice(1);
  const initialSection = hash && document.getElementById(hash) ? hash : 'ch0';
  showSection(initialSection);
  updateProgress();
});
