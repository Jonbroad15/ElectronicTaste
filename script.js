document.addEventListener('DOMContentLoaded', () => {
    
    // --- Security Gate Logic ---
    // In a real production app, client-side passwords are not secure. 
    // This provides a simple gate to keep casual visitors out as requested.
    const CORRECT_PASSWORD = 'raver'; 
    const overlay = document.getElementById('auth-overlay');
    const mainContent = document.getElementById('main-content');
    const passwordInput = document.getElementById('password-input');
    const authSubmit = document.getElementById('auth-submit');
    const authError = document.getElementById('auth-error');

    function attemptLogin() {
        if (passwordInput.value === CORRECT_PASSWORD) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
                mainContent.style.display = 'block';
                // Trigger markdown load after login so it's fresh
                loadMarkdownFiles();
            }, 300);
        } else {
            authError.textContent = 'Incorrect password. Try again.';
            passwordInput.value = '';
            passwordInput.focus();
        }
    }

    authSubmit.addEventListener('click', attemptLogin);
    passwordInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') attemptLogin();
    });


    // --- Navigation Logic ---
    const navBtns = document.querySelectorAll('.nav-btn');
    const sections = document.querySelectorAll('.content-section');

    navBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            navBtns.forEach(b => b.classList.remove('active'));
            sections.forEach(s => s.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });


    // --- Markdown Loading Logic ---
    async function loadMarkdownFiles() {
        const files = {
            'mission': 'specs/mission.md',
            'roadmap': 'specs/roadmap.md',
            'tech': 'specs/tech.md'
        };

        for (const [id, path] of Object.entries(files)) {
            try {
                // Fetch the markdown file
                const response = await fetch(path);
                if (!response.ok) throw new Error('Network response was not ok');
                const text = await response.text();
                
                // Parse and inject
                const contentDiv = document.getElementById(`${id}-content`);
                contentDiv.innerHTML = marked.parse(text);
            } catch (error) {
                console.error(`Error loading ${path}:`, error);
                document.getElementById(`${id}-content`).innerHTML = `<p style="color:red;">Error loading document. Ensure you are running a local server.</p>`;
            }
        }
    }
});
