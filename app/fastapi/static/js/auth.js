import { auth, setToken } from './api.js';
import { showToast } from './components.js';

export function initAuth(onLogin) {
    const form = document.getElementById('form-auth');
    const errorDiv = document.getElementById('auth-error');
    const btnLogin = document.getElementById('btn-login');
    const btnRegister = document.getElementById('btn-register');
    const btnLogout = document.getElementById('btn-logout');

    function showError(msg) {
        errorDiv.textContent = msg;
        errorDiv.classList.add('is-visible');
    }

    function clearError() {
        errorDiv.textContent = '';
        errorDiv.classList.remove('is-visible');
    }

    function getCredentials() {
        const username = document.getElementById('auth-username').value.trim();
        const password = document.getElementById('auth-password').value;
        if (!username || !password) {
            showError('Please enter username and password');
            return null;
        }
        return { username, password };
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        clearError();
        const creds = getCredentials();
        if (!creds) return;

        btnLogin.disabled = true;
        try {
            const result = await auth.login(creds.username, creds.password);
            setToken(result.access_token);
            showToast('Logged in successfully', 'success');
            onLogin();
        } catch (err) {
            showError(err.message || 'Login failed');
        } finally {
            btnLogin.disabled = false;
        }
    });

    btnRegister.addEventListener('click', async () => {
        clearError();
        const creds = getCredentials();
        if (!creds) return;

        if (creds.password.length < 8) {
            showError('Password must be at least 8 characters');
            return;
        }

        btnRegister.disabled = true;
        try {
            await auth.register(creds.username, creds.password);
            showToast('Account created. Logging in...', 'success');
            const result = await auth.login(creds.username, creds.password);
            setToken(result.access_token);
            onLogin();
        } catch (err) {
            showError(err.message || 'Registration failed');
        } finally {
            btnRegister.disabled = false;
        }
    });

    btnLogout.addEventListener('click', () => {
        setToken(null);
        window.location.reload();
    });
}
