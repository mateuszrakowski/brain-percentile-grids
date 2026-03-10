// Toast notifications
const toastContainer = document.getElementById('toast-container');

export function showToast(message, type = 'info', duration = 4000) {
    const toast = document.createElement('div');
    toast.className = `toast toast--${type}`;
    toast.innerHTML = `
        <span class="toast__message">${escapeHtml(message)}</span>
        <button class="toast__close" aria-label="Dismiss">&times;</button>
    `;
    toast.querySelector('.toast__close').addEventListener('click', () => removeToast(toast));
    toastContainer.appendChild(toast);

    if (duration > 0) {
        setTimeout(() => removeToast(toast), duration);
    }
    return toast;
}

function removeToast(toast) {
    if (!toast.parentNode) return;
    toast.classList.add('is-removing');
    toast.addEventListener('animationend', () => toast.remove());
}

// Modal
export function openModal(id) {
    document.getElementById(id).classList.add('is-open');
}

export function closeModal(id) {
    document.getElementById(id).classList.remove('is-open');
}

export function setupModals() {
    document.querySelectorAll('[data-modal-close]').forEach(btn => {
        btn.addEventListener('click', () => {
            const modal = btn.closest('.modal-overlay');
            if (modal) modal.classList.remove('is-open');
        });
    });

    document.querySelectorAll('.modal-overlay').forEach(overlay => {
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.classList.remove('is-open');
        });
    });
}

// Upload zone
export function setupUploadZone(zoneId, inputId, listId, onFilesChanged) {
    const zone = document.getElementById(zoneId);
    const input = document.getElementById(inputId);
    const list = document.getElementById(listId);
    let files = [];

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            input.click();
        }
    });

    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('is-dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('is-dragover'));
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('is-dragover');
        addFiles(Array.from(e.dataTransfer.files));
    });

    input.addEventListener('change', () => {
        addFiles(Array.from(input.files));
        input.value = '';
    });

    function addFiles(newFiles) {
        const valid = newFiles.filter(f =>
            f.name.endsWith('.csv') || f.name.endsWith('.xlsx') || f.name.endsWith('.xls')
        );
        files = [...files, ...valid];
        renderList();
        onFilesChanged(files);
    }

    function renderList() {
        if (files.length === 0) {
            list.innerHTML = '';
            zone.classList.remove('has-files');
            return;
        }
        zone.classList.add('has-files');
        list.innerHTML = files.map((f, i) =>
            `<li>${escapeHtml(f.name)} (${formatSize(f.size)})
                <button type="button" class="btn btn--sm btn--ghost" data-remove="${i}">&times;</button>
            </li>`
        ).join('');
        list.querySelectorAll('[data-remove]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                files.splice(parseInt(btn.dataset.remove), 1);
                renderList();
                onFilesChanged(files);
            });
        });
    }

    return {
        getFiles: () => files,
        clear: () => {
            files = [];
            renderList();
            onFilesChanged(files);
        }
    };
}

// Populate dataset select dropdowns
export function populateDatasetSelect(selectId, datasets, selectedId) {
    const select = document.getElementById(selectId);
    const currentVal = selectedId || select.value;
    select.innerHTML = '<option value="">Select a dataset...</option>';
    datasets.forEach(d => {
        const opt = document.createElement('option');
        opt.value = d.id;
        opt.textContent = d.name;
        if (String(d.id) === String(currentVal)) opt.selected = true;
        select.appendChild(opt);
    });
}

// Helpers
export function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

export function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export function show(el) {
    if (typeof el === 'string') el = document.getElementById(el);
    el.classList.remove('hidden');
    el.removeAttribute('hidden');
}

export function hide(el) {
    if (typeof el === 'string') el = document.getElementById(el);
    el.classList.add('hidden');
}

export function setLoading(btnId, loading) {
    const btn = document.getElementById(btnId);
    btn.disabled = loading;
    if (loading) {
        btn.dataset.origText = btn.textContent;
        btn.innerHTML = '<span class="spinner spinner--sm"></span> Processing...';
    } else {
        btn.textContent = btn.dataset.origText || btn.textContent;
    }
}
