import { isAuthenticated, auth } from './api.js';
import { setupModals, populateDatasetSelect, show, hide } from './components.js';
import { initAuth } from './auth.js';
import { initDatasets, loadDatasets, getCachedDatasets } from './datasets.js';
import { initUpload } from './upload.js';
import { initTraining } from './training.js';
import { initResults } from './results.js';
import { initOos } from './oos.js';

const DATASET_SELECTS = [
    'upload-dataset-select',
    'train-dataset-select',
    'results-dataset-select',
    'oos-dataset-select',
];

function initTabs() {
    const tabs = document.querySelectorAll('.tabs__btn');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => {
                t.classList.remove('is-active');
                t.setAttribute('aria-selected', 'false');
                t.setAttribute('tabindex', '-1');
            });
            panels.forEach(p => {
                p.classList.remove('is-active');
                p.setAttribute('hidden', '');
            });

            tab.classList.add('is-active');
            tab.setAttribute('aria-selected', 'true');
            tab.removeAttribute('tabindex');

            const panelId = tab.getAttribute('aria-controls');
            const panel = document.getElementById(panelId);
            panel.classList.add('is-active');
            panel.removeAttribute('hidden');

            // Refresh dataset selects when switching tabs
            refreshDatasetSelects();
        });
    });
}

function refreshDatasetSelects() {
    const datasets = getCachedDatasets();
    DATASET_SELECTS.forEach(id => populateDatasetSelect(id, datasets));
}

async function onLogin() {
    hide('view-auth');
    show('view-app');

    try {
        const user = await auth.me();
        const headerUser = document.getElementById('header-username');
        headerUser.textContent = user.username;
        headerUser.classList.remove('hidden');
        document.getElementById('btn-logout').classList.remove('hidden');
    } catch {
        // Token might be invalid
    }

    await loadDatasets();
    refreshDatasetSelects();
}

async function init() {
    setupModals();
    initTabs();
    initAuth(onLogin);
    initDatasets();
    initUpload();
    initTraining();
    initResults();
    initOos();

    if (isAuthenticated()) {
        await onLogin();
    }
}

init();
