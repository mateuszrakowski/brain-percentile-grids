import * as api from './api.js';
import { showToast, openModal, closeModal, escapeHtml, show, hide, populateDatasetSelect } from './components.js';

let cachedDatasets = [];

export function getCachedDatasets() {
    return cachedDatasets;
}

export async function loadDatasets() {
    const grid = document.getElementById('datasets-grid');
    const empty = document.getElementById('datasets-empty');
    const loading = document.getElementById('datasets-loading');

    show(loading);
    hide(empty);
    grid.innerHTML = '';

    try {
        const result = await api.datasets.list();
        cachedDatasets = result.datasets;
        hide(loading);

        if (cachedDatasets.length === 0) {
            show(empty);
            return;
        }

        grid.innerHTML = cachedDatasets.map(d => `
            <div class="card card--clickable" data-dataset-id="${d.id}">
                <div class="card__header">
                    <h3 class="card__title">${escapeHtml(d.name)}</h3>
                    <div>
                        ${d.has_models
                            ? '<span class="badge badge--success">Models</span>'
                            : '<span class="badge badge--neutral">No models</span>'}
                    </div>
                </div>
                ${d.description ? `<p class="card__description">${escapeHtml(d.description)}</p>` : ''}
                <div class="card__meta">
                    <span class="card__meta-item">${d.sample_count} records</span>
                    <span class="card__meta-item">${d.structures.length} structures</span>
                    <span class="card__meta-item">${new Date(d.created_at).toLocaleDateString()}</span>
                </div>
            </div>
        `).join('');

        grid.querySelectorAll('[data-dataset-id]').forEach(card => {
            card.addEventListener('click', () => showDatasetDetail(card.dataset.datasetId));
        });

    } catch (err) {
        hide(loading);
        showToast(err.message, 'error');
    }
}

async function showDatasetDetail(datasetId) {
    const listView = document.getElementById('datasets-list-view');
    const detailView = document.getElementById('datasets-detail-view');

    hide(listView);
    show(detailView);

    try {
        const detail = await api.datasets.get(datasetId);

        document.getElementById('detail-dataset-name').textContent = detail.name;
        document.getElementById('detail-dataset-desc').textContent = detail.description || '';

        const badges = document.getElementById('detail-dataset-badges');
        badges.innerHTML = `
            <span class="badge badge--info">${detail.sample_count} records</span>
            <span class="badge badge--info">${detail.structures.length} structures</span>
            ${detail.models.length > 0
                ? `<span class="badge badge--success">${detail.models.length} models</span>`
                : '<span class="badge badge--neutral">No models</span>'}
        `;

        document.getElementById('detail-dataset-meta').innerHTML =
            `Created: ${new Date(detail.created_at).toLocaleString()}`;

        // Load data table
        if (detail.sample_count > 0) {
            show('detail-data-table');
            hide('detail-data-empty');
            loadDataTable(datasetId);
        } else {
            hide('detail-data-table');
            show('detail-data-empty');
        }

        // Models table
        const modelsTbody = document.getElementById('detail-models-tbody');
        if (detail.models.length > 0) {
            show('detail-models-table');
            hide('detail-models-empty');
            modelsTbody.innerHTML = detail.models.map(m => `
                <tr>
                    <td>${escapeHtml(m.structure)}</td>
                    <td>${escapeHtml(m.family)}</td>
                    <td class="numeric">${m.aic.toFixed(2)}</td>
                    <td class="numeric">${m.bic.toFixed(2)}</td>
                    <td>${new Date(m.created_at).toLocaleString()}</td>
                </tr>
            `).join('');
        } else {
            hide('detail-models-table');
            show('detail-models-empty');
        }

    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function loadDataTable(datasetId) {
    const container = document.getElementById('detail-data-table');
    try {
        const result = await api.data.getTable(datasetId);
        const table = container.querySelector('table');

        // Build header
        const thead = table.querySelector('thead tr');
        thead.innerHTML = result.columns.map(col =>
            `<th>${escapeHtml(col)}</th>`
        ).join('');

        // Build body
        const tbody = document.getElementById('detail-data-tbody');
        tbody.innerHTML = result.rows.map(row =>
            `<tr>${result.columns.map(col => {
                const val = row[col];
                if (val == null) return '<td>-</td>';
                if (typeof val === 'number') return `<td class="numeric">${Number(val).toFixed(2)}</td>`;
                return `<td>${escapeHtml(String(val))}</td>`;
            }).join('')}</tr>`
        ).join('');
    } catch (err) {
        showToast('Failed to load data table: ' + err.message, 'error');
    }
}

export function initDatasets() {
    // Back button
    document.getElementById('btn-back-datasets').addEventListener('click', () => {
        hide('datasets-detail-view');
        show('datasets-list-view');
    });

    // New dataset modal
    document.getElementById('btn-new-dataset').addEventListener('click', () => {
        openModal('modal-create-dataset');
    });

    document.getElementById('form-create-dataset').addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = document.getElementById('dataset-name').value.trim();
        const description = document.getElementById('dataset-description').value.trim();

        if (!name) {
            showToast('Dataset name is required', 'error');
            return;
        }

        try {
            await api.datasets.create(name, description || null);
            closeModal('modal-create-dataset');
            document.getElementById('form-create-dataset').reset();
            showToast(`Dataset "${name}" created`, 'success');
            await loadDatasets();
        } catch (err) {
            showToast(err.message, 'error');
        }
    });
}
