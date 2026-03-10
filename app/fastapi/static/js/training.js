import * as api from './api.js';
import { showToast, show, hide, setLoading, escapeHtml } from './components.js';
import { loadDatasets } from './datasets.js';

let trainingResults = null;

export function getTrainingResults() {
    return trainingResults;
}

export function initTraining() {
    const datasetSelect = document.getElementById('train-dataset-select');
    datasetSelect.addEventListener('change', onDatasetChange);
    document.getElementById('btn-train').addEventListener('click', startTraining);
}

async function onDatasetChange() {
    const datasetId = document.getElementById('train-dataset-select').value;
    const structuresDiv = document.getElementById('train-structures');
    const emptyDiv = document.getElementById('train-structures-empty');
    const btn = document.getElementById('btn-train');

    structuresDiv.innerHTML = '';
    btn.disabled = true;

    if (!datasetId) {
        show(emptyDiv);
        return;
    }

    try {
        const result = await api.data.structures(datasetId);
        if (result.structures.length === 0) {
            show(emptyDiv);
            return;
        }
        hide(emptyDiv);
        structuresDiv.innerHTML = result.structures.map(s => `
            <label class="checkbox-item">
                <input type="checkbox" value="${escapeHtml(s)}">
                ${escapeHtml(s)}
            </label>
        `).join('');
        btn.disabled = false;
    } catch (err) {
        showToast(err.message, 'error');
    }
}

async function startTraining() {
    const datasetId = document.getElementById('train-dataset-select').value;
    if (!datasetId) return;

    const checkedBoxes = document.querySelectorAll('#train-structures input:checked');
    const yColumns = checkedBoxes.length > 0
        ? Array.from(checkedBoxes).map(cb => cb.value)
        : null;

    const percentilesStr = document.getElementById('train-percentiles').value;
    const percentiles = percentilesStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));

    if (percentiles.some(p => p <= 0 || p >= 1)) {
        showToast('Percentiles must be between 0 and 1', 'error');
        return;
    }

    const body = {
        x_column: 'PatientAge',
        y_columns: yColumns,
        percentiles,
    };

    setLoading('btn-train', true);
    show('train-progress');
    hide('train-results');
    trainingResults = null;

    const progressBar = document.getElementById('train-progress-bar');
    const progressStatus = document.getElementById('train-progress-status');
    const progressStructure = document.getElementById('train-progress-structure');
    const progressPercent = document.getElementById('train-progress-percent');

    progressBar.style.width = '0%';
    progressBar.classList.remove('is-complete', 'is-error');
    progressStatus.textContent = 'Starting training...';
    progressStructure.textContent = '';
    progressPercent.textContent = '0%';

    try {
        const response = await api.calculations.fitStream(datasetId, body);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const eventData = JSON.parse(line.slice(6));

                if (eventData.type === 'progress') {
                    const pct = eventData.progress;
                    progressBar.style.width = pct + '%';
                    progressPercent.textContent = pct + '%';
                    progressStructure.textContent = eventData.structure || '';
                    progressStatus.textContent = eventData.message || 'Fitting...';
                } else if (eventData.type === 'complete') {
                    progressBar.style.width = '100%';
                    progressBar.classList.add('is-complete');
                    progressPercent.textContent = '100%';
                    progressStatus.textContent = eventData.message;

                    trainingResults = eventData;
                    showTrainingResults(eventData);
                    showToast(eventData.message, 'success');
                    await loadDatasets();
                }
            }
        }
    } catch (err) {
        progressBar.classList.add('is-error');
        progressStatus.textContent = 'Training failed: ' + err.message;
        showToast('Training failed: ' + err.message, 'error');
    } finally {
        setLoading('btn-train', false);
    }
}

function showTrainingResults(data) {
    show('train-results');

    document.getElementById('train-results-summary').innerHTML = `
        <span class="badge badge--success">${data.successful_count} succeeded</span>
        ${data.failed_count > 0 ? `<span class="badge badge--error">${data.failed_count} failed</span>` : ''}
        <span class="text-muted ml-sm">Total time: ${data.total_time.toFixed(1)}s</span>
    `;

    const tbody = document.getElementById('train-results-tbody');
    tbody.innerHTML = Object.entries(data.results).map(([structure, r]) => `
        <tr>
            <td>${escapeHtml(r.structure || structure)}</td>
            <td>${r.family ? escapeHtml(r.family) : '-'}</td>
            <td class="numeric">${r.aic != null ? r.aic.toFixed(2) : '-'}</td>
            <td class="numeric">${r.bic != null ? r.bic.toFixed(2) : '-'}</td>
            <td>
                ${r.converged
                    ? '<span class="badge badge--success">Converged</span>'
                    : `<span class="badge badge--error">Failed</span>${r.error ? ` <span class="text-muted">${escapeHtml(r.error)}</span>` : ''}`
                }
            </td>
        </tr>
    `).join('');
}
