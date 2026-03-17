import * as api from './api.js';
import { showToast, setupUploadZone, show, hide, setLoading, escapeHtml, openPlotModal } from './components.js';

let uploadZone;
let lastCalculationId = null;

export function initOos() {
    uploadZone = setupUploadZone('oos-upload-zone', 'oos-file-input', 'oos-file-list', (files) => {
        document.getElementById('btn-oos-calculate').disabled =
            files.length === 0 || !document.getElementById('oos-dataset-select').value;
    });

    document.getElementById('oos-dataset-select').addEventListener('change', () => {
        const hasDataset = !!document.getElementById('oos-dataset-select').value;
        const hasFiles = uploadZone.getFiles().length > 0;
        document.getElementById('btn-oos-calculate').disabled = !hasDataset || !hasFiles;
        if (hasDataset) {
            loadCalculationHistory();
        }
    });

    document.getElementById('btn-oos-calculate').addEventListener('click', doCalculate);
    document.getElementById('btn-oos-refresh-history').addEventListener('click', loadCalculationHistory);
    document.getElementById('oos-show-stale').addEventListener('change', loadCalculationHistory);

    // Event delegation for history list (attached once, handles all dynamic content)
    document.getElementById('oos-history-list').addEventListener('click', handleHistoryClick);

    // Event delegation for current results table plot buttons (attached once)
    document.getElementById('oos-results-tbody').addEventListener('click', handlePlotClick);
}

async function doCalculate() {
    const datasetId = document.getElementById('oos-dataset-select').value;
    const files = uploadZone.getFiles();

    if (!datasetId || files.length === 0) return;

    setLoading('btn-oos-calculate', true);
    hide('oos-results');

    try {
        const result = await api.calculations.calculate(datasetId, files);

        lastCalculationId = result.calculation_id || null;

        show('oos-results');
        document.getElementById('oos-results-summary').textContent =
            `${result.patients_processed} patients, ${result.structures_processed} structures`;

        renderOosResultsTable(
            document.getElementById('oos-results-tbody'),
            result.results,
            datasetId,
            lastCalculationId
        );

        // Errors
        if (result.errors && result.errors.length > 0) {
            show('oos-errors');
            document.getElementById('oos-errors-list').innerHTML =
                result.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('');
        } else {
            hide('oos-errors');
        }

        showToast(result.message, 'success');

        // Auto-refresh calculation history
        loadCalculationHistory();

    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading('btn-oos-calculate', false);
    }
}

function renderOosResultsTable(tbody, results, datasetId, calculationId) {
    tbody.innerHTML = results.map(r => {
        const pctClass = r.percentile != null && (r.percentile < 5 || r.percentile > 95)
            ? 'text-error' : '';
        const hasPlot = r.has_plot || (calculationId != null);
        const plotBtn = hasPlot && calculationId && r.id
            ? `<button type="button" class="btn--plot" data-plot-url="${escapeHtml(api.calculations.getResultPlotUrl(datasetId, calculationId, r.id))}" data-plot-title="${escapeHtml(r.patient_id)} - ${escapeHtml(r.structure)}">View</button>`
            : '-';
        return `<tr>
            <td>${escapeHtml(r.patient_id)}</td>
            <td>${escapeHtml(r.structure)}</td>
            <td class="numeric ${pctClass}">${r.percentile != null ? r.percentile.toFixed(1) + '%' : '-'}</td>
            <td class="numeric">${r.z_score != null ? r.z_score.toFixed(3) : '-'}</td>
            <td class="numeric">${r.age != null ? r.age.toFixed(2) : '-'}</td>
            <td class="numeric">${r.value != null ? r.value.toFixed(0) : '-'}</td>
            <td>${r.is_extrapolated
                ? '<span class="badge badge--warning">Yes</span>'
                : '<span class="badge badge--neutral">No</span>'}</td>
            <td>${plotBtn}</td>
        </tr>`;
    }).join('');
}

function handlePlotClick(e) {
    const btn = e.target.closest('[data-plot-url]');
    if (!btn) return;
    const url = btn.dataset.plotUrl;
    const title = btn.dataset.plotTitle || 'Patient Plot';
    openPlotModal(title, url, api.getToken());
}

async function loadCalculationHistory() {
    const datasetId = document.getElementById('oos-dataset-select').value;
    if (!datasetId) return;

    const includeStale = document.getElementById('oos-show-stale').checked;
    const listEl = document.getElementById('oos-history-list');
    const emptyEl = document.getElementById('oos-history-empty');
    const loadingEl = document.getElementById('oos-history-loading');

    listEl.innerHTML = '';
    hide(emptyEl);
    show(loadingEl);

    try {
        const data = await api.calculations.listCalculations(datasetId, includeStale);
        hide(loadingEl);

        if (!data.calculations || data.calculations.length === 0) {
            show(emptyEl);
            return;
        }

        listEl.innerHTML = data.calculations.map(calc => {
            const date = new Date(calc.created_at).toLocaleString();
            const fileCount = calc.source_filenames ? calc.source_filenames.length : 0;
            const stale = calc.is_stale
                ? '<span class="badge badge--warning">Stale</span>'
                : '';
            return `<div class="calc-history-card" data-calc-id="${calc.id}">
                <div class="calc-history-card__header">
                    <div class="calc-history-card__info">
                        <span class="calc-history-card__date">${escapeHtml(date)}</span>
                        <span class="calc-history-card__meta">
                            ${fileCount} file${fileCount !== 1 ? 's' : ''}
                            &middot; ${calc.patients_count} patients
                            &middot; ${calc.structures_count} structures
                        </span>
                        ${stale}
                    </div>
                    <div class="calc-history-card__actions">
                        <button
                            type="button"
                            class="btn--danger-sm"
                            data-delete-calc="${calc.id}"
                            title="Delete calculation"
                        >&times;</button>
                        <span class="calc-history-card__toggle" aria-hidden="true">&#x25B6;</span>
                    </div>
                </div>
                <div class="calc-history-card__body">
                    <div class="empty-state">
                        <div class="spinner spinner--sm"></div>
                        <p class="empty-state__text mt-sm">Loading results...</p>
                    </div>
                </div>
            </div>`;
        }).join('');

    } catch (err) {
        hide(loadingEl);
        showToast('Failed to load calculation history: ' + err.message, 'error');
    }
}

async function handleHistoryClick(e) {
    // Delete button
    const deleteBtn = e.target.closest('[data-delete-calc]');
    if (deleteBtn) {
        e.stopPropagation();
        const calcId = deleteBtn.dataset.deleteCalc;
        if (!confirm('Delete this calculation and its plots?')) return;

        const datasetId = document.getElementById('oos-dataset-select').value;
        try {
            await api.calculations.deleteCalculation(datasetId, calcId);
            showToast('Calculation deleted', 'success');
            loadCalculationHistory();
        } catch (err) {
            showToast('Failed to delete: ' + err.message, 'error');
        }
        return;
    }

    // Plot button inside expanded results
    const plotBtn = e.target.closest('[data-plot-url]');
    if (plotBtn) {
        const url = plotBtn.dataset.plotUrl;
        const title = plotBtn.dataset.plotTitle || 'Patient Plot';
        openPlotModal(title, url, api.getToken());
        return;
    }

    // Toggle expand/collapse
    const header = e.target.closest('.calc-history-card__header');
    if (!header) return;

    const card = header.closest('.calc-history-card');
    const wasExpanded = card.classList.contains('is-expanded');

    if (wasExpanded) {
        card.classList.remove('is-expanded');
        return;
    }

    card.classList.add('is-expanded');

    // Load results if body only contains the spinner placeholder
    const body = card.querySelector('.calc-history-card__body');
    if (body.querySelector('.spinner')) {
        const datasetId = document.getElementById('oos-dataset-select').value;
        const calcId = card.dataset.calcId;

        try {
            const calc = await api.calculations.getCalculation(datasetId, calcId);
            body.innerHTML = renderHistoryResultsTable(calc.results, datasetId, calcId);
        } catch (err) {
            body.innerHTML = `<p class="text-error">${escapeHtml(err.message)}</p>`;
        }
    }
}

function renderHistoryResultsTable(results, datasetId, calculationId) {
    if (!results || results.length === 0) {
        return '<p class="text-muted">No results in this calculation.</p>';
    }

    const rows = results.map(r => {
        const pctClass = r.percentile != null && (r.percentile < 5 || r.percentile > 95)
            ? 'text-error' : '';
        const plotBtn = r.has_plot
            ? `<button type="button" class="btn--plot" data-plot-url="${escapeHtml(api.calculations.getResultPlotUrl(datasetId, calculationId, r.id))}" data-plot-title="${escapeHtml(r.patient_id)} - ${escapeHtml(r.structure)}">View</button>`
            : '-';
        return `<tr>
            <td>${escapeHtml(r.patient_id)}</td>
            <td>${escapeHtml(r.structure)}</td>
            <td class="numeric ${pctClass}">${r.percentile != null ? r.percentile.toFixed(1) + '%' : '-'}</td>
            <td class="numeric">${r.z_score != null ? r.z_score.toFixed(3) : '-'}</td>
            <td class="numeric">${r.age != null ? r.age.toFixed(2) : '-'}</td>
            <td class="numeric">${r.value != null ? r.value.toFixed(0) : '-'}</td>
            <td>${r.is_extrapolated
                ? '<span class="badge badge--warning">Yes</span>'
                : '<span class="badge badge--neutral">No</span>'}</td>
            <td>${plotBtn}</td>
        </tr>`;
    }).join('');

    return `<div class="table-container">
        <table class="table table--compact">
            <thead>
                <tr>
                    <th>Patient ID</th>
                    <th>Structure</th>
                    <th>Percentile</th>
                    <th>Z-Score</th>
                    <th>Age</th>
                    <th>Value</th>
                    <th>Extrapolated</th>
                    <th>Plot</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    </div>`;
}
