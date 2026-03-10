import * as api from './api.js';
import { showToast, setupUploadZone, show, hide, setLoading, escapeHtml } from './components.js';

let uploadZone;

export function initOos() {
    uploadZone = setupUploadZone('oos-upload-zone', 'oos-file-input', 'oos-file-list', (files) => {
        document.getElementById('btn-oos-calculate').disabled =
            files.length === 0 || !document.getElementById('oos-dataset-select').value;
    });

    document.getElementById('oos-dataset-select').addEventListener('change', () => {
        const hasDataset = !!document.getElementById('oos-dataset-select').value;
        const hasFiles = uploadZone.getFiles().length > 0;
        document.getElementById('btn-oos-calculate').disabled = !hasDataset || !hasFiles;
    });

    document.getElementById('btn-oos-calculate').addEventListener('click', doCalculate);
}

async function doCalculate() {
    const datasetId = document.getElementById('oos-dataset-select').value;
    const files = uploadZone.getFiles();

    if (!datasetId || files.length === 0) return;

    setLoading('btn-oos-calculate', true);
    hide('oos-results');

    try {
        const result = await api.calculations.calculate(datasetId, files);

        show('oos-results');
        document.getElementById('oos-results-summary').textContent =
            `${result.patients_processed} patients, ${result.structures_processed} structures`;

        const tbody = document.getElementById('oos-results-tbody');
        tbody.innerHTML = result.results.map(r => {
            const pctClass = r.percentile != null && (r.percentile < 5 || r.percentile > 95)
                ? 'text-error' : '';
            return `<tr>
                <td>${escapeHtml(r.patient_id)}</td>
                <td>${escapeHtml(r.structure)}</td>
                <td class="numeric ${pctClass}">${r.percentile != null ? r.percentile.toFixed(1) + '%' : '-'}</td>
                <td class="numeric">${r.z_score != null ? r.z_score.toFixed(3) : '-'}</td>
                <td class="numeric">${r.age != null ? r.age.toFixed(2) : '-'}</td>
                <td class="numeric">${r.value.toFixed(0)}</td>
                <td>${r.is_extrapolated
                    ? '<span class="badge badge--warning">Yes</span>'
                    : '<span class="badge badge--neutral">No</span>'}</td>
            </tr>`;
        }).join('');

        // Errors
        if (result.errors && result.errors.length > 0) {
            show('oos-errors');
            document.getElementById('oos-errors-list').innerHTML =
                result.errors.map(e => `<li>${escapeHtml(e)}</li>`).join('');
        } else {
            hide('oos-errors');
        }

        showToast(result.message, 'success');

    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading('btn-oos-calculate', false);
    }
}
