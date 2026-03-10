import * as api from './api.js';
import { showToast, setupUploadZone, show, hide, setLoading, escapeHtml } from './components.js';
import { loadDatasets } from './datasets.js';

let uploadZone;

export function initUpload() {
    uploadZone = setupUploadZone('upload-zone', 'upload-file-input', 'upload-file-list', (files) => {
        document.getElementById('btn-upload').disabled =
            files.length === 0 || !document.getElementById('upload-dataset-select').value;
    });

    document.getElementById('upload-dataset-select').addEventListener('change', () => {
        const hasDataset = !!document.getElementById('upload-dataset-select').value;
        const hasFiles = uploadZone.getFiles().length > 0;
        document.getElementById('btn-upload').disabled = !hasDataset || !hasFiles;
    });

    document.getElementById('btn-upload').addEventListener('click', doUpload);
}

async function doUpload() {
    const datasetId = document.getElementById('upload-dataset-select').value;
    const files = uploadZone.getFiles();

    if (!datasetId || files.length === 0) return;

    setLoading('btn-upload', true);
    hide('upload-results');

    try {
        const result = await api.data.upload(datasetId, files);
        const info = result.processing_info;

        show('upload-results');
        document.getElementById('upload-results-content').innerHTML = `
            <p><strong>${escapeHtml(result.message)}</strong></p>
            <div class="card__meta mt-sm">
                <span class="card__meta-item">Files processed: ${info.files_processed}</span>
                <span class="card__meta-item">Records added: ${info.records_added}</span>
                <span class="card__meta-item">Duplicates found: ${info.duplicates_found}</span>
                <span class="card__meta-item">Total records: ${info.total_records}</span>
            </div>
            ${info.structures.length > 0 ? `
                <p class="mt-sm text-muted">Structures: ${info.structures.map(s => escapeHtml(s)).join(', ')}</p>
            ` : ''}
        `;

        showToast(result.message, 'success');
        uploadZone.clear();
        await loadDatasets();

    } catch (err) {
        showToast(err.message, 'error');
    } finally {
        setLoading('btn-upload', false);
    }
}
