import * as api from './api.js';
import { showToast, show, hide, escapeHtml, openPlotModal } from './components.js';
import { getTrainingResults } from './training.js';

let chartInstance = null;
let currentPercentileCurves = null;
let currentAgeGrid = null;

const PERCENTILE_COLORS = {
    '0.05': { line: '#ef4444', fill: 'rgba(239, 68, 68, 0.08)' },
    '0.1':  { line: '#f97316', fill: 'rgba(249, 115, 22, 0.08)' },
    '0.25': { line: '#eab308', fill: 'rgba(234, 179, 8, 0.08)' },
    '0.5':  { line: '#22c55e', fill: 'rgba(34, 197, 94, 0.1)' },
    '0.75': { line: '#3b82f6', fill: 'rgba(59, 130, 246, 0.08)' },
    '0.9':  { line: '#8b5cf6', fill: 'rgba(139, 92, 246, 0.08)' },
    '0.95': { line: '#ec4899', fill: 'rgba(236, 72, 153, 0.08)' },
};

function getPercentileColor(p) {
    const key = String(p);
    return PERCENTILE_COLORS[key] || { line: '#6b7280', fill: 'rgba(107, 114, 128, 0.08)' };
}

export function initResults() {
    const datasetSelect = document.getElementById('results-dataset-select');
    const structureSelect = document.getElementById('results-structure-select');

    datasetSelect.addEventListener('change', async () => {
        structureSelect.innerHTML = '<option value="">Select a structure...</option>';
        structureSelect.disabled = true;
        hide('results-chart-area');
        hide('results-ref-data');
        show('results-empty');

        const datasetId = datasetSelect.value;
        if (!datasetId) return;

        try {
            const detail = await api.datasets.get(datasetId);
            if (detail.models.length === 0) {
                showToast('No fitted models for this dataset', 'warning');
                return;
            }

            detail.models.forEach(m => {
                const opt = document.createElement('option');
                opt.value = m.structure;
                opt.textContent = m.structure;
                structureSelect.appendChild(opt);
            });
            structureSelect.disabled = false;
        } catch (err) {
            showToast(err.message, 'error');
        }
    });

    structureSelect.addEventListener('change', () => {
        const structure = structureSelect.value;
        const datasetId = datasetSelect.value;
        if (!structure || !datasetId) return;
        loadPercentileCurves(datasetId, structure);
    });

    document.getElementById('btn-reference-plot').addEventListener('click', () => {
        const datasetId = datasetSelect.value;
        const structure = structureSelect.value;
        if (!datasetId || !structure) return;
        const url = api.calculations.getReferencePlotUrl(datasetId, structure);
        openPlotModal(`Reference Plot: ${structure}`, url, api.getToken());
    });
}

async function loadPercentileCurves(datasetId, structure) {
    hide('results-empty');

    // Try to get curves from training results first, then from dataset detail
    const training = getTrainingResults();
    let modelResult = null;

    if (training && training.results && training.results[structure]) {
        modelResult = training.results[structure];
    } else {
        try {
            const detail = await api.datasets.get(datasetId);
            // Detail doesn't contain percentile curves - we'd need to refit
            // For now, check if training results are available
            if (!modelResult) {
                showToast('Percentile curves are available after training. Please train models first or switch to a recently trained structure.', 'info', 6000);
                show('results-empty');
                return;
            }
        } catch (err) {
            showToast(err.message, 'error');
            return;
        }
    }

    if (!modelResult || !modelResult.percentile_curves) {
        showToast('No percentile curves available for this structure', 'warning');
        show('results-empty');
        return;
    }

    currentPercentileCurves = modelResult.percentile_curves;
    show('results-chart-area');

    // Load reference data for scatter plot
    let refData = [];
    try {
        const tableData = await api.data.getTable(datasetId);
        refData = tableData.rows
            .filter(r => r.patient_age != null && r[structure] != null)
            .map(r => ({ x: r.patient_age, y: r[structure] }));
    } catch {
        // Non-critical - chart will just lack scatter points
    }

    renderChart(structure, currentPercentileCurves, refData);
    renderRefTable(currentPercentileCurves);
}

function renderChart(structure, curves, refData) {
    const canvas = document.getElementById('results-chart');
    if (chartInstance) chartInstance.destroy();

    const percentileKeys = Object.keys(curves).sort((a, b) => parseFloat(a) - parseFloat(b));
    const nPoints = curves[percentileKeys[0]].length;

    // Create age grid (we don't have exact ages from the model, so use indices)
    // The model generates 200 prediction points across the age range
    // Try to infer from reference data
    let xMin = 0, xMax = nPoints - 1;
    if (refData.length > 0) {
        xMin = Math.min(...refData.map(d => d.x));
        xMax = Math.max(...refData.map(d => d.x));
    }
    currentAgeGrid = Array.from({ length: nPoints }, (_, i) =>
        xMin + (i / (nPoints - 1)) * (xMax - xMin)
    );

    const datasets = [];

    // Percentile lines
    percentileKeys.forEach(p => {
        const color = getPercentileColor(p);
        const pct = (parseFloat(p) * 100).toFixed(0);
        datasets.push({
            label: `${pct}th percentile`,
            data: curves[p].map((y, i) => ({ x: currentAgeGrid[i], y })),
            borderColor: color.line,
            backgroundColor: color.fill,
            borderWidth: parseFloat(p) === 0.5 ? 2.5 : 1.5,
            pointRadius: 0,
            fill: false,
            tension: 0.3,
            order: 1,
        });
    });

    // Reference data scatter
    if (refData.length > 0) {
        datasets.push({
            label: 'Reference data',
            data: refData,
            backgroundColor: 'rgba(148, 163, 184, 0.4)',
            borderColor: 'rgba(148, 163, 184, 0.6)',
            pointRadius: 3,
            pointHoverRadius: 5,
            showLine: false,
            order: 2,
        });
    }

    chartInstance = new Chart(canvas, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: `Percentile Curves: ${structure}`,
                    font: { size: 16, weight: '600' },
                },
                legend: {
                    position: 'bottom',
                    labels: { usePointStyle: true, padding: 16 },
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const ds = ctx.dataset.label;
                            return `${ds}: Age ${ctx.parsed.x.toFixed(1)}, Value ${ctx.parsed.y.toFixed(0)}`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Age (years)' },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                },
                y: {
                    title: { display: true, text: `Volume (${structure})` },
                    grid: { color: 'rgba(0,0,0,0.06)' },
                },
            },
        },
    });
}

function renderRefTable(curves) {
    const percentileKeys = Object.keys(curves).sort((a, b) => parseFloat(a) - parseFloat(b));
    if (!currentAgeGrid || percentileKeys.length === 0) {
        hide('results-ref-data');
        return;
    }

    show('results-ref-data');

    const thead = document.getElementById('results-ref-thead').querySelector('tr');
    thead.innerHTML = '<th>Age</th>' + percentileKeys.map(p =>
        `<th>${(parseFloat(p) * 100).toFixed(0)}th</th>`
    ).join('');

    const tbody = document.getElementById('results-ref-tbody');
    // Show a sampled subset (every 10th point)
    const step = Math.max(1, Math.floor(currentAgeGrid.length / 20));
    const rows = [];
    for (let i = 0; i < currentAgeGrid.length; i += step) {
        rows.push(`<tr>
            <td class="numeric">${currentAgeGrid[i].toFixed(2)}</td>
            ${percentileKeys.map(p => `<td class="numeric">${curves[p][i].toFixed(0)}</td>`).join('')}
        </tr>`);
    }
    tbody.innerHTML = rows.join('');
}

export function addPatientOverlay(patients) {
    if (!chartInstance) return;

    const patientDataset = {
        label: 'Patients (OOS)',
        data: patients.map(p => ({ x: p.age, y: p.value })),
        backgroundColor: 'rgba(220, 38, 38, 0.8)',
        borderColor: '#dc2626',
        pointRadius: 6,
        pointHoverRadius: 8,
        pointStyle: 'triangle',
        showLine: false,
        order: 0,
    };

    chartInstance.data.datasets = chartInstance.data.datasets.filter(
        ds => ds.label !== 'Patients (OOS)'
    );
    chartInstance.data.datasets.push(patientDataset);
    chartInstance.update();
}
