const { ChartJSNodeCanvas } = require('chartjs-node-canvas');

const width = 600;
const height = 400;

// Dark theme colors to match the dashboard
const colors = {
    high: '#ef4444',   // Red-500
    medium: '#f59e0b', // Amber-500
    low: '#10b981',    // Emerald-500
    primary: '#6366f1', // Indigo-500
    grid: '#334155',    // Slate-700
    text: '#94a3b8'     // Slate-400
};

const chartJSNodeCanvas = new ChartJSNodeCanvas({ 
    width, 
    height, 
    backgroundColour: 'transparent' 
});

/**
 * Generate Risk Segmentation Pie Chart
 */
exports.generateRiskChart = async (data) => {
    const configuration = {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                data: [
                    data.high_risk_count || 0,
                    data.medium_risk_count || 0,
                    data.low_risk_count || 0
                ],
                backgroundColor: [colors.high, colors.medium, colors.low],
                borderColor: '#1e293b', // Slate-900
                borderWidth: 2
            }]
        },
        options: {
            plugins: {
                legend: {
                    display: true,
                    labels: { color: colors.text, font: { size: 14 } }
                }
            }
        }
    };
    const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration);
    return `data:image/png;base64,${imageBuffer.toString('base64')}`;
};

/**
 * Generate User Activity Line Chart
 */
exports.generateTrendChart = async (trendData = []) => {
    // If no data provided, use dummy trend for visual consistency
    const labels = trendData.length > 0 ? trendData.map(d => d.month) : ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'];
    const values = trendData.length > 0 ? trendData.map(d => d.value) : [65, 59, 80, 81, 56, 55];

    const configuration = {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label: 'Avg Churn Risk',
                data: values,
                fill: true,
                backgroundColor: 'rgba(99, 102, 241, 0.1)',
                borderColor: colors.primary,
                tension: 0.4,
                pointBackgroundColor: colors.primary
            }]
        },
        options: {
            scales: {
                y: {
                    grid: { color: colors.grid },
                    ticks: { color: colors.text }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: colors.text }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    };
    const imageBuffer = await chartJSNodeCanvas.renderToBuffer(configuration);
    return `data:image/png;base64,${imageBuffer.toString('base64')}`;
};
