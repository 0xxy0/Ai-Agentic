const puppeteer = require('puppeteer');
const ejs = require('ejs');
const path = require('path');
const chartService = require('./chart.service');

/**
 * Orchestrate PDF Generation
 * 1. Generate base64 charts
 * 2. Render EJS template to HTML
 * 3. Use Puppeteer to convert HTML to PDF
 */
exports.generateChurnReport = async (dashboardData) => {
    try {
        // 1. Generate Charts
        const riskChart = await chartService.generateRiskChart(dashboardData.summary);
        // We mock trend data for now since it's not in the dashboard JSON
        const trendChart = await chartService.generateTrendChart();

        // 2. Prepare Template Data
        const templateData = {
            ...dashboardData,
            riskChart,
            trendChart,
            generated_at: dashboardData.generated_at || new Date().toISOString()
        };

        // 3. Render HTML using EJS
        const templatePath = path.join(__dirname, '../templates/report.ejs');
        const html = await ejs.renderFile(templatePath, templateData);

        // 4. Launch Puppeteer
        const browser = await puppeteer.launch({
            headless: 'new',
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });

        const page = await browser.newPage();
        
        // Set content and wait for it to be fully loaded
        await page.setContent(html, { waitUntil: 'networkidle0' });

        // 5. Generate PDF
        const pdfBuffer = await page.pdf({
            format: 'A4',
            printBackground: true,
            margin: {
                top: '0px',
                right: '0px',
                bottom: '0px',
                left: '0px'
            }
        });

        await browser.close();
        return pdfBuffer;

    } catch (error) {
        console.error('PDF Generation Service Error:', error);
        throw error;
    }
};
