const mongoose = require('mongoose');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
require('dotenv').config({ path: path.join(__dirname, '../../../.env') });

const MONGO_URI = process.env.MONGO_URI || 'mongodb+srv://conduitray:jDpdINYNgbKBS4Hq@cluster0.pxsiiwr.mongodb.net/agentic';
const CSV_PATH = path.join(__dirname, '../../../ml-service/data/quarterly_forecast_raw_predictions.csv');

async function upload() {
    try {
        console.log('Connecting to MongoDB...');
        await mongoose.connect(MONGO_URI);
        console.log('Connected.');

        const results = [];
        console.log(`Reading CSV from ${CSV_PATH}...`);

        await new Promise((resolve, reject) => {
            fs.createReadStream(CSV_PATH)
                .pipe(csv())
                .on('data', (data) => {
                    // Clean keys: "Customer ID" -> "customer_id"
                    const cleaned = {};
                    for (const key in data) {
                        const newKey = key.toLowerCase().replace(/ /g, '_');
                        cleaned[newKey] = data[key];
                    }
                    cleaned.timestamp = new Date();
                    results.push(cleaned);
                })
                .on('end', resolve)
                .on('error', reject);
        });

        if (results.length === 0) {
            console.log('No data found in CSV.');
            process.exit(0);
        }

        console.log(`Uploading ${results.length} records to 'forecasts' collection...`);
        
        // Use raw collection access to avoid defining a schema for this one-off upload
        const db = mongoose.connection.db;
        await db.collection('forecasts').insertMany(results);

        console.log('Upload successful!');
        process.exit(0);
    } catch (error) {
        console.error('Upload failed:', error);
        process.exit(1);
    }
}

upload();
