const fs = require('fs');
const path = require('path');

const OUTPUT_PATH = path.join(__dirname, '../ml-service/data/activity.csv');
const USERS = 100;
const MONTHS = 12;

function generate() {
    console.log(`Generating activity data for ${USERS} users over ${MONTHS} months...`);
    const header = 'user_id,month,year,txn_count,spend\n';
    let content = header;

    const currentYear = 2025;

    for (let u = 1; u <= USERS; u++) {
        const userId = `user_${u.toString().padStart(3, '0')}`;
        for (let m = 1; m <= MONTHS; m++) {
            const txns = Math.floor(Math.random() * 20) + 1;
            const spend = (Math.random() * 500 + 50).toFixed(2);
            content += `${userId},${m},${currentYear},${txns},${spend}\n`;
        }
    }

    fs.writeFileSync(OUTPUT_PATH, content);
    console.log(`Successfully generated activity data at ${OUTPUT_PATH}`);
}

generate();
