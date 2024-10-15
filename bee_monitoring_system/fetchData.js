fetch('http://localhost:5000/data')
    .then(response => response.json())
    .then(result => {
        const data = result.data; // Access the 'data' array inside the result object
        const zeroPredictionCount = result.zero_predictions_count; // Get the zero_predictions_count directly

        console.log('Fetched data:', data);  // Log the fetched data in the console

        let latestDate = null; // Variable to hold the latest date
        let totalTemperature = 0; // Variable to sum up all temperatures
        let totalHumidity = 0; // Variable to sum up all humidity
        let totalTemperatureLatest = 0; // Variable to sum up temperatures for the latest date
        let totalHumidityLatest = 0; // Variable to sum up humidity for the latest date
        let itemCount = data.length; // Number of total items in the data
        let latestItemCount = 0; // Number of items with the latest date

        // Step 1: Find the latest date
        data.forEach(item => {
            // Convert item.Date to a JavaScript Date object
            const itemDate = new Date(item.Date);
            console.log('Current itemDate:', itemDate);

            // Update the latest date if necessary
            if (!latestDate || itemDate > latestDate) {
                latestDate = itemDate;
            }

            // Add up the temperature and humidity for all entries
            totalTemperature += item.Temperature;
            totalHumidity += item.Humidity;

            // Count zero predictions
            if (item.prediction === 0) {
                PredictionCount++;
            }
        });

        // Step 2: Filter entries that match the latest date and sum temperature and humidity
        data.forEach(item => {
            const itemDate = new Date(item.Date);
            // Check if this entry is from the latest date
            if (itemDate.toISOString().split('T')[0] === latestDate.toISOString().split('T')[0]) {
                totalTemperatureLatest += item.Temperature;
                totalHumidityLatest += item.Humidity;
                latestItemCount++;
            }
        });

        // Step 3: Calculate the mean for all data and the latest date
        const meanTemperature = totalTemperature / itemCount;
        const meanHumidity = totalHumidity / itemCount;
        const meanTemperatureLatest = totalTemperatureLatest / latestItemCount;
        const meanHumidityLatest = totalHumidityLatest / latestItemCount;

        // Update the card with the latest date and time
        if (latestDate) {
            const latestDateFormatted = latestDate.toLocaleDateString('en-CA'); // Date in YYYY-MM-DD format
            const latestTimeFormatted = latestDate.toLocaleTimeString('en-GB', { hour12: false }); // Time in HH:MM:SS format
            // Select the card elements and update their content
            document.querySelector('.small-card-update-time .card-text').textContent = latestDateFormatted;
            document.querySelector('.small-card-update-time .small-text').textContent = latestTimeFormatted;
        }

        // Update the card with the mean temperature for all data
        document.querySelector('.small-card-temperature .small-text').textContent = meanTemperature.toFixed(2);

        // Update the card with the mean humidity for all data
        document.querySelector('.small-card-humidity .small-text').textContent = meanHumidity.toFixed(2);

        // Update the card with the mean temperature for the latest date
        document.querySelector('.small-card-temperature .card-text').textContent = meanTemperatureLatest.toFixed(2);
        
        // Update the card with the mean humidity for the latest date
        document.querySelector('.small-card-humidity .card-text').textContent = meanHumidityLatest.toFixed(2);
        
        // Step 9: Update the Queen Bee Status card with the zero prediction count
        document.querySelector('.small-card-queen-bee-status .card-text').textContent = PredictionCount;

        if (zeroPredictionCount === 0) {
            document.querySelector('.small-card-queen-bee-status .small-text').textContent = 'Healthy';
        } else {
            document.querySelector('.small-card-queen-bee-status .small-text').textContent = 'Unhealthy';
        }
    })
    .catch(error => {
        console.error('Error fetching data:', error);  // Correctly catching and logging the error
    });
