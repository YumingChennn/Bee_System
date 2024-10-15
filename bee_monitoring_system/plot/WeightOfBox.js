// Set the default font and color
Chart.defaults.font.family = 'Nunito, -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
Chart.defaults.color = '#858796';

// Fetch the data from your server
fetch('http://localhost:5000/data')
    .then(response => response.json())
    .then(result => {
        const data = result.data; // Access the 'data' array inside the result object

        console.log('Fetched data:', data);  // Log the fetched data in the console

        let latestDate = null; // Variable to hold the latest date
        let boxes = []; // Array to hold the box numbers for the latest date
        let weights = []; // Array to hold the weights for the latest date

        // Step 1: Find the latest date
        data.forEach(item => {
            const itemDate = new Date(item.Date);

            // Update the latest date if necessary
            if (!latestDate || itemDate > latestDate) {
                latestDate = itemDate;
            }
        });

        // Step 2: Filter entries that match the latest date and get box numbers and weights
        data.forEach(item => {
            const itemDate = new Date(item.Date);
            // Check if this entry is from the latest date
            if (itemDate.toISOString().split('T')[0] === latestDate.toISOString().split('T')[0]) {
                boxes.push(`Box number ${item['Box Number']}`); // Add the box number to the array
                weights.push(item.Weight); // Add the weight to the array
            }
        });

        // Get the canvas element where the chart will be rendered
        const ctx2 = document.getElementById('horizontalBarChart').getContext('2d');

        // Create gradient for bars
        const gradient2 = ctx2.createLinearGradient(0, 0, 0, 400);
        gradient2.addColorStop(0, 'rgba(14, 203, 220, 0.5)');
        gradient2.addColorStop(1, 'rgba(14, 203, 220, 0.45)');

        // Create the horizontal bar chart
        const horizontalBarChart = new Chart(ctx2, {
            type: 'bar', // Bar chart type
            data: {
                labels: boxes, // Use extracted box numbers as labels
                datasets: [{
                    label: 'Weight (kg)', // Label for the dataset
                    data: weights, // Use extracted weights as data
                    backgroundColor: gradient2, // Apply gradient background to bars
                    borderColor: 'rgba(14, 203, 220, 1)',
                    borderWidth: 2,
                    borderRadius: 10,  // Add rounded corners
                    barThickness: 30,  // Control bar thickness
                    hoverBackgroundColor: 'rgba(14, 203, 220, 0.8)', // Change bar color on hover
                    hoverBorderColor: 'rgba(14, 203, 220, 1)', // Change border on hover
                    hoverBorderWidth: 2,
                }]
            },
            options: {
                indexAxis: 'y', // Horizontal bar
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: {
                            display: false, // Hide grid lines
                        },
                        beginAtZero: true,
                        ticks: {
                            color: '#fff'  // Make x-axis labels white
                        },
                    },
                    y: {
                        grid: {
                            display: false, // Hide grid lines
                        },
                        ticks: {
                            color: '#fff'  // Make y-axis labels white
                        },
                    }
                },
                plugins: {
                    legend: {
                        display: false // Disable legend
                    },
                    tooltip: {
                        enabled: true, // Ensure tooltips are enabled
                        backgroundColor: 'rgba(0, 0, 0, 0.7)', // Dark background for tooltips
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: 'rgba(14, 203, 220, 1)',
                        borderWidth: 1
                    },
                },
                animation: {
                    duration: 2000,
                    easing: 'easeOutBounce',
                },
                hover: {
                    mode: 'nearest',
                    intersect: true
                }
            },
            plugins: [ChartDataLabels] // Register the Data Labels plugin
        });
    })
    .catch(error => {
        console.error('Error fetching data:', error);  // Correctly catching and logging the error
    });
