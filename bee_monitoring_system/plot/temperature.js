// Set the default font and color
Chart.defaults.font.family = 'Nunito, -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
Chart.defaults.color = '#858796';

// Fetch the data
fetch('http://localhost:5000/data')
    .then(response => response.json())
    .then(result => {
        const data = result.data; // Access the 'data' array inside the result object
        console.log('Fetched data:', data);  // This will print the fetched data in the console
        
        let latestDate = null; // Variable to hold the latest date
        const temperatureMap = new Map(); // To hold temperature sums and counts for mean calculation

        // Populate the table with data
        data.forEach(item => {
            // Convert item.Date to a JavaScript Date object
            const itemDate = new Date(item.Date);
            console.log('Current itemDate:', itemDate);

            // Update latest date if necessary
            if (!latestDate || itemDate > latestDate) {
                latestDate = itemDate;
            }

            // Check if this entry is from the latest date
            if (itemDate.toISOString().split('T')[0] === latestDate.toISOString().split('T')[0]) {
                const boxNumber = `Box number ${item['Box Number']}`;
                const temperature = item.Temperature;

                // Calculate mean temperature
                if (!temperatureMap.has(boxNumber)) {
                    temperatureMap.set(boxNumber, { sum: 0, count: 0 });
                }
                const currentBoxData = temperatureMap.get(boxNumber);
                currentBoxData.sum += temperature;
                currentBoxData.count += 1;
            }
        });

        // Prepare data for the chart
        const boxLabels = Array.from(temperatureMap.keys());
        const meanTemperatures = Array.from(temperatureMap.values()).map(boxData => boxData.sum / boxData.count);

        // Get the canvas element where the chart will be rendered
        const ctx = document.getElementById('Chart_temp').getContext('2d');

        // Create gradient for the line chart
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(252, 176, 77, 0.5)');
        gradient.addColorStop(1, 'rgba(252, 176, 77, 0)');

        // Create a new line chart instance
        const myLineChart1 = new Chart(ctx, {
            type: 'line',
            data: {
                labels: boxLabels, // Use the box labels for the x-axis
                datasets: [
                    {
                        label: 'Mean Temperature',
                        lineTension: 0.4,
                        backgroundColor: gradient,  // Apply the gradient background
                        borderColor: "rgba(252, 176, 77, 1)", // Line border color
                        borderWidth: 2,
                        pointRadius: 5,
                        pointBackgroundColor: "rgba(252, 176, 77, 1)",
                        pointBorderColor: "#fff",
                        pointHoverRadius: 7,
                        pointHoverBackgroundColor: "#fff",
                        pointHoverBorderColor: "rgba(252, 176, 77, 1)",
                        fill: true,  // Ensure the area below the line is filled
                        data: meanTemperatures, // Mean temperatures for y-axis
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        left: 0,
                        right: 0,
                        top: 0,
                        bottom: 0
                    }
                },
                scales: {
                    x: {
                      grid: {
                        display: false,
                      },
                      ticks: {
                        color: '#fff'  // Make x-axis labels white
                      }
                    },
                    y: {
                      grid: {
                        display: true,
                        color: "rgba(0, 0, 0, 0.1)",
                        drawBorder: false,
                        borderDash: [5, 5],
                      },
                      ticks: {
                        color: '#fff'  // Make y-axis labels white
                      }
                    },
                },
                plugins: {
                    legend: {
                        display: false,
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.7)',
                        titleFont: { size: 16, family: 'Nunito' },
                        bodyFont: { size: 14, family: 'Nunito' },
                        padding: 10,
                        cornerRadius: 8,
                        callbacks: {
                            label: function(context) {
                                return `${context.dataset.label}: ${context.parsed.y}`;
                            }
                        }
                    },
                    annotation: {
                        annotations: {
                            line1: {
                                type: 'line',
                                yMin: 25,
                                yMax: 25,
                                borderColor: 'red',
                                borderWidth: 1,
                                label: {
                                    content: 'Threshold',
                                    enabled: true,
                                    position: 'start'
                                }
                            }
                        }
                    },
                    datalabels: {
                        align: 'top',
                        anchor: 'end',
                        backgroundColor: 'rgba(54, 162, 235, 0.7)',
                        borderRadius: 4,
                        color: '#fff',
                        font: {
                            weight: 'bold'
                        },
                        formatter: (value) => value,
                    },
                },
                animation: {
                    duration: 2000,
                    easing: 'easeOutBounce',
                },
                hover: {
                    mode: 'nearest',
                    intersect: true,
                },
            },
        });
    })
    .catch(error => {
        console.error('Error fetching data:', error);  // Correctly catching and logging the error
    });
