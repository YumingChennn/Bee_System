// Set default font and color
Chart.defaults.font.family = 'Nunito, -apple-system, system-ui, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
Chart.defaults.color = '#858796';

// Fetch the data
fetch('http://localhost:5000/data')
    .then(response => response.json())
    .then(result => {
        const data = result.data; // Access the 'data' array inside the result object
        console.log('Fetched data:', data);  // This will print the fetched data in the console
        
        let latestDate = null; // Variable to hold the latest date
        const humidityMap = new Map(); // To hold humidity sums and counts for mean calculation

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
                const humidity = item.Humidity;

                // Calculate mean humidity
                if (!humidityMap.has(boxNumber)) {
                  humidityMap.set(boxNumber, { sum: 0, count: 0 });
                }
                const currentBoxData = humidityMap.get(boxNumber);
                currentBoxData.sum += humidity;
                currentBoxData.count += 1;
            }
        });

        // Prepare data for the chart
        const boxLabels_humi = Array.from(humidityMap.keys());
        const meanHumidity = Array.from(humidityMap.values()).map(boxData => boxData.sum / boxData.count);

        // Get the canvas element
        const ctx1 = document.getElementById('Chart_humi').getContext('2d');

        // Create gradient
        const gradient1 = ctx1.createLinearGradient(0, 0, 0, 400);
        gradient1.addColorStop(0, 'rgba(0, 148, 254, 0.5)');
        gradient1.addColorStop(1, 'rgba(0, 148, 254, 0)');

        // Create a new line chart instance
        const myLineChart2 = new Chart(ctx1, {
          type: 'line',
          data: {
            labels: boxLabels_humi,
            datasets: [{
              label: 'Mean Humidity',
              lineTension: 0.4,
              backgroundColor: gradient1,
              borderColor: "rgba(0, 148, 254, 1)",
              borderWidth: 2,
              pointRadius: 5,
              pointBackgroundColor: "rgba(0, 148, 254, 1)",
              pointBorderColor: "#fff",
              pointHoverRadius: 7,
              pointHoverBackgroundColor: "#fff",
              pointHoverBorderColor: "rgba(0, 148, 254, 1)",
              pointHoverBorderWidth: 2,
              fill: true,
              data: meanHumidity,
            }]
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
                  maxTicksLimit: 7,
                  font: {
                    size: 12,
                  },
                  color: '#fff',  // Make x-axis labels white
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
                  maxTicksLimit: 10,
                  padding: 10,
                  font: {
                    size: 12,
                  },
                  color: '#fff',  // Make y-axis labels white
                }
              },
            },
            plugins: {
              legend: {
                display: false,
                labels: {
                  usePointStyle: true,
                  padding: 20,
                  font: {
                    size: 14,
                  },
                  color: '#858796',
                },
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
                    yMin: 50,
                    yMax: 50,
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
              mode: 'index',
              intersect: false,
            },
          },
        });
    })    
    .catch(error => {
      console.error('Error fetching data:', error);  // Correctly catching and logging the error
    });
