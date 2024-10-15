// Get the canvas element where the chart will be rendered
const ctx1 = document.getElementById('myChart1').getContext('2d');

// Create a new doughnut chart instance
const myDoughnutChart = new Chart(ctx1, {
    type: 'doughnut', // Define the type of chart as 'doughnut'
    data: {
        labels: ['Sales', 'Marketing'], // Labels for each segment
        datasets: [{
            label: 'Department Budget',
            data: [300, 150], // Data representing the size of each segment
            backgroundColor: [
                'rgba(255, 99, 132, 0.2)', // Colors for each segment
                'rgba(54, 162, 235, 0.2)',
            ],
            borderColor: [
                'rgba(255, 99, 132, 1)',
                'rgba(54, 162, 235, 1)',
            ],
            borderWidth: 1 // Border thickness around each segment
        }]
    },
    options: {
        responsive: true, // Make the chart responsive
        cutout: 120, // Cutout for the doughnut chart
        layout: {
            padding: {
                left: 25,
                right: 25,
                top: 25,
                bottom: 25
            }
        },
        plugins: {
            legend: {
                display: false // Disable the legend display
            },
            tooltip: {
                backgroundColor: "rgb(255,255,255)",
                bodyFontColor: "#858796",
                borderColor: '#dddfeb',
                borderWidth: 1,
                padding: 15,
                displayColors: false
            }
        }
    },
    plugins: [{
        beforeDraw: (chart) => {
            const { width, height, ctx } = chart;
            ctx.restore();
            const fontSize = (height / 160).toFixed(2);
            ctx.font = `${fontSize}em sans-serif`;
            ctx.textBaseline = 'middle';

            const text = chart.data.datasets[0].data.reduce((a, b) => a + b, 0); // Sum of the dataset
            const textX = Math.round((width - ctx.measureText(text).width) / 2);
            const textY = height / 2;

            ctx.fillText(text, textX, textY);
            ctx.save();
        }
    }]
});
