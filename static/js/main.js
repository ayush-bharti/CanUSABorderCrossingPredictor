/**
 * Main JavaScript file for Border Crossing Predictor
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Set current date as default in date selector
    const dateSelector = document.getElementById('date');
    if (dateSelector) {
        const today = new Date();
        const formattedDate = today.toISOString().split('T')[0];
        
        // Find and select the option with today's date
        for (let i = 0; i < dateSelector.options.length; i++) {
            if (dateSelector.options[i].value === formattedDate) {
                dateSelector.selectedIndex = i;
                break;
            }
        }
    }
    
    // Set current hour as default in time selector
    const timeSelector = document.getElementById('time');
    if (timeSelector) {
        const now = new Date();
        const currentHour = now.getHours();
        const formattedHour = `${currentHour.toString().padStart(2, '0')}:00`;
        
        // Find and select the option with current hour
        for (let i = 0; i < timeSelector.options.length; i++) {
            if (timeSelector.options[i].value === formattedHour) {
                timeSelector.selectedIndex = i;
                break;
            }
        }
    }
    
    // Responsive chart resizing
    window.addEventListener('resize', function() {
        const charts = document.querySelectorAll('[id$="-chart"]');
        charts.forEach(chart => {
            if (chart && chart._Plotly) {
                Plotly.relayout(chart, {
                    'width': chart.offsetWidth
                });
            }
        });
    });
});
