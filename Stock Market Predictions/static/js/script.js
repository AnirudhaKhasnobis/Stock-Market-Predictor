// Add any client-side JavaScript functionality here
document.addEventListener('DOMContentLoaded', function() {
    // This executes when the page loads
    console.log('Stock Market Prediction App Ready');
    
    // Example: Highlight current stock selection
    const stockSelect = document.getElementById('stock_symbol');
    if (stockSelect) {
        stockSelect.addEventListener('change', function() {
            console.log('Selected stock:', this.value);
        });
    }
});