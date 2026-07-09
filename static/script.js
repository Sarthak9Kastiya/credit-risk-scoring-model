document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    const resultPanel = document.getElementById('result-panel');
    const resetBtn = document.getElementById('reset-btn');
    
    const statusIcon = document.getElementById('status-icon');
    const statusText = document.getElementById('status-text');
    const probValue = document.getElementById('prob-value');
    const progressBarFill = document.getElementById('progress-bar-fill');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;

        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        // Convert number types
        data.customer_age = parseInt(data.customer_age);
        data.customer_income = parseFloat(data.customer_income);
        data.employment_duration = parseFloat(data.employment_duration);
        data.loan_amnt = parseFloat(data.loan_amnt);
        data.credit_score = parseInt(data.credit_score);
        data.term_years = parseInt(data.term_years);
        data.cred_hist_length = parseInt(data.cred_hist_length);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Prediction failed on the server');
            }

            const result = await response.json();
            
            showResult(result);
        } catch (error) {
            alert("Error connecting to the prediction server. Make sure the API is running.");
            console.error(error);
        } finally {
            // Restore button state
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        resultPanel.classList.add('fade-out');
        setTimeout(() => {
            resultPanel.classList.add('hidden');
            resultPanel.classList.remove('fade-out');
            form.style.opacity = '1';
            form.style.pointerEvents = 'auto';
        }, 400);
    });

    function showResult(result) {
        // Hide form, show result
        form.style.opacity = '0';
        form.style.pointerEvents = 'none';
        
        resultPanel.classList.remove('hidden');
        
        // Reset classes
        statusIcon.className = 'icon-circle';
        statusText.className = '';
        
        // Process data
        const prob = (result.probability_of_default * 100).toFixed(1);
        probValue.textContent = `${prob}%`;
        
        // Wait a small tick for display block to render before animating width
        setTimeout(() => {
            progressBarFill.style.width = `${prob}%`;
            
            if (result.prediction === 1) {
                statusIcon.classList.add('danger');
                statusText.textContent = 'High Risk of Default';
                statusText.classList.add('text-danger');
                progressBarFill.style.backgroundColor = 'var(--danger-color)';
            } else {
                statusIcon.classList.add('success');
                statusText.textContent = 'Low Risk (Approved)';
                statusText.classList.add('text-success');
                progressBarFill.style.backgroundColor = 'var(--success-color)';
            }
        }, 50);
    }
});
