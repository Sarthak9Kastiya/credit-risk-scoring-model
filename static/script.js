document.getElementById('loan-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const submitBtn = document.getElementById('submit-btn');
    const loader = document.getElementById('submit-loader');
    const btnText = submitBtn.querySelector('span');
    
    // UI Loading state
    btnText.style.display = 'none';
    loader.style.display = 'block';
    submitBtn.disabled = true;
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    
    // Convert numeric fields
    const numericFields = ['customer_age', 'customer_income', 'employment_duration', 'loan_amnt', 'term_years', 'cred_hist_length', 'credit_score'];
    numericFields.forEach(field => {
        data[field] = Number(data[field]);
    });

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.error || result.detail || "An error occurred");
        }
        
        showModal(result);
        
    } catch (error) {
        alert("Error: " + error.message);
    } finally {
        btnText.style.display = 'block';
        loader.style.display = 'none';
        submitBtn.disabled = false;
    }
});

function showModal(data) {
    const modal = document.getElementById('result-modal');
    const statusEl = document.getElementById('res-status');
    const rateEl = document.getElementById('res-rate');
    const gradeEl = document.getElementById('res-grade');
    const probEl = document.getElementById('res-prob');
    const msgEl = document.getElementById('res-message');
    
    // Reset classes
    statusEl.className = 'value';
    
    if (data.status === "DEFAULT") {
        statusEl.textContent = "Loan Denied (High Risk)";
        statusEl.classList.add('status-denied');
        msgEl.textContent = "Our model predicts a high probability of default.";
    } else {
        statusEl.textContent = "Loan Approved (Low Risk)";
        statusEl.classList.add('status-approved');
        msgEl.textContent = "Congratulations! The loan is likely to be paid successfully.";
    }
    
    rateEl.textContent = data.interest_rate ? `${data.interest_rate}%` : "--";
    gradeEl.textContent = data.loan_grade || "--";
    probEl.textContent = `${(data.probability_of_default * 100).toFixed(2)}%`;
    
    modal.classList.add('active');
}

document.getElementById('close-modal').addEventListener('click', () => {
    document.getElementById('result-modal').classList.remove('active');
});

// Close when clicking outside modal
document.getElementById('result-modal').addEventListener('click', (e) => {
    if (e.target.id === 'result-modal') {
        e.target.classList.remove('active');
    }
});
