// static/js/script.js
document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('language-toggle');
    toggle.checked = false;
    updateLanguage(false);
});

document.getElementById('language-toggle').addEventListener('change', (e) => {
    updateLanguage(e.target.checked);
});

function updateLanguage(isGujarati) {
    document.querySelectorAll('[data-gujarati]').forEach(el => {
        const defaultText = el.id === 'title' ? 'Career Recommendation' :
                           el.id === 'skills-label' ? 'Skills' :
                           el.id === 'interests-label' ? 'Interests' :
                           el.id === 'experience-label' ? 'Experience Level' :
                           el.id === 'search-btn' ? 'Search' :
                           el.id === 'result-text' ? 'Recommendation functionality coming soon.' : el.textContent;
        el.textContent = isGujarati ? el.getAttribute('data-gujarati') : defaultText;
    });

    document.querySelectorAll('[data-gujarati-placeholder]').forEach(el => {
        const defaultPlaceholder = el.id === 'skills' ? 'e.g., Python, Java' : 'e.g., AI Research';
        el.placeholder = isGujarati ? el.getAttribute('data-gujarati-placeholder') : defaultPlaceholder;
    });

    document.querySelectorAll('option[data-gujarati]').forEach(el => {
        const defaultText = el.value.charAt(0).toUpperCase() + el.value.slice(1);
        el.textContent = isGujarati ? el.getAttribute('data-gujarati') : defaultText;
    });

    document.querySelector('title').textContent = isGujarati ? 
        document.querySelector('title').getAttribute('data-gujarati') : 
        'Career Recommendation';
}

document.getElementById('recommendation-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const skills = document.getElementById('skills').value;
    const interests = document.getElementById('interests').value;
    const experience = document.getElementById('experience').value;
    const isGujarati = document.getElementById('language-toggle').checked;
    
    // Show loading state
    const searchBtn = document.getElementById('search-btn');
    searchBtn.disabled = true;
    searchBtn.textContent = isGujarati ? 'પ્રક્રિયા કરી રહ્યું છે...' : 'Processing...';
    
    try {
        const response = await fetch('/recommend', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 
                skills, 
                interests, 
                experience,
                language: isGujarati ? 'gujarati' : 'english'
            }),
        });

        const data = await response.json();
        const resultDiv = document.getElementById('result');
        const resultText = document.getElementById('result-text');

        if (data && data.predictions) {
            let output = isGujarati ? 
                `તમારી કારકિર્દીની ભલામણ: <strong>${data.consensus}</strong><br><br>` + 
                "મોડેલ પરિણામો:<br>" : 
                `Your recommended career: <strong>${data.consensus}</strong><br><br>` + 
                "Model results:<br>";
            
            output += data.predictions.join("<br>");
            resultText.innerHTML = output;
        } else {
            resultText.textContent = isGujarati ?
                "ક્ષમા કરો, એક ભૂલ આવી છે. કૃપયા ફરી પ્રયાસ કરો." :
                "Sorry, an error occurred. Please try again.";
        }

        resultDiv.classList.remove('opacity-0');
    } catch (error) {
        console.error('Error:', error);
        const resultText = document.getElementById('result-text');
        resultText.textContent = isGujarati ?
            "ક્ષમા કરો, એક ભૂલ આવી છે. કૃપયા ફરી પ્રયાસ કરો." :
            "Sorry, an error occurred. Please try again.";
    } finally {
        // Reset button
        searchBtn.disabled = false;
        searchBtn.textContent = isGujarati ? 'શોધો' : 'Search';
    }
});