class TransformerDemo {
    constructor() {
        this.vocabulary = ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'cat', 'runs', 'fast', 'slow', 'big', 'small', 'red', 'blue', 'green'];
        this.layers = 6;
        this.heads = 8;
        this.currentTokens = [];
        this.isProcessing = false;
    }

    tokenize(text) {
        return text.toLowerCase().split(/\s+/).filter(token => token.length > 0);
    }

    generateAttentionWeights(tokens) {
        const weights = [];
        for (let i = 0; i < tokens.length; i++) {
            weights[i] = [];
            for (let j = 0; j < tokens.length; j++) {
                weights[i][j] = Math.random() * 0.8 + 0.1;
            }
        }
        return weights;
    }

    visualizeAttention(tokens, weights) {
        const container = document.getElementById('attentionViz');
        container.innerHTML = '';
        
        const table = document.createElement('div');
        table.style.display = 'grid';
        table.style.gridTemplateColumns = `repeat(${tokens.length + 1}, 1fr)`;
        table.style.gap = '2px';
        table.style.fontSize = '12px';

        // Header row
        table.appendChild(this.createCell('', true));
        tokens.forEach(token => {
            table.appendChild(this.createCell(token, true));
        });

        // Data rows
        tokens.forEach((token, i) => {
            table.appendChild(this.createCell(token, true));
            weights[i].forEach((weight, j) => {
                const cell = this.createCell(weight.toFixed(2));
                const intensity = weight;
                cell.style.background = `rgba(255, 107, 107, ${intensity})`;
                table.appendChild(cell);
            });
        });

        container.appendChild(table);
    }

    createCell(content, isHeader = false) {
        const cell = document.createElement('div');
        cell.textContent = content;
        cell.style.padding = '8px';
        cell.style.textAlign = 'center';
        cell.style.background = isHeader ? 'rgba(255,255,255,0.1)' : 'rgba(255,255,255,0.05)';
        cell.style.borderRadius = '4px';
        return cell;
    }

    async animateTokenFlow(tokens) {
        const container = document.getElementById('tokenFlow');
        container.innerHTML = '';

        for (let layer = 0; layer < this.layers; layer++) {
            const layerDiv = document.createElement('div');
            layerDiv.className = 'layer';
            layerDiv.innerHTML = `<strong>Layer ${layer + 1}</strong>`;
            
            const tokenContainer = document.createElement('div');
            tokenContainer.style.marginTop = '10px';
            
            tokens.forEach((token, i) => {
                const tokenEl = document.createElement('span');
                tokenEl.className = 'token';
                tokenEl.textContent = token;
                tokenEl.style.animationDelay = `${i * 0.1}s`;
                tokenContainer.appendChild(tokenEl);
            });

            // Add attention heads visualization
            const headsContainer = document.createElement('div');
            headsContainer.style.marginTop = '10px';
            headsContainer.innerHTML = '<small>Attention Heads: </small>';
            
            for (let h = 0; h < this.heads; h++) {
                const head = document.createElement('div');
                head.className = 'attention-head';
                head.style.background = `hsl(${h * 45}, 70%, 60%)`;
                head.title = `Head ${h + 1}`;
                headsContainer.appendChild(head);
            }

            layerDiv.appendChild(tokenContainer);
            layerDiv.appendChild(headsContainer);
            container.appendChild(layerDiv);

            await this.sleep(300);
            this.animateTokens(tokenContainer);
        }
    }

    animateTokens(container) {
        const tokens = container.querySelectorAll('.token');
        tokens.forEach((token, i) => {
            setTimeout(() => {
                token.classList.add('active');
                setTimeout(() => token.classList.remove('active'), 500);
            }, i * 100);
        });
    }

    updateProgress(percentage) {
        document.getElementById('progressFill').style.width = `${percentage}%`;
    }

    async processText() {
        if (this.isProcessing) return;
        
        this.isProcessing = true;
        const input = document.getElementById('textInput').value;
        const tokens = this.tokenize(input);
        this.currentTokens = tokens;

        this.updateProgress(20);
        
        // Generate and visualize attention
        const attentionWeights = this.generateAttentionWeights(tokens);
        this.visualizeAttention(tokens, attentionWeights);
        
        this.updateProgress(50);
        
        // Animate token flow
        await this.animateTokenFlow(tokens);
        
        this.updateProgress(80);
        
        // Show processing result
        const output = document.getElementById('output');
        output.innerHTML = `
            <div><strong>Input tokens:</strong> [${tokens.join(', ')}]</div>
            <div><strong>Processed through:</strong> ${this.layers} transformer layers</div>
            <div><strong>Attention heads:</strong> ${this.heads} per layer</div>
            <div><strong>Total parameters:</strong> ~${(tokens.length * this.layers * this.heads * 64).toLocaleString()}</div>
        `;
        
        this.updateProgress(100);
        setTimeout(() => this.updateProgress(0), 1000);
        
        this.isProcessing = false;
    }

    generateText() {
        if (this.currentTokens.length === 0) {
            alert('Please process some text first!');
            return;
        }

        const nextToken = this.vocabulary[Math.floor(Math.random() * this.vocabulary.length)];
        const input = document.getElementById('textInput');
        input.value += ' ' + nextToken;
        
        // Auto-process the new text
        setTimeout(() => this.processText(), 100);
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize demo
const demo = new TransformerDemo();

// Global functions for HTML
function processText() {
    demo.processText();
}

function generateText() {
    demo.generateText();
}

// Auto-process initial text on load
window.addEventListener('load', () => {
    setTimeout(() => demo.processText(), 500);
});

// Add some interactive effects
document.addEventListener('mousemove', (e) => {
    const cursor = document.createElement('div');
    cursor.style.position = 'fixed';
    cursor.style.left = e.clientX + 'px';
    cursor.style.top = e.clientY + 'px';
    cursor.style.width = '4px';
    cursor.style.height = '4px';
    cursor.style.background = 'rgba(255, 107, 107, 0.6)';
    cursor.style.borderRadius = '50%';
    cursor.style.pointerEvents = 'none';
    cursor.style.zIndex = '9999';
    document.body.appendChild(cursor);
    
    setTimeout(() => cursor.remove(), 500);
});
