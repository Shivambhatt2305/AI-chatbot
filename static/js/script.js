document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const clearHistoryBtn = document.getElementById('clearHistoryBtn');
    const themeToggle = document.getElementById('themeToggle');
    const themeIcon = themeToggle.querySelector('i');
    
    // Templates
    const userMessageTemplate = document.getElementById('userMessageTemplate');
    const aiMessageTemplate = document.getElementById('aiMessageTemplate');
    const solutionTemplate = document.getElementById('solutionTemplate');
    
    // Markdown converter
    const converter = new showdown.Converter({
        tables: true,
        simplifiedAutoLink: true,
        strikethrough: true,
        tasklists: true,
        emoji: true
    });
    
    // Auto-resize textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.scrollHeight > 120) {
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
        }
    });
    
    // Theme toggle functionality
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const savedTheme = localStorage.getItem('theme');
    
    if (savedTheme === 'dark' || (!savedTheme && prefersDarkMode)) {
        document.body.classList.add('dark-mode');
        themeIcon.classList.replace('fa-moon', 'fa-sun');
    }
    
    themeToggle.addEventListener('click', () => {
        document.body.classList.toggle('dark-mode');
        
        if (document.body.classList.contains('dark-mode')) {
            themeIcon.classList.replace('fa-moon', 'fa-sun');
            localStorage.setItem('theme', 'dark');
        } else {
            themeIcon.classList.replace('fa-sun', 'fa-moon');
            localStorage.setItem('theme', 'light');
        }
    });
    
    // Send message on Enter key (but allow Shift+Enter for new lines)
    messageInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Send message on button click
    sendButton.addEventListener('click', sendMessage);
    
    // Clear history
    clearHistoryBtn.addEventListener('click', clearHistory);
    
    // Handle suggestion chips
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('suggestion-chip')) {
            const text = e.target.dataset.text;
            messageInput.value = text;
            messageInput.dispatchEvent(new Event('input'));
            sendMessage();
        }
    });
    
    // Copy solution to clipboard
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('copy-button') || e.target.parentElement.classList.contains('copy-button')) {
            const button = e.target.classList.contains('copy-button') ? e.target : e.target.parentElement;
            const solutionContent = button.closest('.solution-container').querySelector('.solution-content');
            
            if (solutionContent) {
                navigator.clipboard.writeText(solutionContent.innerText).then(() => {
                    const originalText = button.innerHTML;
                    button.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    setTimeout(() => {
                        button.innerHTML = originalText;
                    }, 2000);
                });
            }
        }
    });
    
    // Send message function
    function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;
        
        // Clear input
        messageInput.value = '';
        messageInput.style.height = 'auto';
        
        // Remove welcome message if present
        const welcomeMessage = document.querySelector('.welcome-message');
        if (welcomeMessage) {
            welcomeMessage.remove();
        }
        
        // Add user message
        const userMessageClone = document.importNode(userMessageTemplate.content, true);
        const userBubble = userMessageClone.querySelector('.message-bubble');
        const userTimestamp = userMessageClone.querySelector('.timestamp');
        
        userBubble.textContent = message;
        userTimestamp.textContent = getCurrentTime();
        chatMessages.appendChild(userMessageClone);
        
        // Add AI message with typing indicator
        const aiMessageClone = document.importNode(aiMessageTemplate.content, true);
        const aiMessage = aiMessageClone.querySelector('.message');
        const aiTimestamp = aiMessageClone.querySelector('.timestamp');
        
        aiTimestamp.textContent = getCurrentTime();
        chatMessages.appendChild(aiMessageClone);
        
        // Scroll to bottom
        scrollToBottom();
        
        // Send request to server
        fetch('/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            // Remove typing indicator
            const typingIndicator = aiMessage.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            const aiBubble = aiMessage.querySelector('.message-bubble');
            
            if (data.error) {
                aiBubble.classList.add('error');
                aiBubble.textContent = data.error;
            } else {
                // Add AI response
                aiBubble.textContent = "I've prepared a detailed solution for you. Please check below.";
                
                // Add solution container
                const solutionClone = document.importNode(solutionTemplate.content, true);
                const solutionContent = solutionClone.querySelector('.solution-content');
                
                // Convert markdown to HTML
                const htmlContent = converter.makeHtml(data.response);
                solutionContent.innerHTML = htmlContent;
                
                // Add solution after AI message
                aiMessage.insertAdjacentElement('afterend', solutionClone.querySelector('.solution-container'));
                
                // Highlight code blocks
                document.querySelectorAll('pre code').forEach((block) => {
                    hljs.highlightElement(block);
                });
            }
            
            // Scroll to bottom
            scrollToBottom();
        })
        .catch(error => {
            console.error('Error:', error);
            const typingIndicator = aiMessage.querySelector('.typing-indicator');
            if (typingIndicator) {
                typingIndicator.remove();
            }
            
            const aiBubble = aiMessage.querySelector('.message-bubble');
            aiBubble.classList.add('error');
            aiBubble.textContent = "Sorry, there was an error processing your request. Please try again.";
            
            scrollToBottom();
        });
    }
    
    // Clear chat history
    function clearHistory() {
        fetch('/clear_history', {
            method: 'POST',
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                chatMessages.innerHTML = '';
                
                // Add welcome message back
                const welcomeHTML = `
                    <div class="welcome-message">
                        <div class="welcome-icon">
                            <i class="fas fa-comments"></i>
                        </div>
                        <h2>Welcome to Gemini Chat</h2>
                        <p>Ask me anything and I'll provide detailed answers!</p>
                        <div class="suggestion-chips">
                            <button class="suggestion-chip" data-text="What can you do?">What can you do?</button>
                            <button class="suggestion-chip" data-text="Tell me about AI">Tell me about AI</button>
                            <button class="suggestion-chip" data-text="Write a short poem">Write a short poem</button>
                        </div>
                    </div>
                `;
                chatMessages.innerHTML = welcomeHTML;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
    
    // Helper function to get current time
    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // Helper function to scroll to bottom of chat
    function scrollToBottom() {
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 100);
    }
    
    // Initialize - focus input field
    messageInput.focus();
});