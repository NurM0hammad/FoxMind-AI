// static/js/chat.js

class GeminiChat {
    constructor() {
        this.sessionId = null;
        this.isLoading = false;
        this.currentModel = document.getElementById('modelSelect')?.value || 'gemini-1.5-flash';
        this.currentPersonality = document.getElementById('personalitySelect')?.value || 'default';
        this.temperature = 0.7;
        this.messageHistory = [];
        this.streaming = false;
        
        this.init();
    }
    
    init() {
        this.initializeElements();
        this.attachEventListeners();
        this.loadConversations();
        this.loadHistory();
        this.updateModelBadge();
    }
    
    initializeElements() {
        this.chatForm = document.getElementById('chatForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messagesContainer = document.getElementById('messagesContainer');
        this.welcomeScreen = document.getElementById('welcomeScreen');
        this.conversationsList = document.getElementById('conversationsList');
        this.modelSelect = document.getElementById('modelSelect');
        this.personalitySelect = document.getElementById('personalitySelect');
        this.temperatureInput = document.getElementById('temperature');
        this.tempValue = document.getElementById('tempValue');
        this.resetBtn = document.getElementById('resetBtn');
        this.newChatBtn = document.getElementById('newChatBtn');
        this.menuToggle = document.getElementById('menuToggle');
        this.sidebar = document.getElementById('sidebar');
        this.currentModelBadge = document.querySelector('#currentModel span');
        this.currentPersonalityBadge = document.querySelector('#currentPersonality span');
    }
    
    attachEventListeners() {
        // Form submission
        this.chatForm.addEventListener('submit', (e) => this.handleSubmit(e));
        
        // Input handling
        this.messageInput.addEventListener('input', () => this.handleInput());
        this.messageInput.addEventListener('keydown', (e) => this.handleKeyDown(e));
        
        // Model selection
        this.modelSelect.addEventListener('change', (e) => {
            this.currentModel = e.target.value;
            this.updateModelBadge();
        });
        
        // Personality selection
        this.personalitySelect.addEventListener('change', (e) => {
            this.currentPersonality = e.target.value;
            this.currentPersonalityBadge.textContent = this.capitalizeFirst(this.currentPersonality);
        });
        
        // Temperature control
        this.temperatureInput.addEventListener('input', (e) => {
            this.temperature = e.target.value;
            this.tempValue.textContent = this.temperature;
        });
        
        // Reset button
        this.resetBtn.addEventListener('click', () => this.resetConversation());
        
        // New chat button
        this.newChatBtn.addEventListener('click', () => this.newConversation());
        
        // Menu toggle for mobile
        this.menuToggle.addEventListener('click', () => {
            this.sidebar.classList.toggle('active');
        });
        
        // Click outside to close sidebar on mobile
        document.addEventListener('click', (e) => {
            if (window.innerWidth <= 992) {
                if (!this.sidebar.contains(e.target) && !this.menuToggle.contains(e.target)) {
                    this.sidebar.classList.remove('active');
                }
            }
        });
        
        // Suggestion buttons
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const prompt = btn.dataset.prompt;
                this.messageInput.value = prompt;
                this.handleInput();
                this.chatForm.dispatchEvent(new Event('submit'));
            });
        });
    }
    
    async handleSubmit(e) {
        e.preventDefault();
        
        const message = this.messageInput.value.trim();
        if (!message || this.isLoading) return;
        
        // Hide welcome screen
        if (this.welcomeScreen) {
            this.welcomeScreen.style.display = 'none';
        }
        
        // Add user message to UI
        this.addMessageToUI('user', message);
        
        // Clear input
        this.messageInput.value = '';
        this.handleInput();
        
        // Show typing indicator
        this.showTypingIndicator();
        
        // Send to API
        await this.sendMessage(message);
    }
    
    async sendMessage(message) {
        this.isLoading = true;
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    model: this.currentModel,
                    personality: this.currentPersonality,
                    temperature: parseFloat(this.temperature),
                    stream: false
                })
            });
            
            const data = await response.json();
            
            // Remove typing indicator
            this.removeTypingIndicator();
            
            if (data.error) {
                this.showError(data.error);
                return;
            }
            
            // Add assistant message to UI
            this.addMessageToUI('assistant', data.response, data.usage);
            
            // Refresh conversations list
            this.loadConversations();
            
        } catch (error) {
            console.error('Error:', error);
            this.removeTypingIndicator();
            this.showError('Failed to send message. Please try again.');
        } finally {
            this.isLoading = false;
        }
    }
    
    addMessageToUI(role, content, usage = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        
        // Format content with markdown (simplified version)
        const formattedContent = this.formatMessage(content);
        
        messageDiv.innerHTML = `
            <div class="message-avatar">
                <i class="fas ${role === 'user' ? 'fa-user' : 'fa-robot'}"></i>
            </div>
            <div class="message-content-wrapper">
                <div class="message-content">
                    ${formattedContent}
                </div>
                <div class="message-meta">
                    <span class="message-timestamp">
                        <i class="far fa-clock"></i> ${timestamp}
                    </span>
                    ${usage ? `
                        <span class="message-tokens">
                            <i class="fas fa-coins"></i> ${usage.total_tokens || 0} tokens
                        </span>
                    ` : ''}
                </div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        
        // Scroll to bottom
        this.scrollToBottom();
    }
    
    formatMessage(content) {
        // Simple markdown-like formatting
        // Bold
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // Italic
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
        
        // Code blocks
        content = content.replace(/```(.*?)```/gs, '<pre><code>$1</code></pre>');
        
        // Inline code
        content = content.replace(/`(.*?)`/g, '<code>$1</code>');
        
        // Line breaks
        content = content.replace(/\n/g, '<br>');
        
        return content;
    }
    
    showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'typing-indicator';
        indicator.id = 'typingIndicator';
        indicator.innerHTML = `
            <span></span>
            <span></span>
            <span></span>
        `;
        this.messagesContainer.appendChild(indicator);
        this.scrollToBottom();
    }
    
    removeTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
    
    handleInput() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText || this.isLoading;
        
        // Auto-resize textarea
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = (this.messageInput.scrollHeight) + 'px';
    }
    
    handleKeyDown(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!this.sendButton.disabled) {
                this.chatForm.dispatchEvent(new Event('submit'));
            }
        }
    }
    
    async loadConversations() {
        try {
            const response = await fetch('/api/conversations');
            const data = await response.json();
            
            if (data.conversations) {
                this.renderConversations(data.conversations);
                document.getElementById('conversationCount').textContent = data.conversations.length;
            }
        } catch (error) {
            console.error('Error loading conversations:', error);
        }
    }
    
    renderConversations(conversations) {
        if (!this.conversationsList) return;
        
        if (conversations.length === 0) {
            this.conversationsList.innerHTML = `
                <div class="loading-conversations">
                    <i class="far fa-comment-dots" style="font-size: 2rem; color: var(--gray);"></i>
                    <span>No conversations yet</span>
                </div>
            `;
            return;
        }
        
        this.conversationsList.innerHTML = conversations.map(conv => `
            <div class="conversation-item ${conv.id === this.sessionId ? 'active' : ''}" data-id="${conv.id}">
                <div class="conversation-preview">${conv.preview || 'Empty conversation'}</div>
                <div class="conversation-meta">
                    <span class="conversation-date">
                        <i class="far fa-calendar-alt"></i>
                        ${this.formatDate(conv.updated_at)}
                    </span>
                    <span class="conversation-model">${conv.model || 'Unknown'}</span>
                </div>
                <button class="delete-conv" onclick="event.stopPropagation(); chat.deleteConversation('${conv.id}')">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        `).join('');
        
        // Add click handlers
        document.querySelectorAll('.conversation-item').forEach(item => {
            item.addEventListener('click', () => {
                const convId = item.dataset.id;
                this.loadConversation(convId);
            });
        });
    }
    
    formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = now - date;
        
        // Less than 24 hours
        if (diff < 86400000) {
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        
        // Less than 7 days
        if (diff < 604800000) {
            return date.toLocaleDateString([], { weekday: 'short' });
        }
        
        return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
    
    async loadHistory() {
        try {
            const response = await fetch('/api/history');
            const data = await response.json();
            
            if (data.history && data.history.length > 0) {
                // Hide welcome screen
                if (this.welcomeScreen) {
                    this.welcomeScreen.style.display = 'none';
                }
                
                // Clear messages container
                this.messagesContainer.innerHTML = '';
                
                // Add messages to UI
                data.history.forEach(msg => {
                    this.addMessageToUI(msg.role, msg.content);
                });
            }
            
            // Update session ID
            this.sessionId = data.session_id;
            
        } catch (error) {
            console.error('Error loading history:', error);
        }
    }
    
    async loadConversation(conversationId) {
        try {
            const response = await fetch(`/api/load/${conversationId}`, {
                method: 'POST'
            });
            
            if (response.ok) {
                this.sessionId = conversationId;
                this.messagesContainer.innerHTML = '';
                this.welcomeScreen.style.display = 'none';
                
                // Update active state in sidebar
                document.querySelectorAll('.conversation-item').forEach(item => {
                    item.classList.toggle('active', item.dataset.id === conversationId);
                });
                
                // Load conversation history
                await this.loadHistory();
                
                // Close sidebar on mobile
                if (window.innerWidth <= 992) {
                    this.sidebar.classList.remove('active');
                }
            }
        } catch (error) {
            console.error('Error loading conversation:', error);
            this.showError('Failed to load conversation');
        }
    }
    
    async deleteConversation(conversationId) {
        if (!confirm('Are you sure you want to delete this conversation?')) return;
        
        try {
            const response = await fetch(`/api/delete/${conversationId}`, {
                method: 'DELETE'
            });
            
            if (response.ok) {
                this.loadConversations();
                this.showToast('Conversation deleted', 'success');
            }
        } catch (error) {
            console.error('Error deleting conversation:', error);
            this.showError('Failed to delete conversation');
        }
    }
    
    async resetConversation() {
        try {
            const response = await fetch('/api/reset', {
                method: 'POST'
            });
            
            if (response.ok) {
                // Clear messages
                this.messagesContainer.innerHTML = '';
                
                // Show welcome screen
                if (this.welcomeScreen) {
                    this.welcomeScreen.style.display = 'block';
                }
                
                // Show success message
                this.showToast('Conversation reset successfully', 'success');
            }
        } catch (error) {
            console.error('Error resetting conversation:', error);
            this.showError('Failed to reset conversation');
        }
    }
    
    newConversation() {
        // Reset UI
        this.messagesContainer.innerHTML = '';
        this.welcomeScreen.style.display = 'block';
        
        // Reset session
        this.sessionId = null;
        
        // Reset conversation on server
        this.resetConversation();
        
        // Close sidebar on mobile
        if (window.innerWidth <= 992) {
            this.sidebar.classList.remove('active');
        }
    }
    
    updateModelBadge() {
        if (this.currentModelBadge) {
            this.currentModelBadge.textContent = this.currentModel;
        }
    }
    
    capitalizeFirst(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    showError(message) {
        this.showToast(message, 'error');
    }
    
    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        toast.innerHTML = `
            <i class="fas ${icons[type] || icons.info}"></i>
            <div class="toast-content">
                <div class="toast-title">${type.charAt(0).toUpperCase() + type.slice(1)}</div>
                <div class="toast-message">${message}</div>
            </div>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chat = new GeminiChat();
});