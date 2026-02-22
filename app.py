import os
import json
import uuid
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai import types
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', os.urandom(24).hex())
app.config['SESSION_TYPE'] = 'filesystem'
CORS(app)

# ============================================
# GEMINI API CONFIGURATION
# ============================================

def get_api_key():
    """Get Gemini API key from environment"""
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
    return api_key

# Configure Gemini
GEMINI_API_KEY = get_api_key()
AVAILABLE_MODELS = []

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("✅ Gemini API configured successfully")
    
    # List available models dynamically
    try:
        models = genai.list_models()
        AVAILABLE_MODELS = []
        for m in models:
            if 'generateContent' in m.supported_generation_methods:
                # Extract just the model name (remove everything before 'models/')
                model_name = m.name.replace('models/', '')
                AVAILABLE_MODELS.append({
                    'name': model_name,
                    'display_name': m.display_name if hasattr(m, 'display_name') else model_name,
                    'description': m.description if hasattr(m, 'description') else '',
                    'supported_methods': m.supported_generation_methods
                })
                logger.info(f"Found model: {model_name} - {m.description[:50] if hasattr(m, 'description') else ''}")
        
        logger.info(f"✅ Available models: {[m['name'] for m in AVAILABLE_MODELS]}")
    except Exception as e:
        logger.error(f"Error listing models: {e}")
else:
    logger.warning("⚠️ No Gemini API key found")

# ============================================
# CONFIGURATION
# ============================================

CONVERSATIONS_DIR = 'conversations'
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)

# Build MODELS dictionary from AVAILABLE_MODELS
MODELS = {}
for model_info in AVAILABLE_MODELS:
    model_name = model_info['name']
    
    # Determine model capabilities based on name
    is_pro = 'pro' in model_name.lower()
    is_flash = 'flash' in model_name.lower()
    is_vision = 'vision' in model_name.lower()
    
    MODELS[model_name] = {
        'name': model_name,
        'description': model_info.get('description', f'Gemini model: {model_name}'),
        'context_window': 1000000 if '1.5' in model_name else 30000,
        'supports_streaming': True,
        'supports_vision': is_vision or '1.5' in model_name
    }

# If no models found, provide fallback
if not MODELS:
    MODELS = {
        'gemini-1.5-pro': {
            'name': 'gemini-1.5-pro',
            'description': 'Best for complex reasoning, coding, and analysis',
            'context_window': 1000000,
            'supports_streaming': True,
            'supports_vision': True
        },
        'gemini-1.5-flash': {
            'name': 'gemini-1.5-flash',
            'description': 'Fast, efficient, good for everyday conversations',
            'context_window': 1000000,
            'supports_streaming': True,
            'supports_vision': True
        },
        'gemini-pro': {
            'name': 'gemini-pro',
            'description': 'Legacy pro model',
            'context_window': 30000,
            'supports_streaming': True,
            'supports_vision': False
        }
    }

# System prompts for different personalities
SYSTEM_PROMPTS = {
    'default': "You are a helpful, friendly AI assistant. Be conversational, informative, and engaging.",
    
    'coding': """You are an expert programming assistant. Help users write clean, efficient code.
                 Provide examples, explain concepts, and follow best practices. Use markdown for code blocks.
                 Be detailed in your explanations and suggest best practices.""",
    
    'creative': "You are a creative assistant. Help with writing, brainstorming, and creative projects. Be imaginative, inspiring, and think outside the box.",
    
    'academic': "You are an academic tutor. Provide thorough explanations, cite sources when possible, and help with learning complex topics. Be patient and educational.",
    
    'concise': "You are a concise assistant. Give brief, direct answers. Avoid unnecessary details unless specifically asked for more information.",
    
    'gemini': "You are Google's Gemini AI assistant. Be helpful, harmless, and honest. Provide accurate, up-to-date information."
}

# ============================================
# GEMINI CHATBOT CLASS
# ============================================

class GeminiChatbot:
    def __init__(self):
        self.conversations = {}
        self.load_conversations()
    
    def load_conversations(self):
        """Load existing conversations from disk"""
        if os.path.exists(CONVERSATIONS_DIR):
            for filename in os.listdir(CONVERSATIONS_DIR):
                if filename.endswith('.json'):
                    try:
                        with open(os.path.join(CONVERSATIONS_DIR, filename), 'r') as f:
                            conv_id = filename.replace('.json', '')
                            self.conversations[conv_id] = json.load(f)
                            logger.info(f"Loaded conversation: {conv_id}")
                    except Exception as e:
                        logger.error(f"Error loading conversation {filename}: {e}")
    
    def save_conversation(self, session_id):
        """Save conversation to disk"""
        if session_id in self.conversations:
            filepath = os.path.join(CONVERSATIONS_DIR, f"{session_id}.json")
            try:
                with open(filepath, 'w') as f:
                    json.dump(self.conversations[session_id], f, indent=2)
                logger.info(f"Saved conversation: {session_id}")
            except Exception as e:
                logger.error(f"Error saving conversation {session_id}: {e}")
    
    def get_or_create_conversation(self, session_id, model=None, personality='default'):
        """Get existing conversation or create new one"""
        if model is None:
            # Use first available model or fallback
            model = next(iter(MODELS.keys())) if MODELS else 'gemini-pro'
        
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'id': session_id,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'model': model,
                'personality': personality,
                'messages': [],
                'gemini_chat': None,
                'gemini_model': None
            }
            
            # Initialize Gemini chat session if API key exists
            if GEMINI_API_KEY:
                self._initialize_gemini_session(session_id, model, personality)
        
        return self.conversations[session_id]
    
    def _initialize_gemini_session(self, session_id, model, personality):
        """Initialize a new Gemini chat session"""
        try:
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 8192,
            }
            
            # Get system instruction based on personality
            system_instruction = SYSTEM_PROMPTS.get(personality, SYSTEM_PROMPTS['default'])
            
            # Safety settings
            safety_settings = {
                types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
            
            # Create model with configuration
            self.conversations[session_id]['gemini_model'] = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                system_instruction=system_instruction,
                safety_settings=safety_settings
            )
            
            # Initialize chat with existing history if any
            history = []
            for msg in self.conversations[session_id].get('messages', []):
                if msg['role'] in ['user', 'assistant']:
                    history.append({
                        'role': 'user' if msg['role'] == 'user' else 'model',
                        'parts': [msg['content']]
                    })
            
            self.conversations[session_id]['gemini_chat'] = self.conversations[session_id]['gemini_model'].start_chat(
                history=history
            )
            
            logger.info(f"Initialized Gemini session for {session_id} with model {model}")
            
        except Exception as e:
            logger.error(f"Error initializing Gemini: {e}")
            self.conversations[session_id]['gemini_chat'] = None
            self.conversations[session_id]['gemini_model'] = None
    
    def get_response(self, user_message, session_id, model=None, 
                    personality='default', temperature=0.7, stream=False):
        """Get response from Gemini API"""
        try:
            if model is None:
                model = next(iter(MODELS.keys())) if MODELS else 'gemini-pro'
            
            # Get or create conversation
            conversation = self.get_or_create_conversation(session_id, model, personality)
            
            # Add user message to history
            conversation['messages'].append({
                'role': 'user',
                'content': user_message,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check if Gemini is available
            if not GEMINI_API_KEY:
                error_msg = "Gemini API key not configured. Please add GEMINI_API_KEY to .env file."
                logger.error(error_msg)
                return {'error': error_msg}
            
            if not conversation.get('gemini_chat'):
                # Reinitialize if chat session is missing
                self._initialize_gemini_session(session_id, model, personality)
                
                if not conversation.get('gemini_chat'):
                    error_msg = "Failed to initialize Gemini chat session"
                    logger.error(error_msg)
                    return {'error': error_msg}
            
            # Update generation config if temperature changed
            if temperature != 0.7 and conversation.get('gemini_model'):
                conversation['gemini_model'].generation_config.temperature = temperature
            
            # Get chat session
            chat = conversation['gemini_chat']
            
            if stream:
                # Return streaming generator
                return self._stream_response(chat, user_message, conversation, session_id)
            else:
                # Get regular response
                response = chat.send_message(user_message)
                response_text = response.text
                
                # Add assistant message
                conversation['messages'].append({
                    'role': 'assistant',
                    'content': response_text,
                    'timestamp': datetime.now().isoformat()
                })
                
                conversation['updated_at'] = datetime.now().isoformat()
                conversation['model'] = model
                conversation['personality'] = personality
                self.save_conversation(session_id)
                
                # Get token usage if available
                usage = {}
                if hasattr(response, 'usage_metadata'):
                    usage = {
                        'prompt_tokens': response.usage_metadata.prompt_token_count,
                        'completion_tokens': response.usage_metadata.candidates_token_count,
                        'total_tokens': response.usage_metadata.total_token_count
                    }
                
                return {
                    'response': response_text,
                    'usage': usage,
                    'model': model,
                    'status': 'success'
                }
                
        except Exception as e:
            logger.error(f"Error in get_response: {e}")
            return {'error': str(e)}
    
    def _stream_response(self, chat, user_message, conversation, session_id):
        """Stream response from Gemini"""
        try:
            response = chat.send_message(user_message, stream=True)
            
            full_response = ""
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    yield f"data: {json.dumps({'chunk': chunk.text})}\n\n"
            
            # Add complete message to history
            conversation['messages'].append({
                'role': 'assistant',
                'content': full_response,
                'timestamp': datetime.now().isoformat()
            })
            
            conversation['updated_at'] = datetime.now().isoformat()
            self.save_conversation(session_id)
            
            yield f"data: {json.dumps({'done': True, 'full_response': full_response})}\n\n"
            
        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def reset_conversation(self, session_id):
        """Reset a conversation"""
        if session_id in self.conversations:
            # Keep metadata but reset messages and chat
            self.conversations[session_id]['messages'] = []
            self.conversations[session_id]['gemini_chat'] = None
            self.conversations[session_id]['updated_at'] = datetime.now().isoformat()
            
            # Reinitialize chat
            model = self.conversations[session_id].get('model', next(iter(MODELS.keys())))
            personality = self.conversations[session_id].get('personality', 'default')
            self._initialize_gemini_session(session_id, model, personality)
            
            self.save_conversation(session_id)
            return True
        return False

# Initialize chatbot
chatbot = GeminiChatbot()

# ============================================
# FLASK ROUTES
# ============================================

@app.route('/')
def home():
    """Render main chat interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        logger.info(f"New session created: {session['session_id']}")
    
    return render_template('index.html', 
                         models=MODELS,
                         api_configured=bool(GEMINI_API_KEY),
                         personalities=list(SYSTEM_PROMPTS.keys()))

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = session.get('session_id')
        
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        model = data.get('model')
        if not model and MODELS:
            model = next(iter(MODELS.keys()))
        
        personality = data.get('personality', 'default')
        temperature = float(data.get('temperature', 0.7))
        use_stream = data.get('stream', False)
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Check if API is configured
        if not GEMINI_API_KEY:
            return jsonify({'error': 'Gemini API key not configured. Please add GEMINI_API_KEY to .env file.'}), 503
        
        result = chatbot.get_response(
            user_message, session_id, model, personality, temperature, stream=use_stream
        )
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        session_id = session.get('session_id')
        
        if not session_id:
            session_id = str(uuid.uuid4())
            session['session_id'] = session_id
        
        model = data.get('model')
        if not model and MODELS:
            model = next(iter(MODELS.keys()))
        
        personality = data.get('personality', 'default')
        temperature = float(data.get('temperature', 0.7))
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Check if API is configured
        if not GEMINI_API_KEY:
            return jsonify({'error': 'Gemini API key not configured. Please add GEMINI_API_KEY to .env file.'}), 503
        
        result = chatbot.get_response(
            user_message, session_id, model, personality, temperature, stream=True
        )
        
        if isinstance(result, dict) and 'error' in result:
            return jsonify({'error': result['error']}), 500
        
        return Response(
            stream_with_context(result),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in stream endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_conversation():
    """Reset current conversation"""
    try:
        session_id = session.get('session_id')
        if session_id:
            if chatbot.reset_conversation(session_id):
                return jsonify({'status': 'success'})
            else:
                return jsonify({'error': 'Conversation not found'}), 404
        
        return jsonify({'error': 'No active session'}), 400
        
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get conversation history"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in chatbot.conversations:
            messages = chatbot.conversations[session_id]['messages']
            model = chatbot.conversations[session_id].get('model', next(iter(MODELS.keys())) if MODELS else 'unknown')
            personality = chatbot.conversations[session_id].get('personality', 'default')
            
            return jsonify({
                'history': messages,
                'model': model,
                'personality': personality
            })
        
        return jsonify({'history': [], 'model': next(iter(MODELS.keys())) if MODELS else 'unknown', 'personality': 'default'})
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations', methods=['GET'])
def list_conversations():
    """List all conversations"""
    try:
        conversations = []
        for conv_id, conv_data in chatbot.conversations.items():
            # Count non-system messages
            messages = [m for m in conv_data.get('messages', []) if m['role'] != 'system']
            preview = messages[0].get('content', '')[:50] if messages else 'Empty conversation'
            
            conversations.append({
                'id': conv_id,
                'created_at': conv_data.get('created_at'),
                'updated_at': conv_data.get('updated_at'),
                'message_count': len(messages),
                'preview': preview,
                'model': conv_data.get('model', 'unknown'),
                'personality': conv_data.get('personality', 'default')
            })
        
        # Sort by updated_at (most recent first)
        conversations.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return jsonify({'conversations': conversations})
        
    except Exception as e:
        logger.error(f"Error listing conversations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load/<conversation_id>', methods=['POST'])
def load_conversation(conversation_id):
    """Load a specific conversation"""
    try:
        if conversation_id in chatbot.conversations:
            session['session_id'] = conversation_id
            return jsonify({'status': 'success'})
        
        return jsonify({'error': 'Conversation not found'}), 404
        
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete/<conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation"""
    try:
        if conversation_id in chatbot.conversations:
            del chatbot.conversations[conversation_id]
            
            # Delete from disk
            filepath = os.path.join(CONVERSATIONS_DIR, f"{conversation_id}.json")
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted conversation file: {conversation_id}")
            
            # Clear session if it's the current conversation
            if session.get('session_id') == conversation_id:
                session['session_id'] = str(uuid.uuid4())
            
            return jsonify({'status': 'success'})
        
        return jsonify({'error': 'Conversation not found'}), 404
        
    except Exception as e:
        logger.error(f"Error deleting conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models"""
    return jsonify(MODELS)

@app.route('/api/personalities', methods=['GET'])
def get_personalities():
    """Get available personalities"""
    return jsonify(list(SYSTEM_PROMPTS.keys()))

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get API status"""
    return jsonify({
        'api_configured': bool(GEMINI_API_KEY),
        'models_available': list(MODELS.keys()) if GEMINI_API_KEY else [],
        'personalities_available': list(SYSTEM_PROMPTS.keys()),
        'active_session': session.get('session_id') is not None
    })

@app.route('/api/export', methods=['GET'])
def export_conversation():
    """Export current conversation as JSON"""
    try:
        session_id = session.get('session_id')
        if session_id and session_id in chatbot.conversations:
            return jsonify(chatbot.conversations[session_id])
        
        return jsonify({'error': 'No conversation found'}), 404
        
    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    
    print("\n" + "="*60)
    print("🚀 Gemini AI Chatbot Starting...")
    print("="*60)
    
    if GEMINI_API_KEY:
        print("✅ Gemini API configured successfully")
        print(f"📁 Conversations will be saved in: {CONVERSATIONS_DIR}/")
        print(f"🤖 Available models:")
        for model_name in MODELS.keys():
            print(f"   - {model_name}")
        print(f"🎭 Available personalities: {', '.join(SYSTEM_PROMPTS.keys())}")
    else:
        print("❌ ERROR: No Gemini API key found")
        print("📝 Please add GEMINI_API_KEY to your .env file")
        print("🔑 Get your API key from: https://makersuite.google.com/app/apikey")
    
    print(f"\n🌐 Server running at: http://localhost:{port}")
    print("="*60 + "\n")
    
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)