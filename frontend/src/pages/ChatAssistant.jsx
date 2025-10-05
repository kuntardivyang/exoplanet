import { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Loader2, RotateCcw, BookOpen, Lightbulb } from 'lucide-react';
import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function ChatAssistant() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [suggestions, setSuggestions] = useState([]);
  const [sources, setSources] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async (messageText = input) => {
    if (!messageText.trim()) return;

    const userMessage = {
      role: 'user',
      content: messageText,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(`${API_URL}/chat/query`, {
        message: messageText,
        conversation_id: conversationId,
        include_context: true,
        n_context: 3,
      }, {
        timeout: 60000  // 60 second timeout for LLM responses
      });

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        timestamp: new Date().toISOString(),
        sources: response.data.sources || [],
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setSuggestions(response.data.suggestions || []);
      setSources(response.data.sources || []);
      if (response.data.conversation_id) {
        setConversationId(response.data.conversation_id);
      }
    } catch (error) {
      const errorMessage = {
        role: 'assistant',
        content: `Error: ${error.response?.data?.detail || 'Could not get response. Make sure Ollama is running.'}`,
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    sendMessage(suggestion);
    setSuggestions([]);
  };

  const clearConversation = () => {
    setMessages([]);
    setSuggestions([]);
    setSources([]);
    setConversationId(null);
  };

  const quickQuestions = [
    "How does the transit method work?",
    "Compare Kepler and TESS datasets",
    "Which model has the best accuracy?",
    "Explain a false positive classification",
    "What are hot Jupiters?",
  ];

  return (
    <div className="h-[calc(100vh-200px)] flex flex-col">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-4xl font-bold gradient-text mb-2 flex items-center">
          <Sparkles className="w-10 h-10 mr-3 text-primary-500" />
          AI Assistant
        </h1>
        <p className="text-gray-400">Ask me anything about exoplanet detection, models, or datasets</p>
      </div>

      <div className="flex-1 flex gap-6 min-h-0">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col card min-h-0">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto mb-4 space-y-4 pr-2">
            {messages.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-center p-8">
                <Sparkles className="w-16 h-16 text-primary-500 mb-4" />
                <h2 className="text-2xl font-bold mb-2">Welcome to the Exoplanet AI Assistant!</h2>
                <p className="text-gray-400 mb-6">
                  I can help you understand exoplanet detection, explain model predictions,
                  and compare datasets.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 w-full max-w-2xl">
                  {quickQuestions.map((q, i) => (
                    <button
                      key={i}
                      onClick={() => sendMessage(q)}
                      className="text-left p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-sm"
                    >
                      <Lightbulb className="w-4 h-4 inline mr-2 text-primary-500" />
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={idx}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      msg.role === 'user'
                        ? 'bg-primary-600 text-white'
                        : msg.isError
                        ? 'bg-red-500/20 border border-red-500'
                        : 'bg-dark-700'
                    }`}
                  >
                    <div className="flex items-start space-x-2 mb-2">
                      {msg.role === 'assistant' && (
                        <Sparkles className="w-5 h-5 text-primary-500 flex-shrink-0 mt-1" />
                      )}
                      <div className="flex-1">
                        <p className="text-xs text-gray-400 mb-1">
                          {msg.role === 'user' ? 'You' : 'AI Assistant'}
                        </p>
                        <div className="prose prose-invert max-w-none">
                          <p className="whitespace-pre-wrap">{msg.content}</p>
                        </div>
                      </div>
                    </div>
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-3 pt-3 border-t border-dark-600">
                        <p className="text-xs text-gray-500 mb-2">Sources:</p>
                        <div className="space-y-1">
                          {msg.sources.slice(0, 2).map((source, i) => (
                            <div key={i} className="text-xs text-gray-400 bg-dark-800 p-2 rounded">
                              {source.document}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-dark-700 rounded-lg p-4">
                  <Loader2 className="w-5 h-5 text-primary-500 animate-spin" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Suggestions */}
          {suggestions.length > 0 && !loading && (
            <div className="mb-4 p-3 bg-dark-700 rounded-lg">
              <p className="text-xs text-gray-400 mb-2">Suggested questions:</p>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, i) => (
                  <button
                    key={i}
                    onClick={() => handleSuggestionClick(suggestion)}
                    className="text-xs px-3 py-1 bg-primary-500/20 hover:bg-primary-500/30 text-primary-400 rounded-full transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && !loading && sendMessage()}
              placeholder="Ask me anything about exoplanets..."
              className="input flex-1"
              disabled={loading}
            />
            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="btn-primary flex items-center space-x-2"
            >
              {loading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
            {messages.length > 0 && (
              <button
                onClick={clearConversation}
                className="btn-secondary"
                title="Clear conversation"
              >
                <RotateCcw className="w-5 h-5" />
              </button>
            )}
          </div>
        </div>

        {/* Sidebar - Context & Sources */}
        <div className="w-80 space-y-4">
          {/* Sources Panel */}
          <div className="card">
            <h3 className="text-lg font-bold mb-3 flex items-center">
              <BookOpen className="w-5 h-5 mr-2 text-primary-500" />
              Context Sources
            </h3>
            {sources.length > 0 ? (
              <div className="space-y-2 max-h-60 overflow-y-auto">
                {sources.map((source, i) => (
                  <div key={i} className="bg-dark-700 p-3 rounded-lg text-sm">
                    <p className="text-gray-300 mb-1">{source.document}</p>
                    {source.metadata && (
                      <div className="flex items-center justify-between mt-2">
                        <span className="text-xs text-gray-500">
                          {source.metadata.type}
                        </span>
                        {source.relevance && (
                          <span className="text-xs text-primary-400">
                            {(source.relevance * 100).toFixed(0)}% relevant
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-500 text-sm">
                Sources will appear here when you ask questions
              </p>
            )}
          </div>

          {/* Quick Actions */}
          <div className="card">
            <h3 className="text-lg font-bold mb-3">Quick Actions</h3>
            <div className="space-y-2">
              <button
                onClick={() => sendMessage("Explain the best model's performance")}
                className="w-full text-left p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-sm"
              >
                📊 Model Performance
              </button>
              <button
                onClick={() => sendMessage("Compare all datasets")}
                className="w-full text-left p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-sm"
              >
                📁 Dataset Comparison
              </button>
              <button
                onClick={() => sendMessage("Show me light curve examples")}
                className="w-full text-left p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-sm"
              >
                🌊 Light Curves
              </button>
              <button
                onClick={() => sendMessage("Explain false positive detection")}
                className="w-full text-left p-3 bg-dark-700 hover:bg-dark-600 rounded-lg transition-colors text-sm"
              >
                🔍 Detection Methods
              </button>
            </div>
          </div>

          {/* Info */}
          <div className="card bg-primary-500/10 border border-primary-500/30">
            <h3 className="text-sm font-bold mb-2 text-primary-400">💡 Tip</h3>
            <p className="text-xs text-gray-400">
              I can explain predictions, compare models, and answer questions about your exoplanet
              data using RAG (Retrieval Augmented Generation) with your actual training results.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatAssistant;
