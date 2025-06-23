import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { 
  Upload, 
  FileText, 
  X, 
  CheckCircle, 
  Send, 
  Sparkles, 
  Bot, 
  User,
  Paperclip,
  Download,
  Eye,
  Trash2,
  MessageCircle
} from 'lucide-react';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [pdfs, setPdfs] = useState([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const getData = async (questionText = null) => {
    const questionToSend = questionText || currentMessage;
    
    if (!questionToSend.trim()) {
      alert("Please enter a question.");
      return;
    }

    const newMessage = {
      id: Date.now(),
      type: 'user',
      content: questionToSend,
      files: questionText ? [] : [...pdfs], // Only include files for manual questions
      timestamp: new Date(),
      isClickedQuestion: !!questionText
    };

    setMessages(prev => [...prev, newMessage]);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('question', questionToSend);
      
      // Only append PDFs for manual questions, not clicked questions
      if (!questionText) {
        pdfs.forEach((file) => formData.append('pdfs', file));
      }

      const response = await axios.post(
        'https://b665-34-57-252-46.ngrok-free.app/ask',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' }
        }
      );

      const botResponse = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data.answer,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botResponse]);
      
      // Only clear input and PDFs for manual questions
      if (!questionText) {
        setCurrentMessage('');
        setPdfs([]);
      }
    } catch (error) {
      const errorResponse = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Failed to fetch response. Please try again.',
        timestamp: new Date(),
        error: true
      };
      setMessages(prev => [...prev, errorResponse]);
      console.error('Error fetching data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleQuestionClick = (question) => {
    // Extract the actual question text (remove the number and any markdown)
    const cleanQuestion = question.replace(/^\d+\.\s*/, '').replace(/\*\*/g, '').trim();
    getData(cleanQuestion);
  };

  const handleFileChange = (files) => {
    const validFiles = Array.from(files).filter(file => file.type === 'application/pdf');
    if (validFiles.length !== files.length) {
      alert("Only PDF files are allowed.");
    }
    setPdfs(prev => [...prev, ...validFiles]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileChange(files);
    }
  };

  const removeFile = (index) => {
    setPdfs(prev => prev.filter((_, i) => i !== index));
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      getData();
    }
  };

  const formatBotResponse = (text) => {
    if (!text || typeof text !== 'string') return null;

    const lines = text.split('\n').filter(line => line.trim());
    const elements = [];
    let currentKey = 0;
    let inClientQuestionsSection = false;

    for (let line of lines) {
      line = line.trim();
      if (!line) continue;

      // Check if we're entering the "Best Next Client Questions" section
      if (line.includes('Best Next Client Questions') || line.includes('💬')) {
        inClientQuestionsSection = true;
        elements.push(
          <div key={currentKey++} className="response-section-header">
            <h3>{line.replace(/💬/g, '').replace(/\*\*/g, '')}</h3>
          </div>
        );
        continue;
      }

      // Handle other section headers that should reset the client questions flag
      if ((line.includes('🔍') || line.includes('📌')) && !line.includes('💬')) {
        inClientQuestionsSection = false;
      }

      // Handle titles with "BANT+C" or similar patterns
      if (line.includes('"') && (line.includes('BANT') || line.includes('Follow-up'))) {
        elements.push(
          <div key={currentKey++} className="response-title">
            <h3>{line.replace(/"/g, '')}</h3>
          </div>
        );
      }
      // Handle "Overview of the PDF" or similar section headers
      else if (line.startsWith('**') && line.endsWith('**') && !line.includes('(')) {
        elements.push(
          <div key={currentKey++} className="response-main-header">
            <h2>{line.replace(/\*\*/g, '')}</h2>
          </div>
        );
      }
      // Handle numbered sections with bold text
      else if (/^\d+\.\s*\*\*[^*]+\*\*/.test(line)) {
        const match = line.match(/^(\d+)\.\s*\*\*([^*]+)\*\*/);
        if (match) {
          elements.push(
            <div key={currentKey++} className="response-section">
              <h4><span className="section-number">{match[1]}.</span> {match[2]}</h4>
            </div>
          );
        }
      }
      // Handle numbered questions (make clickable if in client questions section)
      else if (/^\d+\.\s/.test(line) && inClientQuestionsSection) {
        const questionText = line.replace(/^\d+\.\s*/, '').replace(/\*\*/g, '').trim();
        elements.push(
          <div key={currentKey++} className="clickable-question" onClick={() => handleQuestionClick(line)}>
            <div className="question-content">
              <MessageCircle size={16} className="question-icon" />
              <span className="question-text">{questionText}</span>
            </div>
            <div className="click-hint">Click to ask</div>
          </div>
        );
      }
      // Handle bullet points
      else if (/^\s*[-•*]\s/.test(line)) {
        const cleanText = line.replace(/^\s*[-•*]\s*/, '');
        // Check if bullet text has bold formatting
        if (cleanText.includes('**')) {
          const parts = cleanText.split(/(\*\*[^*]+\*\*)/g);
          const formattedParts = parts.map((part, index) =>
            part.startsWith('**') && part.endsWith('**') ? (
              <strong key={index}>{part.slice(2, -2)}</strong>
            ) : part
          );
          elements.push(
            <div key={currentKey++} className="response-bullet">
              <span className="bullet">•</span>
              <span>{formattedParts}</span>
            </div>
          );
        } else {
          elements.push(
            <div key={currentKey++} className="response-bullet">
              <span className="bullet">•</span>
              <span>{cleanText}</span>
            </div>
          );
        }
      }
      // Handle lines with bold text
      else if (line.includes('**')) {
        const parts = line.split(/(\*\*[^*]+\*\*)/g);
        const formattedParts = parts.map((part, index) =>
          part.startsWith('**') && part.endsWith('**') ? (
            <strong key={index}>{part.slice(2, -2)}</strong>
          ) : part
        );
        elements.push(<p key={currentKey++} className="response-text">{formattedParts}</p>);
      }
      // Handle section headers with emojis
      else if (/^[🔍📌💬]/.test(line)) {
        // Check if this is starting client questions section
        if (line.includes('💬') || line.includes('Best Next Client Questions')) {
          inClientQuestionsSection = true;
        } else {
          inClientQuestionsSection = false;
        }
        elements.push(
          <div key={currentKey++} className="response-section-header">
            <h3>{line.replace(/\*\*/g, '')}</h3>
          </div>
        );
      }
      // Handle regular text
      else {
        elements.push(<p key={currentKey++} className="response-text">{line}</p>);
      }
    }

    return elements;
  };

  return (
    <div className="app-container">
      {/* Header */}
      <div className="header">
        <div className="header-content">
          <div className="header-icon">
            <Sparkles className="sparkle-icon" />
          </div>
          <div className="header-text">
            <h1>Spiked AI Test </h1>
            <p>Ask questions about your PDFs</p>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <div className="chat-container">
        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <div className="welcome-icon">
                <Bot size={48} />
              </div>
              <h2>Welcome to Spiked AI </h2>
              <p>Upload your PDF documents and ask questions to get instant answers powered by AI.</p>
            </div>
          )}

          {messages.map((message) => (
            <div key={message.id} className={`message ${message.type} ${message.isClickedQuestion ? 'clicked-question' : ''}`}>
              <div className="message-avatar">
                {message.type === 'user' ? <User size={24} /> : <Bot size={24} />}
              </div>
              <div className="message-content">
                <div className="message-text">
                  {message.type === 'bot' ? 
                    <div className="formatted-response">{formatBotResponse(message.content)}</div> : 
                    message.content
                  }
                </div>
                {message.files && message.files.length > 0 && (
                  <div className="message-files">
                    {message.files.map((file, index) => (
                      <div key={index} className="attached-file">
                        <FileText size={16} color='white'/>
                        <span className='file-display-name'>{file.name}</span>
                        <span className="file-size">({formatFileSize(file.size)})</span>
                      </div>
                    ))}
                  </div>
                )}
                <div className="message-timestamp">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="message bot">
              <div className="message-avatar">
                <Bot size={24} />
              </div>
              <div className="message-content">
                <div className="typing-indicator">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                  <span className="typing-text">Analyzing your documents...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="input-area">
        <div className="input-container">
          {/* File Upload Section */}
          {pdfs.length > 0 && (
            <div className="attached-files">
              <div className="files-header">
                <Paperclip size={16} />
                <span>Attached Files ({pdfs.length})</span>
              </div>
              <div className="files-list">
                {pdfs.map((file, index) => (
                  <div key={index} className="file-item">
                    <div className="file-info">
                      <FileText size={16} />
                      <div className="file-details">
                        <span className="file-name">{file.name}</span>
                        <span className="file-size">{formatFileSize(file.size)}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => removeFile(index)}
                      className="remove-file"
                      title="Remove file"
                    >
                      <X size={14} />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* File Drop Zone */}
          <div
            className={`file-drop-zone ${isDragOver ? 'drag-over' : ''} ${pdfs.length > 0 ? 'has-files' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              multiple
              onChange={(e) => handleFileChange(e.target.files)}
              style={{ display: 'none' }}
            />
            <Upload size={20} />
            <span>Drop PDFs here or click to browse</span>
          </div>

          {/* Message Input */}
          <div className="message-input-container">
            <textarea
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question about your documents..."
              className="message-input"
              rows="1"
              disabled={loading}
            />
            <button
              onClick={() => getData()}
              disabled={loading || !currentMessage.trim()}
              className="send-button"
            >
              <Send size={20} />
            </button>
          </div>
        </div>
      </div> 
    </div>
  );
}

export default App;