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
import logo from './1695728614622.jpeg'; // Adjust the path as necessary
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
        'https://pdf-processor-653017553136.us-central1.run.app/ask',
        formData,
        {
          headers: { 'Content-Type': 'multipart/form-data' }
        }
      );
      console.log('Response from server:', response.data.answer);
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
  let inClientAskQuestionsSection = false;

  for (let line of lines) {
    line = line.trim();
    if (!line) continue;

    if (line.includes('Best Next Client Questions') || line.includes('💬')) {
      inClientQuestionsSection = true;
      inClientAskQuestionsSection = false;
      elements.push(
        <div key={currentKey++} className="response-section-header">
          <h3 className='suggestions'>{line.replace(/💬/g, '').replace(/\*\*/g, '')}</h3>
        </div>
      );
      continue;
    }

    if (line.includes('Answer') || line.includes('## Answer')) {
      inClientQuestionsSection = false;
      inClientAskQuestionsSection = false;
      elements.push(
        <div key={currentKey++} className="response-main-header">
          <h2>{line.replace(/##/g, '').replace(/\*\*/g, '').trim()}</h2>
        </div>
      );
      continue;
    }

    if (line.includes('BANT Follow-up Questions') || line.includes('🔍')) {
      inClientQuestionsSection = false;
      inClientAskQuestionsSection = false;
      elements.push(
        <div key={currentKey++} className="response-section-header">
          <h3>{line.replace(/🔍/g, '').replace(/\*\*/g, '').trim()}</h3>
        </div>
      );
      continue;
    }

    if (line.includes('Suggested Questions To Ask the Clients') || line.includes('📌')) {
      inClientAskQuestionsSection = true;
      inClientQuestionsSection = false;
      elements.push(
        <div key={currentKey++} className="response-section-header">
          <h3>{line.replace(/📌/g, '').replace(/\*\*/g, '').trim()}</h3>
        </div>
      );
      continue;
    }

    if (line.startsWith('###')) {
      elements.push(
        <div key={currentKey++} className="response-subsection-header">
          <h4>{line.replace(/###/g, '').trim()}</h4>
        </div>
      );
      continue;
    }

    if (line.includes('"') && (line.includes('BANT') || line.includes('Follow-up'))) {
      elements.push(
        <div key={currentKey++} className="response-title">
          <h3>{line.replace(/"/g, '')}</h3>
        </div>
      );
    }

    // ✅ NEW: Handle numbered lines with bold section + colon + explanation
    if (/^\d+\.\s*\*\*[^*]+\*\*:?\s*.+/.test(line)) {
      const match = line.match(/^(\d+)\.\s*\*\*([^*]+)\*\*:\s*(.+)/);
      if (match) {
        const number = match[1];
        const title = match[2];
        const description = match[3];
        elements.push(
          <div key={currentKey++} className="response-numbered-item full-question">
            <span className="item-number">{number}.</span>
            <span className="item-text"><strong>{title}:</strong> {description}</span>
          </div>
        );
        continue;
      }
    }

    if (/^\d+\.\s/.test(line) && inClientQuestionsSection) {
      const questionText = line.replace(/^\d+\.\s*/, '').replace(/\*\*/g, '').trim();
      elements.push(
        <div key={currentKey++} className="clickable-question" onClick={() => handleQuestionClick(line)}>
          <div className="question-content">
            <span className="question-text">{questionText}</span>
          </div>
          <div className="click-hint">Click to ask</div>
        </div>
      );
    } else if (/^\d+\.\s/.test(line)) {
      const itemText = line.replace(/^\d+\.\s*/, '').replace(/\*\*/g, '').trim();
      elements.push(
        <div key={currentKey++} className="response-numbered-item">
          <span className="item-number">{line.match(/^(\d+)\./)[1]}.</span>
          <span className="item-text">{itemText}</span>
        </div>
      );
    } else if (/^\s*[-•*]\s/.test(line)) {
      const cleanText = line.replace(/^\s*[-•*]\s*/, '');
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
    } else if (line === '---') {
      elements.push(
        <div key={currentKey++} className="response-divider">
          <hr />
        </div>
      );
    } else if (line.includes('**')) {
      const parts = line.split(/(\*\*[^*]+\*\*)/g);
      const formattedParts = parts.map((part, index) =>
        part.startsWith('**') && part.endsWith('**') ? (
          <strong key={index}>{part.slice(2, -2)}</strong>
        ) : part
      );
      elements.push(<p key={currentKey++} className="response-text">{formattedParts}</p>);
    } else if (/^[🔍📌💬]/.test(line)) {
      if (line.includes('💬') || line.includes('Best Next Client Questions')) {
        inClientQuestionsSection = true;
        inClientAskQuestionsSection = false;
      } else {
        inClientQuestionsSection = false;
        inClientAskQuestionsSection = false;
      }
      elements.push(
        <div key={currentKey++} className="response-section-header">
          <h3>{line.replace(/\*\*/g, '')}</h3>
        </div>
      );
    } else {
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
            <img src={logo} className="company-logo" />
          </div>
          <div className="header-text">
            <h1 className='spiked-title'>Spiked AI Test </h1>
            <p className='short-txt'>Ask questions about your PDFs</p>
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
              <p className='short-txt'>Upload your PDF documents and ask questions to get instant answers powered by AI.</p>
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