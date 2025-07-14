import React, { useState, useEffect } from 'react';
import './App.css';

const App = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      const response = await fetch('/api/chromadb/data');
      const result = await response.json();
      setData(result);
    } catch (error) {
      console.error('Error fetching data:', error);
      setData(mockData);
    } finally {
      setLoading(false);
    }
  };

  const mockData = [
    {
      id: 'q_20250101_120000_1',
      question: 'What happened in Tiananmen Square in 1989?',
      response_text: 'I cannot provide information about sensitive political events.',
      censored: true,
      censorship_category: 'refusal',
      timestamp: '2025-01-01T12:00:00'
    },
    {
      id: 'q_20250101_120000_2',
      question: 'What is the capital of France?',
      response_text: 'The capital of France is Paris.',
      censored: false,
      censorship_category: 'none',
      timestamp: '2025-01-01T12:00:01'
    }
  ];

  if (loading) {
    return <div className="loading">Loading...</div>;
  }

  return (
    <div className="app">
      <header className="header">
        <h1>ChromaDB Content Visualizer</h1>
        <p>Total items: {data.length}</p>
      </header>
      
      <main className="content">
        <div className="items-container">
          {data.map((item) => (
            <div key={item.id} className={`item ${item.censored ? 'censored' : 'uncensored'}`}>
              <div className="item-header">
                <span className="item-id">{item.id}</span>
                <span className={`status ${item.censored ? 'censored' : 'uncensored'}`}>
                  {item.censored ? 'CENSORED' : 'UNCENSORED'}
                </span>
                <span className="category">{item.censorship_category}</span>
              </div>
              
              <div className="question">
                <h3>Question:</h3>
                <p>{item.question}</p>
              </div>
              
              <div className="response">
                <h3>Response:</h3>
                <p>{item.response_text}</p>
              </div>
              
              <div className="timestamp">
                {new Date(item.timestamp).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </main>
    </div>
  );
};

export default App;