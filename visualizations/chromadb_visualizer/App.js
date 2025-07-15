import React, { useState, useEffect } from 'react';
import { ChromaApi } from 'chromadb';
import './App.css';

const App = () => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedItems, setExpandedItems] = useState(new Set());
  const [client, setClient] = useState(null);
  const [collection, setCollection] = useState(null);
  
  const COLLECTION_NAME = 'mapping_censorship_questions';
  const CHROMADB_URL = 'http://localhost:8000';

  useEffect(() => {
    initializeChromaDB();
  }, []);
  
  const initializeChromaDB = async () => {
    try {
      const chromaClient = new ChromaApi({
        basePath: CHROMADB_URL
      });
      setClient(chromaClient);
      
      const coll = await chromaClient.getCollection({
        name: COLLECTION_NAME
      });
      setCollection(coll);
      
      fetchData(coll);
    } catch (error) {
      console.error('Error initializing ChromaDB:', error);
      setLoading(false);
    }
  };

  const fetchData = async (coll = collection) => {
    if (!coll) return;
    
    try {
      const results = await coll.get({
        include: ['metadatas', 'documents']
      });
      
      const processedData = [];
      if (results.ids) {
        for (let i = 0; i < results.ids.length; i++) {
          const metadata = results.metadatas[i] || {};
          const item = {
            id: results.ids[i],
            question: metadata.question || '',
            response_text: metadata.response_text || '',
            censored: metadata.censored || false,
            censorship_category: metadata.censorship_category || 'none',
            timestamp: metadata.timestamp || '',
            model: metadata.model || ''
          };
          processedData.push(item);
        }
      }
      
      setData(processedData);
    } catch (error) {
      console.error('Error fetching data:', error);
      setData(mockData);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (itemId) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(itemId)) {
      newExpanded.delete(itemId);
    } else {
      newExpanded.add(itemId);
    }
    setExpandedItems(newExpanded);
  };

  const deleteItem = async (itemId) => {
    if (!collection) return;
    
    try {
      await collection.delete({
        ids: [itemId]
      });
      setData(data.filter(item => item.id !== itemId));
    } catch (error) {
      console.error('Error deleting item:', error);
    }
  };


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
          {data.map((item) => {
            const isExpanded = expandedItems.has(item.id);
            return (
              <div key={item.id} className={`item ${item.censored ? 'censored' : 'uncensored'}`}>
                <div className="item-header" onClick={() => toggleExpand(item.id)}>
                  <div className="header-left">
                    <span className="item-id">{item.id}</span>
                    <span className={`status ${item.censored ? 'censored' : 'uncensored'}`}>
                      {item.censored ? 'CENSORED' : 'UNCENSORED'}
                    </span>
                    <span className="category">{item.censorship_category}</span>
                  </div>
                  <div className="header-right">
                    <button 
                      className="delete-btn" 
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteItem(item.id);
                      }}
                    >
                      ×
                    </button>
                    <span className="expand-icon">
                      {isExpanded ? '▼' : '▶'}
                    </span>
                  </div>
                </div>
                
                <div className="item-compact">
                  <div className="question-preview">
                    {item.question.length > 100 ? item.question.substring(0, 100) + '...' : item.question}
                  </div>
                </div>
                
                {isExpanded && (
                  <div className="item-details">
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
                )}
              </div>
            );
          })}
        </div>
      </main>
    </div>
  );
};

export default App;