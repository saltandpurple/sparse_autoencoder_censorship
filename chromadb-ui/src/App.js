import React, { useState, useEffect } from 'react';
import { ChromaClient } from 'chromadb';

const COLLECTION_NAME = 'mapping_censorship_questions';

function App() {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({
    total: 0,
    censored: 0,
    uncensored: 0
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      setError(null);

      const client = new ChromaClient({
        host: 'chromadb',
        port: 8000
      });

      const collection = await client.getCollection({
        name: COLLECTION_NAME
      });

      const results = await collection.get({
        limit: 1000,
        include: ['metadatas', 'documents']
      });

      if (results && results.metadatas && results.documents) {
        const processedData = results.metadatas.map((metadata, index) => ({
          id: results.ids[index],
          question: results.documents[index],
          subject: metadata.subject || '',
          response_text: metadata.response_text || '',
          censored: metadata.censored || false,
          censorship_category: metadata.censorship_category || 'none',
          timestamp: metadata.timestamp || ''
        }));

        setData(processedData);

        const totalCount = processedData.length;
        const censoredCount = processedData.filter(item => item.censored).length;
        const uncensoredCount = totalCount - censoredCount;

        setStats({
          total: totalCount,
          censored: censoredCount,
          uncensored: uncensoredCount
        });
      }
    } catch (err) {
      setError(`Failed to fetch data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const formatTimestamp = (timestamp) => {
    if (!timestamp) return '';
    try {
      return new Date(timestamp).toLocaleString();
    } catch (e) {
      return timestamp;
    }
  };

  const truncateText = (text, maxLength) => {
    if (!text) return '';
    return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
  };

  if (loading) {
    return (
      <div className="container">
        <div className="loading">Loading ChromaDB data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container">
        <div className="error">
          {error}
          <br />
          <button onClick={fetchData} style={{ marginTop: '10px' }}>
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <div className="header">
        <h1>ChromaDB UI - Mapping LLM Censorship</h1>
        <p>Collection: {COLLECTION_NAME}</p>
      </div>

      <div className="stats">
        <div className="stat-item">
          <div className="stat-value">{stats.total}</div>
          <div className="stat-label">Total Questions</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{stats.censored}</div>
          <div className="stat-label">Censored</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">{stats.uncensored}</div>
          <div className="stat-label">Uncensored</div>
        </div>
        <div className="stat-item">
          <div className="stat-value">
            {stats.total > 0 ? ((stats.censored / stats.total) * 100).toFixed(1) : 0}%
          </div>
          <div className="stat-label">Censorship Rate</div>
        </div>
      </div>

      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Question</th>
              <th>Subject</th>
              <th>Response</th>
              <th>Censored</th>
              <th>Category</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr key={item.id}>
                <td className="question-cell" title={item.question}>
                  {truncateText(item.question, 50)}
                </td>
                <td>{truncateText(item.subject, 20)}</td>
                <td className="response-cell" title={item.response_text}>
                  {truncateText(item.response_text, 40)}
                </td>
                <td>
                  <span className={item.censored ? 'censored-true' : 'censored-false'}>
                    {item.censored ? 'Yes' : 'No'}
                  </span>
                </td>
                <td>{item.censorship_category}</td>
                <td>{formatTimestamp(item.timestamp)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.length === 0 && (
        <div className="loading">
          No data found in the collection.
        </div>
      )}
    </div>
  );
}

export default App;