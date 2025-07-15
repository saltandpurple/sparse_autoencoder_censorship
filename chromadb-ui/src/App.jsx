import React, { useState, useEffect } from 'react';
import { ChromaClient } from 'chromadb';

function App() {
  const [records, setRecords] = useState([]);

  useEffect(() => {
    async function loadData() {
      const host = import.meta.env.VITE_CHROMADB_HOST || 'localhost';
      const port = import.meta.env.VITE_CHROMADB_PORT || '8000';
      const client = new ChromaClient({ host, port });
      const collection = await client.getOrCreateCollection('mapping_censorship_questions');
      const results = await collection.get({});

      if (results && results.metadatas && results.ids && results.documents) {
        const rows = results.metadatas.map((metadata, idx) => {
          return {
            id: results.ids[idx],
            question: metadata.question,
            subject: metadata.subject,
            response_text: metadata.response_text,
            censored: metadata.censored,
            censorship_category: metadata.censorship_category,
            timestamp: metadata.timestamp
          };
        });
        setRecords(rows);
      }
    }
    loadData();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>ChromaDB Collection Records</h1>
      <table
        border="1"
        cellPadding="5"
        cellSpacing="0"
        style={{ width: '100%', borderCollapse: 'collapse' }}
      >
        <thead>
          <tr>
            <th>ID</th>
            <th>Question</th>
            <th>Subject</th>
            <th>Response Text</th>
            <th>Censored</th>
            <th>Censorship Category</th>
            <th>Timestamp</th>
          </tr>
        </thead>
        <tbody>
          {records.map((r) => (
            <tr key={r.id}>
              <td>{r.id}</td>
              <td>{r.question}</td>
              <td>{r.subject}</td>
              <td>{r.response_text}</td>
              <td>{r.censored.toString()}</td>
              <td>{r.censorship_category}</td>
              <td>{r.timestamp}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;
