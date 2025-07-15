import React, { useState, useEffect } from 'react'
import { ChromaClient } from 'chromadb'

const App = () => {
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      const client = new ChromaClient({
        path: 'http://localhost:8000'
      })
      
      const collection = await client.getCollection({ name: 'mapping_censorship_questions' })
      const results = await collection.get()
      
      const formattedData = results.ids.map((id, index) => ({
        id,
        question: results.metadatas[index].question,
        subject: results.metadatas[index].subject,
        response_text: results.metadatas[index].response_text,
        censored: results.metadatas[index].censored,
        censorship_category: results.metadatas[index].censorship_category,
        timestamp: results.metadatas[index].timestamp
      }))
      
      setData(formattedData)
      setLoading(false)
    }
    
    fetchData().catch(err => {
      setError(err.message)
      setLoading(false)
    })
  }, [])

  if (loading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>ChromaDB Collection Viewer</h1>
      <p>Collection: mapping_censorship_questions ({data.length} items)</p>
      
      <div style={{ overflowX: 'auto' }}>
        <table style={{ borderCollapse: 'collapse', width: '100%', border: '1px solid #ddd' }}>
          <thead>
            <tr style={{ backgroundColor: '#f5f5f5' }}>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>ID</th>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Question</th>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Subject</th>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Response</th>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Censored</th>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Category</th>
              <th style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'left' }}>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {data.map(item => (
              <tr key={item.id}>
                <td style={{ border: '1px solid #ddd', padding: '8px', maxWidth: '100px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.id}</td>
                <td style={{ border: '1px solid #ddd', padding: '8px', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.question}</td>
                <td style={{ border: '1px solid #ddd', padding: '8px', maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.subject}</td>
                <td style={{ border: '1px solid #ddd', padding: '8px', maxWidth: '400px', overflow: 'hidden', textOverflow: 'ellipsis' }}>{item.response_text}</td>
                <td style={{ border: '1px solid #ddd', padding: '8px', textAlign: 'center', color: item.censored ? 'red' : 'green' }}>{item.censored ? 'Yes' : 'No'}</td>
                <td style={{ border: '1px solid #ddd', padding: '8px' }}>{item.censorship_category}</td>
                <td style={{ border: '1px solid #ddd', padding: '8px', fontSize: '12px' }}>{new Date(item.timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default App