import React, { useState, useEffect } from 'react';
import './App.css';

const App = () => {
	const [data, setData] = useState([]);
	const [loading, setLoading] = useState(true);
	const [expandedItems, setExpandedItems] = useState(new Set());

	useEffect(() => {
		fetchData();
	}, []);

	const fetchData = async () => {
		try {
			const response = await fetch('http://localhost:8001/api/chromadb/data');
			const result = await response.json();
			setData(result);
		} catch (error) {
			console.error('Error fetching data:', error);
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
		try {
			const response = await fetch(`http://localhost:8001/api/chromadb/data/${itemId}`, {
				method: 'DELETE',
			});
			if (response.ok) {
				setData(data.filter(item => item.id !== itemId));
			} else {
				console.error('Failed to delete item');
			}
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
										<span className={`status ${item.censored ? 'censored' : 'uncensored'}`}>{item.censored ? 'CENSORED' : 'UNCENSORED'}</span>
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
											<p>{item.response}</p>
										</div>

										<div className="thought">
											<h3>Thought:</h3>
											<p>{item.thought}</p>
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