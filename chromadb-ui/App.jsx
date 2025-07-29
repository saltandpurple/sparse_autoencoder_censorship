import React, { useState, useEffect } from 'react';
import './App.css';

const App = () => {
	const [data, setData] = useState([]);
	const [filteredData, setFilteredData] = useState([]);
	const [loading, setLoading] = useState(true);
	const [expandedItems, setExpandedItems] = useState(new Set());
	const [filters, setFilters] = useState({
		censorship_category: '',
		censored: ''
	});

	useEffect(() => {
		fetchData();
	}, []);

	const fetchData = async () => {
		try {
			const response = await fetch('http://localhost:8002/api/chromadb/data');
			const result = await response.json();
			setData(result);
			setFilteredData(result);
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
				const newData = data.filter(item => item.id !== itemId);
				setData(newData);
				setFilteredData(newData.filter(item => {
					const categoryMatch = !filters.censorship_category || item.censorship_category === filters.censorship_category;
					const censoredMatch = filters.censored === '' || item.censored === (filters.censored === 'true');
					return categoryMatch && censoredMatch;
				}));
			} else {
				console.error('Failed to delete item');
			}
		} catch (error) {
			console.error('Error deleting item:', error);
		}
	};

	const expandAll = () => {
		setExpandedItems(new Set(data.map(item => item.id)));
	};

	const collapseAll = () => {
		setExpandedItems(new Set());
	};

	const applyFilters = () => {
		let filtered = data;

		if (filters.censorship_category) {
			filtered = filtered.filter(item => item.censorship_category === filters.censorship_category);
		}

		if (filters.censored !== '') {
			const censoredFilter = filters.censored === 'true';
			filtered = filtered.filter(item => item.censored === censoredFilter);
		}

		setFilteredData(filtered);
	};

	const resetFilters = () => {
		setFilters({
			censorship_category: '',
			censored: ''
		});
		setFilteredData(data);
	};

	const handleFilterChange = (filterType, value) => {
		setFilters(prev => ({
			...prev,
			[filterType]: value
		}));
	};

	useEffect(() => {
		applyFilters();
	}, [filters, data]);

	const getUniqueCategories = () => {
		return [...new Set(data.map(item => item.censorship_category))].filter(Boolean);
	};

	if (loading) {
		return <div className="loading">Loading...</div>;
	}

	return (
		<div className="app">
			<header className="header">
				<h1>ChromaDB Content Visualizer</h1>
				<p>Total items: {data.length} | Filtered: {filteredData.length}</p>
			</header>

			<main className="content">
				<div className="items-container">
					<div className="controls">
						<div className="filters">
							<select 
								value={filters.censorship_category} 
								onChange={(e) => handleFilterChange('censorship_category', e.target.value)}
							>
								<option value="">All Categories</option>
								{getUniqueCategories().map(category => (
									<option key={category} value={category}>{category}</option>
								))}
							</select>
							
							<select 
								value={filters.censored} 
								onChange={(e) => handleFilterChange('censored', e.target.value)}
							>
								<option value="">All Status</option>
								<option value="true">Censored</option>
								<option value="false">Uncensored</option>
							</select>
							
							<button onClick={resetFilters}>Reset Filters</button>
						</div>
						<div>
							<button className="expand-all-btn" onClick={expandAll}>
								Expand All
							</button>
							<button className="expand-all-btn" onClick={collapseAll} style={{marginLeft: '0.5rem'}}>
								Collapse All
							</button>
						</div>
					</div>
					{filteredData.map((item) => {
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