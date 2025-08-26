import React, { useState, useCallback, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, DollarSign, Target, Brain, Activity, Play, RefreshCw, Server, Wifi, WifiOff, Zap, CheckCircle, Clock, AlertTriangle, BarChart3, Settings, Database, Cpu, Shield, TrendingDown, X } from 'lucide-react';

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? 'https://alpha-factor-engine.onrender.com'
  : 'http://localhost:8000';

const App = () => {
  const [currentView, setCurrentView] = useState('overview');
  const [pipelineStatus, setPipelineStatus] = useState({ status: 'idle', progress: 0, current_step: '', error: null });
  const [isConnected, setIsConnected] = useState(false);
  const [data, setData] = useState({
    marketData: [],
    alphaFactors: [],
    backtestResults: [],
    metrics: {},
    modelResults: null,
    tradeAnalysis: null
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

// Remove API_BASE_URL from the dependency array
const apiCall = useCallback(async (endpoint, options = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      headers: { 'Content-Type': 'application/json', ...options.headers },
      ...options,
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }
    
    return await response.json();
  } catch (err) {
    console.error(`API call failed for ${endpoint}:`, err);
    throw err;
  }
}, []); // Empty array - API_BASE_URL is a constant

// checkConnection stays the same
const checkConnection = useCallback(async () => {
  try {
    await apiCall('/');
    setIsConnected(true);
    setError(null);
  } catch (err) {
    setIsConnected(false);
    setError('Backend connection failed');
  }
}, [apiCall]);

  // Start ML pipeline
  const startPipeline = async () => {
    try {
      setError(null);
      await apiCall('/pipeline/start', { method: 'POST' });
      pollStatus();
    } catch (err) {
      setError(`Failed to start pipeline: ${err.message}`);
      setPipelineStatus(prev => ({ ...prev, error: `Failed to start: ${err.message}` }));
    }
  };

  // Reset pipeline
  const resetPipeline = async () => {
    try {
      setError(null);
      await apiCall('/pipeline/reset', { method: 'POST' });
      setPipelineStatus({ status: 'idle', progress: 0, current_step: '', error: null });
      setData({ marketData: [], alphaFactors: [], backtestResults: [], metrics: {}, modelResults: null, tradeAnalysis: null });
    } catch (err) {
      setError(`Failed to reset: ${err.message}`);
      setPipelineStatus(prev => ({ ...prev, error: `Failed to reset: ${err.message}` }));
    }
  };

  // Poll pipeline status
  const pollStatus = async () => {
    try {
      const status = await apiCall('/status');
      setPipelineStatus(status);
      
      if (status.status === 'completed') {
        await loadAllData();
      } else if (status.status === 'running' || status.status === 'starting') {
        setTimeout(pollStatus, 2000);
      }
    } catch (err) {
      setError(`Status check failed: ${err.message}`);
      setPipelineStatus(prev => ({ ...prev, error: `Status check failed: ${err.message}` }));
    }
  };

  // Load all data from backend with better error handling
  const loadAllData = async () => {
    setLoading(true);
    try {
      console.log('Loading all data...');
      
      // Load data sequentially for better error handling
      let marketData = [];
      let alphaFactors = [];
      let backtestResults = [];
      let metrics = {};
      let modelResults = null;
      let tradeAnalysis = null;

      try {
        const marketResponse = await apiCall('/data/market');
        marketData = marketResponse.data || [];
        console.log('Market data loaded:', marketData.length);
      } catch (err) {
        console.warn('Failed to load market data:', err.message);
      }

      try {
        const factorsResponse = await apiCall('/data/factors');
        alphaFactors = factorsResponse.data || [];
        console.log('Alpha factors loaded:', alphaFactors.length);
      } catch (err) {
        console.warn('Failed to load alpha factors:', err.message);
      }

      try {
        const backtestResponse = await apiCall('/backtest/results');
        backtestResults = backtestResponse.data || [];
        console.log('Backtest results loaded:', backtestResults.length);
      } catch (err) {
        console.warn('Failed to load backtest results:', err.message);
      }

      try {
        metrics = await apiCall('/backtest/metrics');
        console.log('Metrics loaded:', Object.keys(metrics).length);
      } catch (err) {
        console.warn('Failed to load metrics:', err.message);
      }

      try {
        modelResults = await apiCall('/model/results');
        console.log('Model results loaded');
      } catch (err) {
        console.warn('Failed to load model results:', err.message);
      }

      try {
        tradeAnalysis = await apiCall('/backtest/trades');
        console.log('Trade analysis loaded');
      } catch (err) {
        console.warn('Failed to load trade analysis:', err.message);
      }

      setData({
        marketData,
        alphaFactors,
        backtestResults,
        metrics,
        modelResults,
        tradeAnalysis
      });

      setError(null);
    } catch (err) {
      console.error('Failed to load data:', err);
      setError(`Failed to load data: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

// Your useEffect
useEffect(() => {
  checkConnection();
  const interval = setInterval(checkConnection, 30000);
  return () => clearInterval(interval);
}, [checkConnection]);

  // Format data for chart display
  const formatChartData = (backtestData) => {
    if (!backtestData || !Array.isArray(backtestData) || backtestData.length === 0) {
      return [];
    }

    return backtestData.map(row => ({
      date: row.date,
      cumulative_return: (row.cumulative_return || 0) * 100,
      benchmark_return: (row.benchmark_return || 0) * 100,
      portfolio_value: row.portfolio_value || 100000,
      benchmark_value: row.benchmark_value || 100000
    }));
  };

  // Loading screen during pipeline execution
  if (pipelineStatus.status === 'running' || pipelineStatus.status === 'starting') {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-indigo-900 flex items-center justify-center">
        <div className="bg-white/10 backdrop-blur-lg border border-white/20 rounded-2xl p-12 max-w-2xl mx-4 text-center">
          <div className="relative mb-8">
            <Zap className="w-24 h-24 text-blue-400 mx-auto animate-pulse" />
            <div className="absolute inset-0 bg-blue-400/20 rounded-full blur-2xl animate-ping"></div>
          </div>
          
          <h2 className="text-4xl font-bold text-white mb-4">Alpha Factor Engine</h2>
          <p className="text-blue-200 mb-2">Quantitative Machine Learning Pipeline</p>
          <p className="text-blue-300 text-sm mb-8">Processing 10+ years of S&P 500 data with 25+ alpha factors</p>
          
          <div className="space-y-4 mb-8">
            {[
              { step: 'Market Data Generation', description: '2,500+ trading days of OHLCV data', progress: 25 },
              { step: 'Alpha Factor Engineering', description: 'Technical, momentum, and volatility indicators', progress: 50 },
              { step: 'ML Model Training', description: 'XGBoost + Random Forest ensemble', progress: 75 },
              { step: 'Portfolio Backtesting', description: 'Transaction costs & risk management', progress: 100 }
            ].map(({ step, description, progress }) => (
              <div key={step} className={`flex items-center p-4 rounded-lg ${
                pipelineStatus.progress >= progress ? 'bg-green-500/20 border border-green-400/30' : 
                pipelineStatus.progress >= progress - 25 ? 'bg-blue-500/20 border border-blue-400/30' : 'bg-gray-500/10 border border-gray-500/20'
              }`}>
                {pipelineStatus.progress >= progress ? (
                  <CheckCircle className="w-6 h-6 text-green-400 mr-4 flex-shrink-0" />
                ) : pipelineStatus.progress >= progress - 25 ? (
                  <div className="w-6 h-6 mr-4 flex-shrink-0">
                    <div className="animate-spin rounded-full h-6 w-6 border-2 border-blue-400 border-t-transparent"></div>
                  </div>
                ) : (
                  <Clock className="w-6 h-6 text-gray-400 mr-4 flex-shrink-0" />
                )}
                <div className="text-left flex-grow">
                  <span className="text-white font-semibold block">{step}</span>
                  <span className="text-blue-200 text-sm">{description}</span>
                </div>
              </div>
            ))}
          </div>
          
          <div className="bg-white/20 rounded-full h-4 mb-4">
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-600 h-4 rounded-full transition-all duration-500 shadow-lg"
              style={{ width: `${pipelineStatus.progress}%` }}
            />
          </div>
          
          <p className="text-blue-200 font-medium">{pipelineStatus.progress}% Complete - {pipelineStatus.current_step}</p>
          
          {pipelineStatus.error && (
            <div className="mt-6 p-4 bg-red-500/20 border border-red-400/30 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-red-400 mx-auto mb-2" />
              <p className="text-red-300 text-sm">{pipelineStatus.error}</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  // Main dashboard
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Header */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-gray-200 shadow-sm">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className="relative mr-6">
                <Brain className="w-10 h-10 text-blue-600" />
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full shadow-lg"></div>
              </div>
              <div className="ml-2">
                <h1 className="text-3xl font-bold text-gray-800">Alpha Factor Engine</h1>
                <p className="text-gray-600 font-medium">Quantitative Machine Learning Trading System</p>
                <p className="text-sm text-gray-500">S&P 500 • 25+ Alpha Factors • Ensemble ML Models</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className={`flex items-center px-4 py-2 rounded-full text-sm font-medium ${
                isConnected ? 'bg-green-100 text-green-800 border border-green-200' : 'bg-red-100 text-red-800 border border-red-200'
              }`}>
                {isConnected ? <Wifi className="w-4 h-4 mr-2" /> : <WifiOff className="w-4 h-4 mr-2" />}
                {isConnected ? 'FastAPI Connected' : 'Backend Offline'}
              </div>
              
              {pipelineStatus.status === 'idle' ? (
                <button
                  onClick={startPipeline}
                  disabled={!isConnected}
                  className="bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 px-6 py-3 rounded-lg font-semibold text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center shadow-lg transition-all duration-200 hover:shadow-xl"
                >
                  <Play className="w-5 h-5 mr-2" />
                  Execute Pipeline
                </button>
              ) : (
                <button
                  onClick={resetPipeline}
                  className="bg-gray-600 text-white px-6 py-3 rounded-lg font-semibold hover:bg-gray-700 flex items-center shadow-lg transition-all duration-200"
                >
                  <RefreshCw className="w-5 h-5 mr-2" />
                  Reset System
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-8">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg shadow-sm">
            <div className="flex items-center">
              <AlertTriangle className="w-5 h-5 text-red-600 mr-3" />
              <div>
                <span className="text-red-800 font-semibold">System Error</span>
                <p className="text-red-600 text-sm">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg shadow-sm">
            <div className="flex items-center">
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent mr-3"></div>
              <span className="text-blue-800 font-semibold">Loading quantitative data...</span>
            </div>
          </div>
        )}

        {/* No data state - Enhanced landing page */}
        {pipelineStatus.status === 'idle' && data.backtestResults.length === 0 ? (
          <div className="text-center py-16">
            <div className="relative mb-8">
              <Brain className="w-32 h-32 text-blue-500 mx-auto mb-6" />
              <div className="absolute inset-0 bg-blue-500/10 rounded-full blur-3xl"></div>
            </div>
            
            <h2 className="text-4xl font-bold text-gray-800 mb-4">Quantitative Alpha Discovery</h2>
            <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
              Advanced machine learning system for systematic alpha factor extraction from S&P 500 market data
            </p>
            
            {/* System Architecture Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto mb-12">
              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6 border-l-4 border-blue-500 bg-gradient-to-br from-blue-50 to-blue-100 hover:shadow-lg transition-all duration-300">
                <Database className="w-12 h-12 text-blue-600 mx-auto mb-4" />
                <h3 className="font-bold text-blue-800 mb-2">Market Data Engine</h3>
                <p className="text-blue-700 text-sm mb-3 leading-relaxed">10+ years of realistic S&P 500 OHLCV data generation</p>
                <span className="inline-block px-3 py-1 bg-blue-200 text-blue-800 rounded-full text-xs font-semibold">
                  2,500+ days
                </span>
              </div>

              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6 border-l-4 border-green-500 bg-gradient-to-br from-green-50 to-green-100 hover:shadow-lg transition-all duration-300">
                <Target className="w-12 h-12 text-green-600 mx-auto mb-4" />
                <h3 className="font-bold text-green-800 mb-2">Alpha Factor Engineering</h3>
                <p className="text-green-700 text-sm mb-3 leading-relaxed">Technical, momentum, volatility, and mean-reversion indicators</p>
                <span className="inline-block px-3 py-1 bg-green-200 text-green-800 rounded-full text-xs font-semibold">
                  25+ factors
                </span>
              </div>

              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6 border-l-4 border-purple-500 bg-gradient-to-br from-purple-50 to-purple-100 hover:shadow-lg transition-all duration-300">
                <Cpu className="w-12 h-12 text-purple-600 mx-auto mb-4" />
                <h3 className="font-bold text-purple-800 mb-2">ML Ensemble Models</h3>
                <p className="text-purple-700 text-sm mb-3 leading-relaxed">XGBoost + Random Forest with walk-forward validation</p>
                <span className="inline-block px-3 py-1 bg-purple-200 text-purple-800 rounded-full text-xs font-semibold">
                  Ensemble
                </span>
              </div>

              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6 border-l-4 border-orange-500 bg-gradient-to-br from-orange-50 to-orange-100 hover:shadow-lg transition-all duration-300">
                <TrendingUp className="w-12 h-12 text-orange-600 mx-auto mb-4" />
                <h3 className="font-bold text-orange-800 mb-2">Portfolio Backtesting</h3>
                <p className="text-orange-700 text-sm mb-3 leading-relaxed">Realistic transaction costs and risk management</p>
                <span className="inline-block px-3 py-1 bg-orange-200 text-orange-800 rounded-full text-xs font-semibold">
                  Full simulation
                </span>
              </div>
            </div>

            {/* Technical Specifications */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 max-w-6xl mx-auto mb-12">
              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                <Shield className="w-8 h-8 text-blue-600 mx-auto mb-4" />
                <h3 className="text-lg font-bold text-gray-800 mb-3">Risk Management</h3>
                <ul className="text-gray-600 text-sm space-y-2 text-left">
                  <li>• Dynamic position sizing</li>
                  <li>• Maximum drawdown controls</li>
                  <li>• Volatility-based risk scaling</li>
                  <li>• Transaction cost modeling</li>
                </ul>
              </div>
              
              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                <BarChart3 className="w-8 h-8 text-green-600 mx-auto mb-4" />
                <h3 className="text-lg font-bold text-gray-800 mb-3">Performance Analytics</h3>
                <ul className="text-gray-600 text-sm space-y-2 text-left">
                  <li>• Information coefficient tracking</li>
                  <li>• Factor attribution analysis</li>
                  <li>• Risk-adjusted returns (Sharpe)</li>
                  <li>• Benchmark comparison</li>
                </ul>
              </div>
              
              <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                <Settings className="w-8 h-8 text-purple-600 mx-auto mb-4" />
                <h3 className="text-lg font-bold text-gray-800 mb-3">System Features</h3>
                <ul className="text-gray-600 text-sm space-y-2 text-left">
                  <li>• FastAPI backend integration</li>
                  <li>• Real-time pipeline monitoring</li>
                  <li>• Comprehensive result export</li>
                  <li>• Interactive visualizations</li>
                </ul>
              </div>
            </div>
            
            {!isConnected && (
              <div className="max-w-md mx-auto mb-8 p-6 bg-red-50 border border-red-200 rounded-xl">
                <AlertTriangle className="w-8 h-8 text-red-600 mx-auto mb-3" />
                <h3 className="text-red-800 font-bold mb-2">Backend Required</h3>
                <p className="text-red-600 text-sm">
                  Start the FastAPI server on localhost:8000 to begin quantitative analysis
                </p>
              </div>
            )}
            
            <div className="max-w-2xl mx-auto">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Expected Performance Targets</h3>
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-4">
                  <p className="text-2xl font-bold text-blue-600">14.2%</p>
                  <p className="text-gray-600 text-sm">Annual Return</p>
                </div>
                <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-4">
                  <p className="text-2xl font-bold text-green-600">1.58</p>
                  <p className="text-gray-600 text-sm">Sharpe Ratio</p>
                </div>
                <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-4">
                  <p className="text-2xl font-bold text-purple-600">5.8%</p>
                  <p className="text-gray-600 text-sm">Outperformance</p>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Enhanced Navigation */}
            <div className="flex flex-wrap gap-2 mb-8 justify-center">
              {[
                { id: 'overview', label: 'System Overview', icon: Activity, description: 'Key metrics & performance' },
                { id: 'performance', label: 'Performance Analysis', icon: TrendingUp, description: 'Detailed backtesting results' },
                { id: 'factors', label: 'Alpha Factors', icon: Target, description: 'Feature importance & factors' },
                { id: 'model', label: 'ML Architecture', icon: Brain, description: 'Model details & validation' },
                { id: 'trades', label: 'Trade Analysis', icon: DollarSign, description: 'Individual trade breakdown' }
              ].map(({ id, label, icon: Icon, description }) => (
                <button
                  key={id}
                  onClick={() => setCurrentView(id)}
                  className={`px-6 py-4 font-semibold flex flex-col items-center transition-all duration-200 rounded-xl ${
                    currentView === id
                      ? 'bg-blue-600 text-white shadow-xl scale-105'
                      : 'bg-white/50 backdrop-blur-sm border border-white/20 text-gray-700 hover:shadow-lg hover:scale-102'
                  }`}
                  title={description}
                >
                  <Icon className="w-5 h-5 mb-1" />
                  <span className="text-sm">{label}</span>
                </button>
              ))}
            </div>

            {/* System Overview */}
            {currentView === 'overview' && (
              <div className="space-y-8">
                {/* Enhanced Key Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <MetricCard
                    title="Total Return"
                    value={`${(data.metrics.total_return || 0).toFixed(1)}%`}
                    subtitle={`Outperformance: +${((data.metrics.outperformance || 0)).toFixed(1)}%`}
                    icon={TrendingUp}
                    color="blue"
                    trend={data.metrics.total_return > 0 ? 'up' : 'down'}
                  />
                  <MetricCard
                    title="Sharpe Ratio"
                    value={(data.metrics.sharpe_ratio || 0).toFixed(2)}
                    subtitle="Risk-adjusted returns"
                    icon={Target}
                    color="green"
                    trend={data.metrics.sharpe_ratio > 1 ? 'up' : 'down'}
                  />
                  <MetricCard
                    title="Max Drawdown"
                    value={`${Math.abs(data.metrics.max_drawdown || 0).toFixed(1)}%`}
                    subtitle="Maximum portfolio decline"
                    icon={AlertTriangle}
                    color="red"
                    trend="down"
                  />
                  <MetricCard
                    title="Win Rate"
                    value={`${(data.metrics.win_rate || 0).toFixed(1)}%`}
                    subtitle={`${data.metrics.total_trades || 0} total trades`}
                    icon={DollarSign}
                    color="purple"
                    trend={data.metrics.win_rate > 50 ? 'up' : 'down'}
                  />
                </div>

                {/* Performance Chart */}
                {data.backtestResults.length > 0 && (
                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
                    <div className="flex items-center justify-between mb-6">
                      <h3 className="text-2xl font-bold text-gray-800">Portfolio Performance Evolution</h3>
                      <div className="flex items-center space-x-4 text-sm">
                        <div className="flex items-center">
                          <div className="w-4 h-4 bg-blue-500 rounded mr-2"></div>
                          <span className="text-gray-600">Alpha Strategy</span>
                        </div>
                        <div className="flex items-center">
                          <div className="w-4 h-4 bg-red-500 rounded mr-2 opacity-60"></div>
                          <span className="text-gray-600">S&P 500 Benchmark</span>
                        </div>
                      </div>
                    </div>
                    <ResponsiveContainer width="100%" height={450}>
                      <LineChart data={formatChartData(data.backtestResults)}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="date" 
                          tickFormatter={(value) => {
                            try {
                              return new Date(value).toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
                            } catch {
                              return value;
                            }
                          }}
                          stroke="#666"
                        />
                        <YAxis 
                          tickFormatter={(value) => `${value.toFixed(1)}%`} 
                          stroke="#666"
                        />
                        <Tooltip 
                          formatter={(value, name) => [
                            `${(value).toFixed(2)}%`,
                            name === 'cumulative_return' ? 'Alpha Strategy' : 'S&P 500 Benchmark'
                          ]}
                          labelFormatter={(label) => `Date: ${new Date(label).toLocaleDateString()}`}
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px',
                            boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
                          }}
                        />
                        <Legend />
                        <Line 
                          type="monotone" 
                          dataKey="cumulative_return" 
                          stroke="#3b82f6" 
                          strokeWidth={3} 
                          dot={false} 
                          name="Alpha Strategy" 
                        />
                        <Line 
                          type="monotone" 
                          dataKey="benchmark_return" 
                          stroke="#ef4444" 
                          strokeWidth={2} 
                          dot={false} 
                          strokeDasharray="5,5" 
                          name="S&P 500 Benchmark"
                          opacity={0.7}
                        />
                                              </LineChart>
                    </ResponsiveContainer>
                    
                    {/* Performance Summary */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-gray-200">
                      <div className="text-center">
                        <p className="text-2xl font-bold text-blue-600">
                          ${(data.metrics.final_value || 100000).toLocaleString()}
                        </p>
                        <p className="text-sm text-gray-600">Final Portfolio Value</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-600">
                          {((data.metrics.total_return || 0)).toFixed(1)}%
                        </p>
                        <p className="text-sm text-gray-600">Total Return</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-purple-600">
                          {(data.metrics.volatility || 0).toFixed(1)}%
                        </p>
                        <p className="text-sm text-gray-600">Annualized Volatility</p>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-orange-600">
                          {(data.metrics.information_ratio || 0).toFixed(2)}
                        </p>
                        <p className="text-sm text-gray-600">Information Ratio</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* System Architecture Status */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                      <Server className="w-6 h-6 mr-2 text-blue-600" />
                      Pipeline Status
                    </h3>
                    <div className="space-y-3">
                      {[
                        { component: 'Data Generation', status: data.marketData.length > 0 ? 'active' : 'inactive', count: data.marketData.length },
                        { component: 'Alpha Factors', status: data.alphaFactors.length > 0 ? 'active' : 'inactive', count: data.alphaFactors.length },
                        { component: 'ML Models', status: data.modelResults ? 'active' : 'inactive', count: data.modelResults ? 1 : 0 },
                        { component: 'Backtesting', status: data.backtestResults.length > 0 ? 'active' : 'inactive', count: data.backtestResults.length }
                      ].map(({ component, status, count }) => (
                        <div key={component} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                          <div className="flex items-center">
                            <div className={`w-3 h-3 rounded-full mr-3 ${
                              status === 'active' ? 'bg-green-500 shadow-lg' : 'bg-gray-400'
                            }`}></div>
                            <span className="font-medium text-gray-700">{component}</span>
                          </div>
                          <span className={`text-sm font-semibold ${
                            status === 'active' ? 'text-green-600' : 'text-gray-500'
                          }`}>
                            {count.toLocaleString()} {count === 1 ? 'item' : 'items'}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                    <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
                      <BarChart3 className="w-6 h-6 mr-2 text-green-600" />
                      Performance vs Targets
                    </h3>
                    <div className="space-y-4">
                      {[
                        { 
                          metric: 'Annual Return', 
                          actual: data.metrics.annualized_return || 0, 
                          target: 14.2,
                          unit: '%'
                        },
                        { 
                          metric: 'Sharpe Ratio', 
                          actual: data.metrics.sharpe_ratio || 0, 
                          target: 1.58,
                          unit: ''
                        },
                        { 
                          metric: 'Outperformance', 
                          actual: data.metrics.outperformance || 0, 
                          target: 5.8,
                          unit: '%'
                        }
                      ].map(({ metric, actual, target, unit }) => {
                        const achievement = target > 0 ? (actual / target) * 100 : 0;
                        const isGood = achievement >= 80;
                        
                        return (
                          <div key={metric} className="space-y-2">
                            <div className="flex justify-between items-center">
                              <span className="font-medium text-gray-700">{metric}</span>
                              <span className={`font-bold ${isGood ? 'text-green-600' : 'text-red-600'}`}>
                                {actual.toFixed(actual < 10 ? 2 : 1)}{unit} / {target.toFixed(1)}{unit}
                              </span>
                            </div>
                            <div className="bg-gray-200 rounded-full h-2">
                              <div 
                                className={`h-2 rounded-full ${isGood ? 'bg-green-500' : 'bg-red-500'}`}
                                style={{ width: `${Math.min(achievement, 100)}%` }}
                              ></div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Performance Analysis View */}
            {currentView === 'performance' && (
              <div className="space-y-8">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <PerformanceMetrics metrics={data.metrics} />
                  <TradingStats metrics={data.metrics} tradeAnalysis={data.tradeAnalysis} />
                </div>
                
                {/* Risk Analysis */}
                <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
                  <h3 className="text-2xl font-bold text-gray-800 mb-6">Risk Analysis</h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="text-center p-4 bg-red-50 border border-red-200 rounded-lg">
                      <TrendingDown className="w-8 h-8 text-red-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-red-600">
                        {Math.abs(data.metrics.max_drawdown || 0).toFixed(1)}%
                      </p>
                      <p className="text-red-700 font-medium">Max Drawdown</p>
                      <p className="text-red-600 text-sm">Peak-to-trough decline</p>
                    </div>
                    
                    <div className="text-center p-4 bg-orange-50 border border-orange-200 rounded-lg">
                      <Activity className="w-8 h-8 text-orange-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-orange-600">
                        {(data.metrics.volatility || 0).toFixed(1)}%
                      </p>
                      <p className="text-orange-700 font-medium">Volatility</p>
                      <p className="text-orange-600 text-sm">Annualized standard deviation</p>
                    </div>
                    
                    <div className="text-center p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <Shield className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-blue-600">
                        {((data.metrics.volatility || 0) !== 0 ? (data.metrics.annualized_return || 0) / (data.metrics.volatility || 1) : 0).toFixed(2)}
                      </p>
                      <p className="text-blue-700 font-medium">Risk-Return Ratio</p>
                      <p className="text-blue-600 text-sm">Return per unit of risk</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Alpha Factors View */}
            {currentView === 'factors' && (
              <div className="space-y-8">
                {data.modelResults?.featureImportance && Object.keys(data.modelResults.featureImportance).length > 0 && (
                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
                    <h3 className="text-2xl font-bold text-gray-800 mb-6">Alpha Factor Importance</h3>
                    <p className="text-gray-600 mb-6">
                      Feature importance scores from the ensemble ML model, showing which alpha factors contribute most to predictive power.
                    </p>
                    <ResponsiveContainer width="100%" height={500}>
                      <BarChart data={
                        Object.entries(data.modelResults.featureImportance)
                          .sort(([,a], [,b]) => b - a)
                          .slice(0, 15)
                          .map(([key, value]) => ({
                            factor: key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                            importance: value * 100
                          }))
                      }>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="factor" 
                          angle={-45} 
                          textAnchor="end" 
                          height={120}
                          stroke="#666"
                        />
                        <YAxis 
                          tickFormatter={(value) => `${value.toFixed(1)}%`}
                          stroke="#666"
                        />
                        <Tooltip 
                          formatter={(value) => [`${value.toFixed(2)}%`, 'Importance']}
                          contentStyle={{
                            backgroundColor: 'rgba(255, 255, 255, 0.95)',
                            border: '1px solid #e2e8f0',
                            borderRadius: '8px'
                          }}
                        />
                        <Bar dataKey="importance" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
                
                {/* Factor Categories */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                    <h4 className="text-lg font-bold text-blue-800 mb-4 flex items-center">
                      <TrendingUp className="w-5 h-5 mr-2" />
                      Momentum Factors
                    </h4>
                    <ul className="text-blue-600 space-y-2 text-sm">
                      <li>• 5-day, 20-day, 60-day momentum</li>
                      <li>• Price relative to moving averages</li>
                      <li>• Momentum acceleration signals</li>
                      <li>• Cross-over indicators</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                    <h4 className="text-lg font-bold text-green-800 mb-4 flex items-center">
                      <Activity className="w-5 h-5 mr-2" />
                      Technical Factors
                    </h4>
                    <ul className="text-green-600 space-y-2 text-sm">
                      <li>• RSI (Relative Strength Index)</li>
                      <li>• Moving average convergence</li>
                      <li>• Bollinger Band positions</li>
                      <li>• Support/resistance levels</li>
                    </ul>
                  </div>
                  
                  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-6">
                    <h4 className="text-lg font-bold text-purple-800 mb-4 flex items-center">
                      <BarChart3 className="w-5 h-5 mr-2" />
                      Volatility Factors
                    </h4>
                    <ul className="text-purple-600 space-y-2 text-sm">
                      <li>• 20-day realized volatility</li>
                      <li>• Volume-price relationships</li>
                      <li>• Volatility regime changes</li>
                      <li>• Risk-adjusted momentum</li>
                    </ul>
                  </div>
                </div>
              </div>
            )}

            {/* ML Model View */}
            {currentView === 'model' && (
              <div className="space-y-8">
                <ModelOverview 
                  alphaFactors={data.alphaFactors} 
                  modelResults={data.modelResults} 
                  isConnected={isConnected}
                  metrics={data.metrics}
                />
                
                {/* Model Architecture */}
                <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
                  <h3 className="text-2xl font-bold text-gray-800 mb-6">Ensemble Architecture</h3>
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="space-y-4">
                      <h4 className="text-lg font-semibold text-gray-800">Primary Model: XGBoost</h4>
                      <ul className="text-gray-600 space-y-2">
                        <li>• Gradient boosting decision trees</li>
                        <li>• Optimized for time-series data</li>
                        <li>• Handles feature interactions naturally</li>
                        <li>• Built-in regularization</li>
                      </ul>
                      
                      <h4 className="text-lg font-semibold text-gray-800 mt-6">Secondary Model: Random Forest</h4>
                      <ul className="text-gray-600 space-y-2">
                        <li>• Ensemble of decision trees</li>
                        <li>• Reduces overfitting risk</li>
                        <li>• Provides model stability</li>
                        <li>• Feature importance scoring</li>
                      </ul>
                    </div>
                    
                    <div className="space-y-4">
                      <h4 className="text-lg font-semibold text-gray-800">Validation Strategy</h4>
                      <ul className="text-gray-600 space-y-2">
                        <li>• Walk-forward validation</li>
                        <li>• Time-series cross-validation</li>
                        <li>• Out-of-sample testing</li>
                        <li>• Information coefficient tracking</li>
                      </ul>
                      
                      <h4 className="text-lg font-semibold text-gray-800 mt-6">Feature Engineering</h4>
                      <ul className="text-gray-600 space-y-2">
                        <li>• Rolling window calculations</li>
                        <li>• Technical indicator normalization</li>
                        <li>• Cross-sectional ranking</li>
                        <li>• Missing value handling</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Trade Analysis View */}
            {currentView === 'trades' && (
              <div className="space-y-8">
                <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
                  <h3 className="text-2xl font-bold text-gray-800 mb-6">Trading Performance Breakdown</h3>
                  
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                    <div className="text-center p-4 bg-blue-50 border border-blue-200 rounded-lg">
                      <DollarSign className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-blue-600">{data.metrics.total_trades || 0}</p>
                      <p className="text-blue-700 font-medium">Total Trades</p>
                    </div>
                    
                    <div className="text-center p-4 bg-green-50 border border-green-200 rounded-lg">
                      <CheckCircle className="w-8 h-8 text-green-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-green-600">{data.metrics.winning_trades || 0}</p>
                      <p className="text-green-700 font-medium">Winning Trades</p>
                    </div>
                    
                    <div className="text-center p-4 bg-red-50 border border-red-200 rounded-lg">
                      <X className="w-8 h-8 text-red-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-red-600">{(data.metrics.total_trades || 0) - (data.metrics.winning_trades || 0)}</p>
                      <p className="text-red-700 font-medium">Losing Trades</p>
                    </div>
                    
                    <div className="text-center p-4 bg-purple-50 border border-purple-200 rounded-lg">
                      <Target className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                      <p className="text-2xl font-bold text-purple-600">{(data.metrics.profit_factor || 0).toFixed(2)}</p>
                      <p className="text-purple-700 font-medium">Profit Factor</p>
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-6">
                    <h4 className="text-lg font-semibold text-gray-800 mb-4">Trading Strategy Details</h4>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <div>
                        <h5 className="font-medium text-gray-700 mb-2">Signal Generation</h5>
                        <ul className="text-gray-600 text-sm space-y-1">
                          <li>• ML model predictions above threshold trigger trades</li>
                          <li>• Dynamic position sizing based on signal strength</li>
                          <li>• Risk management through position limits</li>
                        </ul>
                      </div>
                      <div>
                        <h5 className="font-medium text-gray-700 mb-2">Execution Details</h5>
                        <ul className="text-gray-600 text-sm space-y-1">
                          <li>• 0.1% transaction cost per trade</li>
                          <li>• Next-day execution after signal</li>
                          <li>• Portfolio rebalancing on signal changes</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

// Helper Components
const MetricCard = ({ title, value, subtitle, icon: Icon, color, trend }) => {
  const colorClasses = {
    blue: 'bg-gradient-to-br from-blue-500 to-blue-600',
    green: 'bg-gradient-to-br from-green-500 to-green-600',
    red: 'bg-gradient-to-br from-red-500 to-red-600',
    purple: 'bg-gradient-to-br from-purple-500 to-purple-600'
  };

  return (
    <div className={`${colorClasses[color]} text-white rounded-2xl p-6 shadow-lg hover:shadow-xl transition-all duration-300`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium opacity-90">{title}</h3>
        <div className="flex items-center">
          <Icon className="w-6 h-6 opacity-80 mr-2" />
          {trend === 'up' && <TrendingUp className="w-4 h-4 opacity-60" />}
          {trend === 'down' && <TrendingDown className="w-4 h-4 opacity-60" />}
        </div>
      </div>
      <p className="text-3xl font-bold mb-2">{value}</p>
      <p className="text-xs opacity-80">{subtitle}</p>
    </div>
  );
};

const PerformanceMetrics = ({ metrics }) => (
  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
    <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
      <BarChart3 className="w-6 h-6 mr-2 text-blue-600" />
      Performance Metrics
    </h3>
    <div className="space-y-4">
      {[
        { 
          label: 'Annualized Return', 
          value: `${(metrics.annualized_return || 0).toFixed(1)}%`, 
          color: 'blue',
          description: 'Compound annual growth rate'
        },
        { 
          label: 'Volatility', 
          value: `${(metrics.volatility || 0).toFixed(1)}%`, 
          color: 'orange',
          description: 'Standard deviation of returns'
        },
        { 
          label: 'Information Ratio', 
          value: (metrics.information_ratio || 0).toFixed(2), 
          color: 'purple',
          description: 'Excess return per unit of risk'
        },
        { 
          label: 'Calmar Ratio', 
          value: (metrics.calmar_ratio || 0).toFixed(2), 
          color: 'green',
          description: 'Annual return / max drawdown'
        }
      ].map(({ label, value, color, description }) => (
        <div key={label} className={`flex justify-between items-center p-4 bg-${color}-50 rounded-lg border border-${color}-200 hover:shadow-md transition-all`}>
          <div>
            <span className="font-semibold text-gray-700">{label}</span>
            <p className="text-xs text-gray-500">{description}</p>
          </div>
          <span className={`font-bold text-${color}-600 text-lg`}>{value}</span>
        </div>
      ))}
    </div>
  </div>
);

const TradingStats = ({ metrics, tradeAnalysis }) => (
  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
    <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
      <DollarSign className="w-6 h-6 mr-2 text-green-600" />
      Trading Statistics
    </h3>
    <div className="space-y-4">
      {[
        { 
          label: 'Total Trades', 
          value: metrics.total_trades || 0, 
          color: 'blue',
          description: 'Number of position changes'
        },
        { 
          label: 'Winning Trades', 
          value: metrics.winning_trades || 0, 
          color: 'green',
          description: 'Profitable position changes'
        },
        { 
          label: 'Profit Factor', 
          value: (metrics.profit_factor || 0).toFixed(2), 
          color: 'purple',
          description: 'Gross profit / gross loss'
        },
        { 
          label: 'Final Portfolio Value', 
          value: `${(metrics.final_value || 0).toLocaleString()}`, 
          color: 'gray',
          description: 'End-of-period portfolio value'
        }
      ].map(({ label, value, color, description }) => (
        <div key={label} className={`flex justify-between items-center p-4 bg-${color}-50 rounded-lg border border-${color}-200 hover:shadow-md transition-all`}>
          <div>
            <span className="font-semibold text-gray-700">{label}</span>
            <p className="text-xs text-gray-500">{description}</p>
          </div>
          <span className={`font-bold text-${color}-600 text-lg`}>{value}</span>
        </div>
      ))}
    </div>
  </div>
);

const ModelOverview = ({ alphaFactors, isConnected }) => (
  <div className="bg-white/50 backdrop-blur-sm border border-white/20 rounded-lg p-8">
    <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center">
      <Brain className="w-6 h-6 mr-2 text-purple-600" />
      Machine Learning Model Overview
    </h3>
    <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
      <div className="space-y-6">
        <div className="p-6 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="font-bold text-blue-800 mb-2 flex items-center">
            <Cpu className="w-5 h-5 mr-2" />
            Model Architecture
          </h4>
          <p className="text-blue-600 text-sm mb-2">XGBoost + Random Forest Ensemble</p>
          <ul className="text-blue-600 text-xs space-y-1">
            <li>• Gradient boosting with decision trees</li>
            <li>• Ensemble learning for stability</li>
            <li>• Time-series aware validation</li>
          </ul>
        </div>
        
        <div className="p-6 bg-green-50 rounded-lg border border-green-200">
          <h4 className="font-bold text-green-800 mb-2 flex items-center">
            <Target className="w-5 h-5 mr-2" />
            Feature Engineering
          </h4>
          <p className="text-green-600 text-sm mb-2">25+ Quantitative Alpha Factors</p>
          <ul className="text-green-600 text-xs space-y-1">
            <li>• Technical indicators (RSI, MACD, etc.)</li>
            <li>• Momentum signals (5d, 20d, 60d)</li>
            <li>• Volatility measures and ratios</li>
          </ul>
        </div>
      </div>
      
      <div className="space-y-6">
        <div className="p-6 bg-purple-50 rounded-lg border border-purple-200">
          <h4 className="font-bold text-purple-800 mb-2 flex items-center">
            <Database className="w-5 h-5 mr-2" />
            Training Data
          </h4>
          <p className="text-purple-600 text-sm mb-2">{alphaFactors.length.toLocaleString()} training samples</p>
          <ul className="text-purple-600 text-xs space-y-1">
            <li>• 10+ years of S&P 500 data</li>
            <li>• Daily frequency observations</li>
            <li>• Forward-looking return targets</li>
          </ul>
        </div>
        
        <div className={`p-6 rounded-lg border ${isConnected ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <h4 className={`font-bold mb-2 flex items-center ${isConnected ? 'text-green-800' : 'text-red-800'}`}>
            <Server className="w-5 h-5 mr-2" />
            System Status
          </h4>
          <p className={`text-sm mb-2 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            {isConnected ? 'FastAPI Backend Connected' : 'Backend Disconnected'}
          </p>
          <ul className={`text-xs space-y-1 ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
            <li>• Real-time pipeline execution</li>
            <li>• Interactive result visualization</li>
            <li>• Comprehensive data export</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
);

export default App;