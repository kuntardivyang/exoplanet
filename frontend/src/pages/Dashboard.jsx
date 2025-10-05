import { useState, useEffect } from 'react';
import { TrendingUp, Target, Clock, Zap, Star, Globe } from 'lucide-react';
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { checkHealth, listDatasets, getFeatureImportance } from '../services/api';

const COLORS = ['#0ea5e9', '#8b5cf6', '#10b981', '#f59e0b'];

function Dashboard() {
  const [health, setHealth] = useState(null);
  const [datasets, setDatasets] = useState([]);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [healthData, datasetsData] = await Promise.all([
        checkHealth(),
        listDatasets(),
      ]);

      setHealth(healthData);
      setDatasets(datasetsData);

      // Try to load feature importance if model is loaded
      if (healthData.model_loaded) {
        try {
          const importanceData = await getFeatureImportance(15);
          setFeatureImportance(importanceData);
        } catch (error) {
          console.log('Feature importance not available');
        }
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  const stats = [
    {
      label: 'System Status',
      value: health?.status === 'healthy' ? 'Online' : 'Offline',
      icon: Zap,
      color: 'text-green-500',
      bgColor: 'bg-green-500/10',
    },
    {
      label: 'Model Status',
      value: health?.model_loaded ? 'Loaded' : 'Not Loaded',
      icon: Target,
      color: health?.model_loaded ? 'text-primary-500' : 'text-yellow-500',
      bgColor: health?.model_loaded ? 'bg-primary-500/10' : 'bg-yellow-500/10',
    },
    {
      label: 'Datasets',
      value: datasets.length,
      icon: Globe,
      color: 'text-purple-500',
      bgColor: 'bg-purple-500/10',
    },
    {
      label: 'Version',
      value: health?.version || '1.0.0',
      icon: Star,
      color: 'text-blue-500',
      bgColor: 'bg-blue-500/10',
    },
  ];

  const getDatasetChartData = () => {
    return datasets.map(ds => ({
      name: ds.name.toUpperCase(),
      samples: ds.rows,
      features: ds.columns,
    }));
  };

  const getTargetDistributionData = () => {
    const keplerDataset = datasets.find(ds => ds.name === 'kepler');
    if (!keplerDataset?.target_distribution) return [];

    return Object.entries(keplerDataset.target_distribution).map(([name, value]) => ({
      name,
      value,
    }));
  };

  const getFeatureImportanceData = () => {
    if (!featureImportance?.feature_importance) return [];

    return Object.entries(featureImportance.feature_importance)
      .slice(0, 10)
      .map(([name, value]) => ({
        feature: name.length > 15 ? name.substring(0, 15) + '...' : name,
        importance: (value * 100).toFixed(2),
      }));
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold gradient-text mb-2">Dashboard</h1>
        <p className="text-gray-400">Monitor your exoplanet detection system</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {stats.map((stat, index) => (
          <div key={index} className="metric-card">
            <div className="flex items-center justify-between mb-3">
              <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
            </div>
            <h3 className="text-gray-400 text-sm font-medium mb-1">{stat.label}</h3>
            <p className={`text-2xl font-bold ${stat.color}`}>{stat.value}</p>
          </div>
        ))}
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dataset Overview */}
        <div className="card">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Globe className="w-5 h-5 mr-2 text-primary-500" />
            Dataset Overview
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={getDatasetChartData()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                }}
              />
              <Legend />
              <Bar dataKey="samples" fill="#0ea5e9" name="Samples" />
              <Bar dataKey="features" fill="#8b5cf6" name="Features" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Target Distribution */}
        {getTargetDistributionData().length > 0 && (
          <div className="card">
            <h2 className="text-xl font-bold mb-4 flex items-center">
              <Target className="w-5 h-5 mr-2 text-primary-500" />
              Kepler Target Distribution
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={getTargetDistributionData()}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                  outerRadius={100}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {getTargetDistributionData().map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Feature Importance */}
        {getFeatureImportanceData().length > 0 && (
          <div className="card lg:col-span-2">
            <h2 className="text-xl font-bold mb-4 flex items-center">
              <TrendingUp className="w-5 h-5 mr-2 text-primary-500" />
              Top 10 Feature Importance
            </h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={getFeatureImportanceData()} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis type="number" stroke="#94a3b8" />
                <YAxis type="category" dataKey="feature" stroke="#94a3b8" width={100} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1e293b',
                    border: '1px solid #334155',
                    borderRadius: '8px',
                  }}
                />
                <Bar dataKey="importance" fill="#10b981" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <a
            href="/predict"
            className="flex items-center p-4 bg-dark-700 rounded-lg hover:bg-dark-600 transition-colors"
          >
            <Target className="w-8 h-8 text-primary-500 mr-3" />
            <div>
              <h3 className="font-semibold">Make Prediction</h3>
              <p className="text-sm text-gray-400">Classify exoplanet candidates</p>
            </div>
          </a>
          <a
            href="/training"
            className="flex items-center p-4 bg-dark-700 rounded-lg hover:bg-dark-600 transition-colors"
          >
            <Zap className="w-8 h-8 text-green-500 mr-3" />
            <div>
              <h3 className="font-semibold">Train Model</h3>
              <p className="text-sm text-gray-400">Start training new models</p>
            </div>
          </a>
          <a
            href="/data"
            className="flex items-center p-4 bg-dark-700 rounded-lg hover:bg-dark-600 transition-colors"
          >
            <Globe className="w-8 h-8 text-purple-500 mr-3" />
            <div>
              <h3 className="font-semibold">Explore Data</h3>
              <p className="text-sm text-gray-400">Analyze datasets</p>
            </div>
          </a>
        </div>
      </div>

      {/* Dataset Details */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">Available Datasets</h2>
        <div className="space-y-4">
          {datasets.map((dataset, index) => (
            <div key={index} className="bg-dark-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <h3 className="text-lg font-semibold text-primary-400">
                  {dataset.name.toUpperCase()}
                </h3>
                <span className="text-sm text-gray-400">{dataset.rows.toLocaleString()} samples</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Rows:</span>
                  <p className="font-semibold">{dataset.rows.toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-400">Columns:</span>
                  <p className="font-semibold">{dataset.columns}</p>
                </div>
                <div>
                  <span className="text-gray-400">Features:</span>
                  <p className="font-semibold">{dataset.features.length}</p>
                </div>
                {dataset.target_distribution && (
                  <div>
                    <span className="text-gray-400">Classes:</span>
                    <p className="font-semibold">
                      {Object.keys(dataset.target_distribution).length}
                    </p>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
