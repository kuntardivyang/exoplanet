import { useState, useEffect } from 'react';
import { Settings, Download, CheckCircle, TrendingUp } from 'lucide-react';
import { listModels, getModelInfo, loadModel, getFeatureImportance } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

function Models() {
  const [models, setModels] = useState([]);
  const [currentModel, setCurrentModel] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadModels();
    loadFeatureImportance();
  }, []);

  const loadModels = async () => {
    try {
      const data = await listModels();
      setModels(data.models);
      setCurrentModel(data.current_model);
    } catch (error) {
      console.error('Error loading models:', error);
    }
  };

  const loadFeatureImportance = async () => {
    try {
      const data = await getFeatureImportance(15);
      setFeatureImportance(data);
    } catch (error) {
      console.log('Feature importance not available');
    }
  };

  const handleSelectModel = async (modelName) => {
    setSelectedModel(modelName);
    setLoading(true);

    try {
      const info = await getModelInfo(modelName);
      setModelInfo(info);
    } catch (error) {
      console.error('Error loading model info:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleLoadModel = async (modelName) => {
    setLoading(true);
    try {
      await loadModel(modelName);
      loadModels();
      alert(`Model ${modelName} loaded successfully!`);
    } catch (error) {
      alert('Error loading model: ' + error.message);
    } finally {
      setLoading(false);
    }
  };

  const getFeatureImportanceData = () => {
    if (!featureImportance?.feature_importance) return [];

    return Object.entries(featureImportance.feature_importance).map(([name, value]) => ({
      feature: name.length > 15 ? name.substring(0, 15) + '...' : name,
      importance: (value * 100).toFixed(2),
    }));
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold gradient-text mb-2">Model Management</h1>
        <p className="text-gray-400">Manage and compare trained models</p>
      </div>

      {/* Current Model */}
      {currentModel && (
        <div className="card border-2 border-primary-500">
          <div className="flex items-start space-x-4">
            <div className="p-3 bg-primary-500/20 rounded-lg">
              <CheckCircle className="w-8 h-8 text-primary-500" />
            </div>
            <div>
              <h2 className="text-xl font-bold mb-1">Currently Active Model</h2>
              <p className="text-2xl font-bold text-primary-400">{currentModel}</p>
              <p className="text-sm text-gray-400 mt-1">This model is being used for predictions</p>
            </div>
          </div>
        </div>
      )}

      {/* Available Models */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4 flex items-center">
          <Settings className="w-6 h-6 mr-2 text-primary-500" />
          Available Models ({models.length})
        </h2>

        {models.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-400 mb-4">No trained models available</p>
            <p className="text-sm text-gray-500">
              Go to the Training page to train your first model
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((model) => (
              <div
                key={model}
                className={`bg-dark-700 rounded-lg p-4 border-2 transition-colors cursor-pointer ${
                  selectedModel === model
                    ? 'border-primary-500'
                    : model === currentModel
                    ? 'border-green-500/50'
                    : 'border-dark-600 hover:border-dark-500'
                }`}
                onClick={() => handleSelectModel(model)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-sm">{model}</h3>
                  {model === currentModel && (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  )}
                </div>
                <p className="text-xs text-gray-400 mb-3">
                  {model.split('_')[0].toUpperCase()} model
                </p>
                {model !== currentModel && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLoadModel(model);
                    }}
                    className="btn-secondary w-full text-xs py-1"
                  >
                    Load Model
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Model Details */}
      {modelInfo && selectedModel && (
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Model Details: {selectedModel}</h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-dark-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Model Type</p>
              <p className="text-lg font-bold">{modelInfo.model_type.toUpperCase()}</p>
            </div>
            <div className="bg-dark-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Features</p>
              <p className="text-lg font-bold">{modelInfo.features_count}</p>
            </div>
            {modelInfo.trained_date && (
              <div className="bg-dark-700 rounded-lg p-4">
                <p className="text-sm text-gray-400 mb-1">Trained</p>
                <p className="text-sm font-bold">{modelInfo.trained_date}</p>
              </div>
            )}
            {modelInfo.classes && (
              <div className="bg-dark-700 rounded-lg p-4">
                <p className="text-sm text-gray-400 mb-1">Classes</p>
                <p className="text-lg font-bold">{modelInfo.classes.length}</p>
              </div>
            )}
          </div>

          {modelInfo.metrics && (
            <div>
              <h3 className="font-semibold mb-3">Performance Metrics</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(modelInfo.metrics).map(([key, value]) => (
                  <div key={key} className="bg-dark-700 rounded-lg p-4">
                    <p className="text-sm text-gray-400 mb-1">
                      {key.replace('_', ' ').toUpperCase()}
                    </p>
                    <p className="text-2xl font-bold text-primary-400">
                      {(value * 100).toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {modelInfo.classes && (
            <div className="mt-6">
              <h3 className="font-semibold mb-3">Target Classes</h3>
              <div className="flex flex-wrap gap-2">
                {modelInfo.classes.map((cls, index) => (
                  <span
                    key={index}
                    className="px-4 py-2 bg-primary-500/20 text-primary-400 rounded-full text-sm"
                  >
                    {cls}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Feature Importance */}
      {featureImportance && (
        <div className="card">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-primary-500" />
            Feature Importance (Top {featureImportance.top_n})
          </h2>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={getFeatureImportanceData()}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="feature" stroke="#94a3b8" angle={-45} textAnchor="end" height={100} />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                }}
              />
              <Bar dataKey="importance" fill="#0ea5e9" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Model Info */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">About Models</h2>
        <div className="space-y-4 text-gray-300">
          <p>
            This system trains multiple machine learning models to identify exoplanets from various
            NASA datasets. Each model type has its own strengths:
          </p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-dark-700 rounded-lg p-4">
              <h3 className="font-semibold text-primary-400 mb-2">Random Forest</h3>
              <p className="text-sm">
                Ensemble method using multiple decision trees. Robust and handles non-linear
                relationships well.
              </p>
            </div>
            <div className="bg-dark-700 rounded-lg p-4">
              <h3 className="font-semibold text-primary-400 mb-2">XGBoost</h3>
              <p className="text-sm">
                Gradient boosting algorithm. Often achieves state-of-the-art results with proper
                tuning.
              </p>
            </div>
            <div className="bg-dark-700 rounded-lg p-4">
              <h3 className="font-semibold text-primary-400 mb-2">LightGBM</h3>
              <p className="text-sm">
                Fast gradient boosting framework. Efficient for large datasets with many features.
              </p>
            </div>
            <div className="bg-dark-700 rounded-lg p-4">
              <h3 className="font-semibold text-primary-400 mb-2">Neural Network</h3>
              <p className="text-sm">
                Deep learning model. Can capture complex patterns but requires more data to train.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Models;
