import { useState, useEffect } from 'react';
import { Play, Clock, CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { trainModel, getTrainingStatus } from '../services/api';

function Training() {
  const [selectedDataset, setSelectedDataset] = useState('kepler');
  const [trainingStatus, setTrainingStatus] = useState(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadTrainingStatus();
    const interval = setInterval(loadTrainingStatus, 3000);
    return () => clearInterval(interval);
  }, []);

  const loadTrainingStatus = async () => {
    try {
      const status = await getTrainingStatus();
      setTrainingStatus(status);
      setIsTraining(status.status === 'training');
    } catch (err) {
      console.error('Error loading training status:', err);
    }
  };

  const handleStartTraining = async () => {
    setError(null);
    setIsTraining(true);

    try {
      await trainModel(selectedDataset);
      loadTrainingStatus();
    } catch (err) {
      setError(err.response?.data?.detail || 'Training failed');
      setIsTraining(false);
    }
  };

  const getStatusIcon = () => {
    if (!trainingStatus) return <Clock className="w-6 h-6 text-gray-400" />;

    switch (trainingStatus.status) {
      case 'training':
        return <RefreshCw className="w-6 h-6 text-primary-500 animate-spin" />;
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'failed':
        return <AlertCircle className="w-6 h-6 text-red-500" />;
      default:
        return <Clock className="w-6 h-6 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    if (!trainingStatus) return 'text-gray-400';

    switch (trainingStatus.status) {
      case 'training':
        return 'text-primary-500';
      case 'completed':
        return 'text-green-500';
      case 'failed':
        return 'text-red-500';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold gradient-text mb-2">Model Training</h1>
        <p className="text-gray-400">Train new models on exoplanet datasets</p>
      </div>

      {/* Training Status */}
      {trainingStatus && (
        <div className="card border-2 border-primary-500/30">
          <div className="flex items-start space-x-4">
            <div className="mt-1">{getStatusIcon()}</div>
            <div className="flex-1">
              <h2 className="text-xl font-bold mb-2">Training Status</h2>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Status:</span>
                  <span className={`font-semibold ${getStatusColor()}`}>
                    {trainingStatus.status.toUpperCase()}
                  </span>
                </div>

                {trainingStatus.message && (
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Message:</span>
                    <span className="font-medium">{trainingStatus.message}</span>
                  </div>
                )}

                {trainingStatus.progress !== undefined && (
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-400">Progress:</span>
                      <span className="font-semibold">{trainingStatus.progress}%</span>
                    </div>
                    <div className="bg-dark-700 rounded-full h-3">
                      <div
                        className="bg-gradient-to-r from-primary-500 to-purple-500 h-3 rounded-full transition-all duration-500"
                        style={{ width: `${trainingStatus.progress}%` }}
                      />
                    </div>
                  </div>
                )}

                {trainingStatus.results && (
                  <div className="mt-4 pt-4 border-t border-dark-700">
                    <h3 className="font-semibold mb-3">Training Results</h3>
                    {trainingStatus.results.best_model && (
                      <div className="bg-dark-700 rounded-lg p-4">
                        <p className="text-sm text-gray-400">Best Model</p>
                        <p className="text-lg font-bold text-primary-400">
                          {trainingStatus.results.best_model}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="bg-red-500/10 border border-red-500 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Training Configuration */}
      <div className="card">
        <h2 className="text-xl font-bold mb-6">Start New Training</h2>

        <div className="space-y-6">
          {/* Dataset Selection */}
          <div>
            <label className="label">Select Dataset</label>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {[
                {
                  id: 'kepler',
                  name: 'Kepler',
                  description: 'Objects of Interest from Kepler mission',
                  samples: '~9,500',
                },
                {
                  id: 'tess',
                  name: 'TESS',
                  description: 'Objects of Interest from TESS mission',
                  samples: '~7,800',
                },
                {
                  id: 'k2',
                  name: 'K2',
                  description: 'Planets and Candidates from K2 mission',
                  samples: '~4,300',
                },
              ].map((dataset) => (
                <button
                  key={dataset.id}
                  onClick={() => setSelectedDataset(dataset.id)}
                  className={`p-4 rounded-lg border-2 text-left transition-colors ${
                    selectedDataset === dataset.id
                      ? 'border-primary-500 bg-primary-500/10'
                      : 'border-dark-600 bg-dark-700 hover:border-dark-500'
                  }`}
                >
                  <h3 className="font-bold text-lg mb-1">{dataset.name}</h3>
                  <p className="text-sm text-gray-400 mb-2">{dataset.description}</p>
                  <p className="text-xs text-gray-500">{dataset.samples} samples</p>
                </button>
              ))}
            </div>
          </div>

          {/* Models Info */}
          <div className="bg-dark-700 rounded-lg p-4">
            <h3 className="font-semibold mb-3">Models to be Trained</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              {['Random Forest', 'XGBoost', 'LightGBM', 'Neural Network', 'Gradient Boosting'].map(
                (model) => (
                  <div key={model} className="bg-dark-600 rounded px-3 py-2 text-sm text-center">
                    {model}
                  </div>
                )
              )}
            </div>
          </div>

          {/* Training Steps */}
          <div className="bg-dark-700 rounded-lg p-4">
            <h3 className="font-semibold mb-3">Training Pipeline</h3>
            <ol className="space-y-2 text-sm text-gray-400">
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">1.</span>
                <span>Load and clean dataset</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">2.</span>
                <span>Feature engineering and selection</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">3.</span>
                <span>Split data (70% train, 10% validation, 20% test)</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">4.</span>
                <span>Train multiple models in parallel</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">5.</span>
                <span>Evaluate and compare models</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary-500 mr-2">6.</span>
                <span>Save best performing model</span>
              </li>
            </ol>
          </div>

          {/* Start Button */}
          <button
            onClick={handleStartTraining}
            disabled={isTraining}
            className="btn-primary w-full flex items-center justify-center space-x-2"
          >
            {isTraining ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                <span>Training in Progress...</span>
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                <span>Start Training on {selectedDataset.toUpperCase()}</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Training Info */}
      <div className="card">
        <h2 className="text-xl font-bold mb-4">About Model Training</h2>
        <div className="space-y-4 text-gray-300">
          <p>
            The training process will build multiple machine learning models on the selected dataset
            to identify exoplanets. The system will:
          </p>
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li>Automatically handle missing values and outliers</li>
            <li>Select the most important features using correlation analysis</li>
            <li>Train 5 different model types for comparison</li>
            <li>Evaluate models using accuracy, precision, recall, and F1 score</li>
            <li>Save the best performing model for predictions</li>
          </ul>
          <div className="bg-primary-500/10 border border-primary-500/30 rounded-lg p-4 mt-4">
            <p className="text-sm">
              <strong className="text-primary-400">Note:</strong> Training may take 5-15 minutes
              depending on the dataset size and your hardware. You can monitor progress in real-time
              on this page.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Training;
