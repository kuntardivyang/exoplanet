import { useState } from 'react';
import { Upload, Sparkles, FileText, Plus, Trash2 } from 'lucide-react';
import { predictSingle, predictBatch, predictUpload } from '../services/api';

function Predict() {
  const [activeTab, setActiveTab] = useState('single');
  const [singleFeatures, setSingleFeatures] = useState({});
  const [batchFeatures, setBatchFeatures] = useState([{}]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const exampleFeatures = [
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad',
    'koi_teq', 'koi_steff', 'koi_slogg', 'koi_srad',
  ];

  const handleSinglePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictSingle(singleFeatures);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.detail || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleBatchPredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictBatch(batchFeatures);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.detail || 'Batch prediction failed');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploadedFile(file);
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const result = await predictUpload(file);
      setPrediction(result);
    } catch (err) {
      setError(err.response?.data?.detail || 'File upload failed');
    } finally {
      setLoading(false);
    }
  };

  const addBatchRow = () => {
    setBatchFeatures([...batchFeatures, {}]);
  };

  const removeBatchRow = (index) => {
    setBatchFeatures(batchFeatures.filter((_, i) => i !== index));
  };

  const updateBatchFeature = (index, feature, value) => {
    const updated = [...batchFeatures];
    updated[index] = { ...updated[index], [feature]: parseFloat(value) || 0 };
    setBatchFeatures(updated);
  };

  const renderPredictionResult = () => {
    if (!prediction) return null;

    if (activeTab === 'single') {
      return (
        <div className="card mt-6 border-2 border-primary-500">
          <h3 className="text-xl font-bold mb-4 flex items-center">
            <Sparkles className="w-6 h-6 mr-2 text-primary-500" />
            Prediction Result
          </h3>

          <div className="space-y-4">
            <div className="bg-dark-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Classification</p>
              <p className="text-2xl font-bold text-primary-400">
                {prediction.predicted_label || prediction.prediction}
              </p>
            </div>

            {prediction.confidence && (
              <div className="bg-dark-700 rounded-lg p-4">
                <p className="text-sm text-gray-400 mb-2">Confidence</p>
                <div className="flex items-center space-x-3">
                  <div className="flex-1 bg-dark-600 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-primary-500 to-purple-500 h-3 rounded-full transition-all"
                      style={{ width: `${prediction.confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-lg font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                </div>
              </div>
            )}

            {prediction.class_probabilities && (
              <div className="bg-dark-700 rounded-lg p-4">
                <p className="text-sm text-gray-400 mb-3">Class Probabilities</p>
                <div className="space-y-2">
                  {Object.entries(prediction.class_probabilities).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center justify-between">
                      <span className="text-sm">{cls}</span>
                      <div className="flex items-center space-x-2 flex-1 ml-4">
                        <div className="flex-1 bg-dark-600 rounded-full h-2">
                          <div
                            className="bg-primary-500 h-2 rounded-full"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                        <span className="text-sm w-16 text-right">{(prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      );
    }

    if (activeTab === 'batch' || activeTab === 'upload') {
      const predictions = prediction.predictions || [];
      return (
        <div className="card mt-6">
          <h3 className="text-xl font-bold mb-4">
            Batch Results ({predictions.length} predictions)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-dark-700">
                  <th className="text-left py-3 px-4">#</th>
                  <th className="text-left py-3 px-4">Prediction</th>
                  <th className="text-left py-3 px-4">Confidence</th>
                  <th className="text-left py-3 px-4">Class</th>
                </tr>
              </thead>
              <tbody>
                {predictions.slice(0, 100).map((pred, index) => (
                  <tr key={index} className="border-b border-dark-700 hover:bg-dark-700">
                    <td className="py-3 px-4">{index + 1}</td>
                    <td className="py-3 px-4">
                      <span className="px-3 py-1 bg-primary-500/20 text-primary-400 rounded-full text-sm">
                        {pred.predicted_label || pred.prediction}
                      </span>
                    </td>
                    <td className="py-3 px-4">
                      {pred.confidence ? (
                        <span className="text-sm">{(pred.confidence * 100).toFixed(1)}%</span>
                      ) : '-'}
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-400">
                      {pred.predicted_label || 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {predictions.length > 100 && (
              <p className="text-sm text-gray-400 mt-4 text-center">
                Showing first 100 of {predictions.length} predictions
              </p>
            )}
          </div>
        </div>
      );
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold gradient-text mb-2">Make Predictions</h1>
        <p className="text-gray-400">Classify exoplanet candidates using trained models</p>
      </div>

      {/* Tabs */}
      <div className="flex space-x-2 border-b border-dark-700">
        {[
          { id: 'single', label: 'Single Prediction' },
          { id: 'batch', label: 'Batch Prediction' },
          { id: 'upload', label: 'Upload CSV' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setActiveTab(tab.id);
              setPrediction(null);
              setError(null);
            }}
            className={`px-6 py-3 font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-primary-500 border-b-2 border-primary-500'
                : 'text-gray-400 hover:text-gray-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500 rounded-lg p-4 text-red-400">
          {error}
        </div>
      )}

      {/* Single Prediction */}
      {activeTab === 'single' && (
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Enter Feature Values</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {exampleFeatures.map((feature) => (
              <div key={feature}>
                <label className="label">{feature}</label>
                <input
                  type="number"
                  step="any"
                  className="input w-full"
                  value={singleFeatures[feature] || ''}
                  onChange={(e) =>
                    setSingleFeatures({ ...singleFeatures, [feature]: parseFloat(e.target.value) || 0 })
                  }
                  placeholder="0.0"
                />
              </div>
            ))}
          </div>
          <button
            onClick={handleSinglePredict}
            disabled={loading || Object.keys(singleFeatures).length === 0}
            className="btn-primary w-full"
          >
            {loading ? 'Predicting...' : 'Predict'}
          </button>
        </div>
      )}

      {/* Batch Prediction */}
      {activeTab === 'batch' && (
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-bold">Enter Multiple Samples</h2>
            <button onClick={addBatchRow} className="btn-secondary flex items-center space-x-2">
              <Plus className="w-4 h-4" />
              <span>Add Row</span>
            </button>
          </div>

          <div className="space-y-4 mb-6">
            {batchFeatures.map((row, index) => (
              <div key={index} className="bg-dark-700 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="font-semibold">Sample {index + 1}</h3>
                  {batchFeatures.length > 1 && (
                    <button
                      onClick={() => removeBatchRow(index)}
                      className="text-red-500 hover:text-red-400"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  )}
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {exampleFeatures.slice(0, 4).map((feature) => (
                    <input
                      key={feature}
                      type="number"
                      step="any"
                      className="input"
                      placeholder={feature}
                      onChange={(e) => updateBatchFeature(index, feature, e.target.value)}
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>

          <button
            onClick={handleBatchPredict}
            disabled={loading}
            className="btn-primary w-full"
          >
            {loading ? 'Predicting...' : `Predict ${batchFeatures.length} Samples`}
          </button>
        </div>
      )}

      {/* Upload CSV */}
      {activeTab === 'upload' && (
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Upload CSV File</h2>
          <div className="border-2 border-dashed border-dark-600 rounded-lg p-8 text-center hover:border-primary-500 transition-colors">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileUpload}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <Upload className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              <p className="text-lg mb-2">
                {uploadedFile ? uploadedFile.name : 'Click to upload CSV file'}
              </p>
              <p className="text-sm text-gray-400">
                File should contain the same features as training data
              </p>
            </label>
          </div>
        </div>
      )}

      {/* Results */}
      {renderPredictionResult()}
    </div>
  );
}

export default Predict;
