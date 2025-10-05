import { useState, useEffect } from 'react';
import { Database, RefreshCw, Download } from 'lucide-react';
import { listDatasets, getDatasetSample } from '../services/api';

function DataExplorer() {
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState('kepler');
  const [sampleData, setSampleData] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadDatasets();
  }, []);

  useEffect(() => {
    if (selectedDataset) {
      loadSampleData(selectedDataset);
    }
  }, [selectedDataset]);

  const loadDatasets = async () => {
    try {
      const data = await listDatasets();
      setDatasets(data);
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  const loadSampleData = async (datasetName) => {
    setLoading(true);
    try {
      const data = await getDatasetSample(datasetName, 20);
      setSampleData(data);
    } catch (error) {
      console.error('Error loading sample data:', error);
    } finally {
      setLoading(false);
    }
  };

  const currentDataset = datasets.find((ds) => ds.name === selectedDataset);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold gradient-text mb-2">Data Explorer</h1>
        <p className="text-gray-400">Explore exoplanet datasets</p>
      </div>

      {/* Dataset Selection */}
      <div className="flex space-x-4">
        {datasets.map((dataset) => (
          <button
            key={dataset.name}
            onClick={() => setSelectedDataset(dataset.name)}
            className={`px-6 py-3 rounded-lg font-medium transition-colors ${
              selectedDataset === dataset.name
                ? 'bg-primary-500 text-white'
                : 'bg-dark-700 text-gray-300 hover:bg-dark-600'
            }`}
          >
            {dataset.name.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Dataset Overview */}
      {currentDataset && (
        <div className="card">
          <h2 className="text-xl font-bold mb-4 flex items-center">
            <Database className="w-6 h-6 mr-2 text-primary-500" />
            {currentDataset.name.toUpperCase()} Dataset Overview
          </h2>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="bg-dark-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Total Rows</p>
              <p className="text-2xl font-bold text-primary-400">
                {currentDataset.rows.toLocaleString()}
              </p>
            </div>
            <div className="bg-dark-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Columns</p>
              <p className="text-2xl font-bold text-purple-400">{currentDataset.columns}</p>
            </div>
            <div className="bg-dark-700 rounded-lg p-4">
              <p className="text-sm text-gray-400 mb-1">Features</p>
              <p className="text-2xl font-bold text-green-400">{currentDataset.features.length}</p>
            </div>
            {currentDataset.target_distribution && (
              <div className="bg-dark-700 rounded-lg p-4">
                <p className="text-sm text-gray-400 mb-1">Classes</p>
                <p className="text-2xl font-bold text-blue-400">
                  {Object.keys(currentDataset.target_distribution).length}
                </p>
              </div>
            )}
          </div>

          {currentDataset.target_distribution && (
            <div className="mt-6">
              <h3 className="font-semibold mb-3">Target Distribution</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(currentDataset.target_distribution).map(([label, count]) => (
                  <div key={label} className="bg-dark-700 rounded-lg p-4">
                    <p className="text-sm text-gray-400 mb-1">{label}</p>
                    <p className="text-xl font-bold">{count.toLocaleString()}</p>
                    <p className="text-xs text-gray-500">
                      {((count / currentDataset.rows) * 100).toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sample Data */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-bold flex items-center">
            <RefreshCw className={`w-5 h-5 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Sample Data (20 rows)
          </h2>
          <button
            onClick={() => loadSampleData(selectedDataset)}
            className="btn-secondary text-sm"
          >
            Refresh
          </button>
        </div>

        {loading ? (
          <div className="flex items-center justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-500"></div>
          </div>
        ) : sampleData?.sample ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-dark-700">
                  {Object.keys(sampleData.sample[0] || {})
                    .slice(0, 10)
                    .map((key) => (
                      <th key={key} className="text-left py-3 px-4 font-semibold">
                        {key}
                      </th>
                    ))}
                </tr>
              </thead>
              <tbody>
                {sampleData.sample.map((row, index) => (
                  <tr key={index} className="border-b border-dark-700 hover:bg-dark-700">
                    {Object.values(row)
                      .slice(0, 10)
                      .map((value, i) => (
                        <td key={i} className="py-3 px-4">
                          {value === null || value === undefined
                            ? '-'
                            : typeof value === 'number'
                            ? value.toFixed(4)
                            : String(value).length > 30
                            ? String(value).substring(0, 30) + '...'
                            : String(value)}
                        </td>
                      ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-500 mt-4 text-center">
              Showing 10 of {Object.keys(sampleData.sample[0] || {}).length} columns
            </p>
          </div>
        ) : (
          <p className="text-gray-400 text-center py-8">No data available</p>
        )}
      </div>

      {/* Features List */}
      {currentDataset && (
        <div className="card">
          <h2 className="text-xl font-bold mb-4">Dataset Features</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {currentDataset.features.map((feature, index) => (
              <div
                key={index}
                className="bg-dark-700 rounded px-3 py-2 text-sm truncate"
                title={feature}
              >
                {feature}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default DataExplorer;
