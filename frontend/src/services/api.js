import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Health check
export const checkHealth = async () => {
  const response = await api.get('/health');
  return response.data;
};

// Predictions
export const predictSingle = async (features, returnProbabilities = true) => {
  const response = await api.post('/predict', {
    features,
    return_probabilities: returnProbabilities,
  });
  return response.data;
};

export const predictBatch = async (samples, returnProbabilities = true) => {
  const response = await api.post('/predict/batch', {
    samples,
    return_probabilities: returnProbabilities,
  });
  return response.data;
};

export const predictUpload = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/predict/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// Models
export const listModels = async () => {
  const response = await api.get('/models');
  return response.data;
};

export const getModelInfo = async (modelName) => {
  const response = await api.get(`/models/${modelName}/info`);
  return response.data;
};

export const loadModel = async (modelName) => {
  const response = await api.post(`/models/load/${modelName}`);
  return response.data;
};

// Training
export const trainModel = async (dataset, modelTypes = null, hyperparameters = null) => {
  const response = await api.post('/train', {
    dataset,
    model_types: modelTypes,
    hyperparameters,
  });
  return response.data;
};

export const getTrainingStatus = async () => {
  const response = await api.get('/train/status');
  return response.data;
};

// Features
export const getFeatureImportance = async (topN = 20) => {
  const response = await api.get('/features/importance', {
    params: { top_n: topN },
  });
  return response.data;
};

// Datasets
export const listDatasets = async () => {
  const response = await api.get('/datasets');
  return response.data;
};

export const getDatasetSample = async (datasetName, n = 10) => {
  const response = await api.get(`/datasets/${datasetName}/sample`, {
    params: { n },
  });
  return response.data;
};

export default api;
