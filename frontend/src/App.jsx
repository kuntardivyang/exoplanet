import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { useState, useEffect } from 'react';
import { Rocket, Activity, Database, Settings, BarChart3, MessageSquare } from 'lucide-react';
import Dashboard from './pages/Dashboard';
import Predict from './pages/Predict';
import Training from './pages/Training';
import DataExplorer from './pages/DataExplorer';
import Models from './pages/Models';
import ChatAssistant from './pages/ChatAssistant';
import { checkHealth } from './services/api';

function App() {
  const [apiStatus, setApiStatus] = useState('checking');
  const [currentPage, setCurrentPage] = useState('dashboard');

  useEffect(() => {
    const checkApiHealth = async () => {
      try {
        await checkHealth();
        setApiStatus('healthy');
      } catch (error) {
        setApiStatus('error');
      }
    };

    checkApiHealth();
    const interval = setInterval(checkApiHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-space">
        {/* Star field background */}
        <div className="star-field">
          {[...Array(100)].map((_, i) => (
            <div
              key={i}
              className="star"
              style={{
                width: Math.random() * 3 + 'px',
                height: Math.random() * 3 + 'px',
                top: Math.random() * 100 + '%',
                left: Math.random() * 100 + '%',
                animationDelay: Math.random() * 3 + 's',
              }}
            />
          ))}
        </div>

        {/* Header */}
        <header className="relative z-10 bg-dark-900/80 backdrop-blur-md border-b border-dark-700">
          <div className="container mx-auto px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <Rocket className="w-8 h-8 text-primary-500" />
                <div>
                  <h1 className="text-2xl font-bold gradient-text">Exoplanet Detector</h1>
                  <p className="text-sm text-gray-400">AI-Powered Discovery System</p>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    apiStatus === 'healthy' ? 'bg-green-500' :
                    apiStatus === 'error' ? 'bg-red-500' :
                    'bg-yellow-500'
                  } animate-pulse`} />
                  <span className="text-sm text-gray-400">
                    API {apiStatus === 'healthy' ? 'Online' : apiStatus === 'error' ? 'Offline' : 'Checking...'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </header>

        {/* Navigation */}
        <nav className="relative z-10 bg-dark-800/60 backdrop-blur-md border-b border-dark-700">
          <div className="container mx-auto px-6">
            <div className="flex space-x-1">
              {[
                { id: 'dashboard', icon: BarChart3, label: 'Dashboard', path: '/' },
                { id: 'predict', icon: Rocket, label: 'Predict', path: '/predict' },
                { id: 'training', icon: Activity, label: 'Training', path: '/training' },
                { id: 'data', icon: Database, label: 'Data Explorer', path: '/data' },
                { id: 'models', icon: Settings, label: 'Models', path: '/models' },
                { id: 'chat', icon: MessageSquare, label: 'AI Assistant', path: '/chat' },
              ].map((item) => (
                <Link
                  key={item.id}
                  to={item.path}
                  onClick={() => setCurrentPage(item.id)}
                  className={`flex items-center space-x-2 px-6 py-3 border-b-2 transition-colors ${
                    currentPage === item.id
                      ? 'border-primary-500 text-primary-500'
                      : 'border-transparent text-gray-400 hover:text-gray-200'
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  <span className="font-medium">{item.label}</span>
                </Link>
              ))}
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="relative z-10 container mx-auto px-6 py-8">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/training" element={<Training />} />
            <Route path="/data" element={<DataExplorer />} />
            <Route path="/models" element={<Models />} />
            <Route path="/chat" element={<ChatAssistant />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="relative z-10 bg-dark-900/80 backdrop-blur-md border-t border-dark-700 mt-20">
          <div className="container mx-auto px-6 py-6">
            <div className="text-center text-gray-400 text-sm">
              <p>NASA Space Apps Challenge 2025 - Exoplanet Detection System</p>
              <p className="mt-1">Built with React, FastAPI, and Machine Learning</p>
            </div>
          </div>
        </footer>
      </div>
    </Router>
  );
}

export default App;
