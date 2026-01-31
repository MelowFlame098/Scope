import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { login } from '../api/client';

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    
    try {
      const response = await login({ email, password });
      localStorage.setItem('token', response.token);
      localStorage.setItem('user_email', email); // Simple user storage
      navigate('/');
      window.location.reload(); // Force reload to update Navbar state
    } catch (err: any) {
      console.error('Login failed:', err);
      setError(err.response?.data?.error || 'Login failed. Please check your credentials.');
    }
  };

  return (
    <div className="glass-card" style={{ maxWidth: '400px', margin: '100px auto', padding: '2rem' }}>
      <h2 style={{ textAlign: 'center', marginBottom: '1.5rem', color: '#fff' }}>Welcome Back</h2>
      {error && <div style={{ color: '#ef4444', marginBottom: '1rem', textAlign: 'center', fontSize: '0.9rem' }}>{error}</div>}
      <form onSubmit={handleSubmit}>
        <div style={{ marginBottom: '1.25rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#a1a1aa', fontSize: '0.9rem' }}>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            style={{ 
              width: '100%', 
              padding: '0.75rem', 
              backgroundColor: 'rgba(0,0,0,0.2)',
              border: '1px solid #3f3f46',
              borderRadius: '6px',
              color: '#fff',
              outline: 'none',
              boxSizing: 'border-box'
            }}
            placeholder="Enter your email"
            required
          />
        </div>
        <div style={{ marginBottom: '1.5rem' }}>
          <label style={{ display: 'block', marginBottom: '0.5rem', color: '#a1a1aa', fontSize: '0.9rem' }}>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            style={{ 
              width: '100%', 
              padding: '0.75rem', 
              backgroundColor: 'rgba(0,0,0,0.2)',
              border: '1px solid #3f3f46',
              borderRadius: '6px',
              color: '#fff',
              outline: 'none',
              boxSizing: 'border-box'
            }}
            placeholder="Enter your password"
            required
          />
        </div>
        <button 
          type="submit" 
          style={{ 
            width: '100%', 
            padding: '0.75rem', 
            backgroundColor: '#8b5cf6', 
            color: '#fff', 
            border: 'none', 
            borderRadius: '6px', 
            fontWeight: 600,
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
          onMouseOver={(e) => e.currentTarget.style.backgroundColor = '#7c3aed'}
          onMouseOut={(e) => e.currentTarget.style.backgroundColor = '#8b5cf6'}
        >
          Sign In
        </button>
      </form>
    </div>
  );
};

export default Login;
