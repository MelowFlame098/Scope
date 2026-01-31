import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

const Navbar: React.FC = () => {
  const navigate = useNavigate();
  const token = localStorage.getItem('token');
  const userEmail = localStorage.getItem('user_email');

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user_email');
    navigate('/login');
  };

  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '1rem 2rem',
      backgroundColor: 'rgba(20, 20, 20, 0.8)', // Semi-transparent black
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
      color: '#fff',
      marginBottom: '20px',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
        <Link to="/" style={{ color: '#8b5cf6', textDecoration: 'none' }}>Scope</Link>
      </div>
      <ul style={{
        display: 'flex',
        listStyle: 'none',
        margin: 0,
        padding: 0,
        gap: '24px',
        alignItems: 'center'
      }}>
        <li>
          <Link to="/" style={{ color: '#fff', textDecoration: 'none', fontWeight: 500 }}>Dashboard</Link>
        </li>
        {!token ? (
          <>
            <li>
              <Link to="/login" style={{ color: '#a1a1aa', textDecoration: 'none' }}>Login</Link>
            </li>
            <li>
              <Link to="/register" style={{ 
                backgroundColor: '#8b5cf6', 
                color: '#fff', 
                padding: '8px 16px', 
                borderRadius: '6px', 
                textDecoration: 'none',
                fontWeight: 500
              }}>Register</Link>
            </li>
          </>
        ) : (
          <>
            <li style={{ color: '#a1a1aa', fontSize: '0.9rem' }}>
              {userEmail}
            </li>
            <li>
              <button 
                onClick={handleLogout}
                style={{
                  background: 'transparent',
                  border: '1px solid rgba(255,255,255,0.2)',
                  color: '#fff',
                  padding: '8px 16px',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  transition: 'all 0.2s'
                }}
              >
                Logout
              </button>
            </li>
          </>
        )}
      </ul>
    </nav>
  );
};

export default Navbar;
