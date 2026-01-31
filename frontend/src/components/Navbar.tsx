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
      backgroundColor: '#333',
      color: '#fff',
      marginBottom: '20px'
    }}>
      <div style={{ fontSize: '1.5rem', fontWeight: 'bold' }}>
        <Link to="/" style={{ color: '#fff', textDecoration: 'none' }}>Scope</Link>
      </div>
      <ul style={{
        display: 'flex',
        listStyle: 'none',
        margin: 0,
        padding: 0,
        gap: '20px',
        alignItems: 'center'
      }}>
        <li>
          <Link to="/" style={{ color: '#fff', textDecoration: 'none' }}>Dashboard</Link>
        </li>
        {!token ? (
          <>
            <li>
              <Link to="/login" style={{ color: '#fff', textDecoration: 'none' }}>Login</Link>
            </li>
            <li>
              <Link to="/register" style={{ color: '#fff', textDecoration: 'none' }}>Register</Link>
            </li>
          </>
        ) : (
          <>
            <li style={{ color: '#ccc', fontSize: '0.9rem' }}>
              {userEmail}
            </li>
            <li>
              <button 
                onClick={handleLogout}
                style={{
                  background: 'none',
                  border: '1px solid #fff',
                  color: '#fff',
                  padding: '5px 10px',
                  borderRadius: '4px',
                  cursor: 'pointer'
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
