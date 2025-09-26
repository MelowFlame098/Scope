import pytest
from httpx import AsyncClient
from unittest.mock import Mock, patch
import json

class TestAuthAPI:
    """Test authentication API endpoints."""
    
    @pytest.mark.asyncio
    async def test_login_success(self, async_client, mock_user_data):
        """Test successful user login."""
        login_data = {
            "email": "test@example.com",
            "password": "testpassword"
        }
        
        with patch('auth.authenticate_user') as mock_auth:
            mock_auth.return_value = mock_user_data
            
            response = await async_client.post(
                "/api/auth/login",
                json=login_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client):
        """Test login with invalid credentials."""
        login_data = {
            "email": "test@example.com",
            "password": "wrongpassword"
        }
        
        with patch('auth.authenticate_user') as mock_auth:
            mock_auth.return_value = None
            
            response = await async_client.post(
                "/api/auth/login",
                json=login_data
            )
            
            assert response.status_code == 401
            assert "Invalid credentials" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_register_success(self, async_client):
        """Test successful user registration."""
        register_data = {
            "email": "newuser@example.com",
            "username": "newuser",
            "password": "newpassword",
            "confirm_password": "newpassword"
        }
        
        with patch('auth.create_user') as mock_create:
            mock_create.return_value = {
                "id": 2,
                "email": "newuser@example.com",
                "username": "newuser"
            }
            
            response = await async_client.post(
                "/api/auth/register",
                json=register_data
            )
            
            assert response.status_code == 201
            data = response.json()
            assert data["email"] == "newuser@example.com"
            assert data["username"] == "newuser"
    
    @pytest.mark.asyncio
    async def test_register_existing_email(self, async_client):
        """Test registration with existing email."""
        register_data = {
            "email": "existing@example.com",
            "username": "newuser",
            "password": "newpassword",
            "confirm_password": "newpassword"
        }
        
        with patch('auth.get_user_by_email') as mock_get_user:
            mock_get_user.return_value = {"id": 1, "email": "existing@example.com"}
            
            response = await async_client.post(
                "/api/auth/register",
                json=register_data
            )
            
            assert response.status_code == 400
            assert "Email already registered" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_get_current_user(self, async_client, mock_user_data):
        """Test getting current user information."""
        with patch('auth.get_current_user') as mock_current_user:
            mock_current_user.return_value = mock_user_data
            
            headers = {"Authorization": "Bearer test-token"}
            response = await async_client.get(
                "/api/auth/me",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["email"] == mock_user_data["email"]
            assert data["username"] == mock_user_data["username"]
    
    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, async_client):
        """Test getting current user with invalid token."""
        with patch('auth.get_current_user') as mock_current_user:
            mock_current_user.side_effect = Exception("Invalid token")
            
            headers = {"Authorization": "Bearer invalid-token"}
            response = await async_client.get(
                "/api/auth/me",
                headers=headers
            )
            
            assert response.status_code == 401
    
    @pytest.mark.asyncio
    async def test_refresh_token(self, async_client):
        """Test token refresh functionality."""
        refresh_data = {
            "refresh_token": "valid-refresh-token"
        }
        
        with patch('auth.refresh_access_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new-access-token",
                "token_type": "bearer"
            }
            
            response = await async_client.post(
                "/api/auth/refresh",
                json=refresh_data
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_logout(self, async_client):
        """Test user logout functionality."""
        with patch('auth.revoke_token') as mock_revoke:
            mock_revoke.return_value = True
            
            headers = {"Authorization": "Bearer test-token"}
            response = await async_client.post(
                "/api/auth/logout",
                headers=headers
            )
            
            assert response.status_code == 200
            assert response.json()["message"] == "Successfully logged out"
    
    @pytest.mark.asyncio
    async def test_password_reset_request(self, async_client):
        """Test password reset request."""
        reset_data = {
            "email": "test@example.com"
        }
        
        with patch('auth.send_password_reset_email') as mock_send_reset:
            mock_send_reset.return_value = True
            
            response = await async_client.post(
                "/api/auth/password-reset",
                json=reset_data
            )
            
            assert response.status_code == 200
            assert "Password reset email sent" in response.json()["message"]
    
    @pytest.mark.asyncio
    async def test_password_reset_confirm(self, async_client):
        """Test password reset confirmation."""
        reset_data = {
            "token": "valid-reset-token",
            "new_password": "newpassword",
            "confirm_password": "newpassword"
        }
        
        with patch('auth.reset_password') as mock_reset:
            mock_reset.return_value = True
            
            response = await async_client.post(
                "/api/auth/password-reset/confirm",
                json=reset_data
            )
            
            assert response.status_code == 200
            assert "Password successfully reset" in response.json()["message"]